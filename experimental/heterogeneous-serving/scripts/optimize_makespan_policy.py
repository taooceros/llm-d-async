#!/usr/bin/env python3
# /// script
# dependencies = [
#     "ortools>=9.8.0",
#     "numpy>=1.24.0",
#     "matplotlib>=3.7.0",
# ]
# ///
"""
Makespan Policy Optimizer for Cloud TPUs & Distributed LLM Serving.
===================================================================
Finds the provably optimal scheduling policy minimizing execution makespan
under hard accelerator HBM capacity limits and static XLA compilation buckets.

Supports dual makespan objectives:
1. Discrete Step Makespan (T_steps = max_i(s_i + D_i)):
   Minimizes discrete decode steps across the serving engine.
   Mathematically strictly equivalent to maximizing average batch size (D_total / T).
2. Wall-Clock Execution Makespan (T_wall = sum_t L(B(t)) in seconds):
   Minimizes physical execution time on Cloud TPU v5e, directly modeling
   non-linear XLA static shape buckets and eliminating tensor padding waste.

Usage:
  python3 scripts/optimize_makespan_policy.py --workload data/workloads/sharegpt_sampled_500.json --objective wall-time
  python3 scripts/optimize_makespan_policy.py --workload sharegpt_sampled_500.json --objective step-makespan
  python3 scripts/optimize_makespan_policy.py --num-requests 16 --method bnb
  python3 scripts/optimize_makespan_policy.py --num-requests 32 --method ilp --objective wall-time
"""

import os
import sys
import json
import time
import argparse
import pathlib
from dataclasses import asdict

# Ensure workspace root is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from admission_control.core.models import Request, HardwareSpec
from admission_control.workloads.loader import load_workload_requests
from admission_control.algorithms.batch_size_optimizer import (
    compute_theoretical_bounds,
    simulate_schedule_metrics,
    MakespanOptimizationSuite,
    ILPMakespanOptimizer,
    BranchAndBoundMakespanOptimizer,
    PolicyMakespanOptimizer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find optimal scheduling policy minimizing makespan on Cloud TPU / GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="data/workloads/sharegpt_sampled_500.json",
        help="Path to workload JSON file (or name in data/workloads/)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="wall-time",
        choices=["wall-time", "step-makespan", "avg-batch-size"],
        help="Optimization target: 'wall-time' (physical seconds), 'step-makespan' (discrete steps), or 'avg-batch-size'",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=120_000,
        help="KV cache HBM token capacity (default: 120,000 for TPU v5e TP=8)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Maximum hardware batch concurrency ceiling (default: 128)",
    )
    parser.add_argument(
        "--batch-window",
        type=int,
        default=64,
        help="Batch cohort window size M (default: 64)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Cap the number of requests evaluated",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "ilp", "bnb", "policy-search"],
        help="Optimization method: 'ilp' (CP-SAT), 'bnb' (Branch-and-Bound), 'policy-search', or 'all'",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="Time limit for exact solvers (ILP / BnB) in seconds",
    )
    parser.add_argument(
        "--step-delta",
        type=int,
        default=1,
        help="Time discretization step delta for ILP solver",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to export JSON results and schedule plan",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print results table as Markdown",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    obj_name = {
        "wall-time": "Physical Wall-Clock Time (Seconds, XLA-Bucket Aware)",
        "step-makespan": "Discrete Step Count (T_steps = max_i(s_i + D_i))",
        "avg-batch-size": "Average Batch Size Concurrency (D_total / T)",
    }.get(args.objective, args.objective)

    print("\n" + "=" * 95)
    print("CLOUDTOOLS OPTIMAL SCHEDULING POLICY: MAKESPAN OPTIMIZER")
    print("=" * 95)
    print(f"Optimization Objective : {obj_name}")
    print(f"Hardware Constraints   : KV Cache Capacity = {args.capacity:,} tokens | Max Concurrency = {args.max_batch_size}")
    print(f"Batch Contract         : Cohort Window M = {args.batch_window} | Inter-Batch Precedence Strictly Enforced")
    print("=" * 95 + "\n")

    # Load workload
    workload_path = args.workload
    if not os.path.exists(workload_path):
        alt_path = PROJECT_ROOT / "data" / "workloads" / args.workload
        if alt_path.exists():
            workload_path = str(alt_path)
        else:
            alt_path_json = PROJECT_ROOT / "data" / "workloads" / f"{args.workload}.json"
            if alt_path_json.exists():
                workload_path = str(alt_path_json)

    requests = load_workload_requests(workload_path, max_requests=args.num_requests)
    print(f"[Workload] Loaded {len(requests)} requests from '{workload_path}'")
    tot_p = sum(r.prompt_len for r in requests)
    tot_d = sum(r.decode_len for r in requests)
    print(f"           Total Prompt: {tot_p:,} tok | Total Decode: {tot_d:,} tok | Total Demand: {tot_p + tot_d:,} tok\n")

    hw = HardwareSpec(capacity_tokens=args.capacity, max_batch_size=args.max_batch_size)

    # 1. Compute and print theoretical bounds
    bounds = compute_theoretical_bounds(requests, hw)
    print(bounds.summary())

    run_ilp = args.method in ["all", "ilp"]
    run_bnb = args.method in ["all", "bnb"]
    run_policy_search = args.method in ["all", "policy-search"]

    suite = MakespanOptimizationSuite(hw=hw, batch_window_m=args.batch_window)
    print(f"\n[Optimizer] Running makespan optimization suite (Objective: {args.objective})...")
    results = suite.run_suite(
        requests=requests,
        run_ilp=run_ilp,
        run_bnb=run_bnb,
        run_policy_search=run_policy_search,
        ilp_time_limit_s=args.time_limit,
        bnb_time_limit_s=args.time_limit,
        step_delta=args.step_delta,
        objective=args.objective,
    )

    # Print comparative results
    print(suite.format_comparison_table(results))

    # Print markdown table if requested
    if args.markdown:
        print("### Markdown Summary Table:\n")
        print("| Strategy / Policy | Method | Makespan Steps | Wall Time (s) | Avg Batch Size | Goodput (tok/s) | Preemptions |")
        print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
        print(f"| **Theoretical Lower Bound** | *Analytic Ceiling* | **{bounds.theoretical_min_makespan:,}** | — | **{bounds.theoretical_max_avg_batch:.2f}** | — | 0 |")
        for m in results["evaluations"]:
            print(f"| {m.policy_name} | {m.method} | {m.makespan_steps:,} | {m.total_time_s:.2f} s | {m.avg_batch_size:.2f} | {m.goodput_tok_per_s:,.1f} | {m.preemptions} |")
        print("\n")

    # Export results if requested
    if args.output:
        out_data = {
            "objective": args.objective,
            "bounds": asdict(bounds),
            "evaluations": [m.to_dict() for m in results["evaluations"]],
            "best_policy": results["best_schedule"].to_dict() if results["best_schedule"] else None,
            "total_runtime_s": results["total_runtime_s"],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"[Export] Saved complete makespan optimization report to '{args.output}'")

    print("[SUCCESS] Makespan optimization completed successfully.")


if __name__ == "__main__":
    main()
