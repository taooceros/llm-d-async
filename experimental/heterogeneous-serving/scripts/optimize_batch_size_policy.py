#!/usr/bin/env python3
# /// script
# dependencies = [
#     "ortools>=9.8.0",
#     "numpy>=1.24.0",
#     "matplotlib>=3.7.0",
# ]
# ///
"""
Batch Size Policy Optimizer for Accelerators (Cloud TPU v5e / GPU).
===================================================================
Finds the provably optimal scheduling policy by maximizing the average batch size
(and equivalently minimizing makespan) under physical KV cache HBM capacity limits
and XLA compilation constraints.

Mathematical Foundations & NP-Hardness:
---------------------------------------
1. Average Batch Size Equivalence:
   bar{B} = (1 / T) * sum_{t=0}^{T-1} B(t) = D_total / T
   Because total decode tokens D_total is invariant, maximizing average batch size
   is mathematically identical to makespan minimization (T_min).
2. NP-Hardness:
   Scheduling dynamic trapezoidal memory profiles M_i(t) = P_i + (t - s_i + 1)
   under peak memory and concurrency ceilings is strongly NP-hard (reduction from RCPSP).
3. Optimization Paradigms:
   - Theoretical Upper Bound Derivation (Critical Path, Volume, and Memory Deficit)
   - Exact Integer Linear Programming (ILP) via OR-Tools CP-SAT
   - Exact Branch-and-Bound Permutation Search with Analytical Lower Bounding
   - Combinatorial Policy Space Optimizer under the M=64 Batch-Bounded Contract

Usage:
  python3 scripts/optimize_batch_size_policy.py --workload data/workloads/sharegpt_sampled_500.json
  python3 scripts/optimize_batch_size_policy.py --num-requests 16 --method bnb
  python3 scripts/optimize_batch_size_policy.py --num-requests 32 --method ilp --time-limit 20
  python3 scripts/optimize_batch_size_policy.py --workload sharegpt_sampled_500.json --method all
"""

import os
import sys
import json
import time
import argparse
import pathlib

# Ensure workspace root is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
from dataclasses import asdict
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from admission_control.core.models import Request, HardwareSpec
from admission_control.workloads.loader import load_workload_requests
from admission_control.algorithms.batch_size_optimizer import (
    compute_theoretical_bounds,
    simulate_schedule_metrics,
    ILPBatchSizeOptimizer,
    BranchAndBoundBatchSizeOptimizer,
    PolicyBatchSizeOptimizer,
    BatchSizeOptimizationSuite,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find optimal scheduling policy maximizing average batch size on Cloud TPU / GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="data/workloads/sharegpt_sampled_500.json",
        help="Path to workload JSON file (or name in data/workloads/)",
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
        help="Cap the number of requests evaluated (e.g. 16 for exact BnB, 32 for fast ILP)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "ilp", "bnb", "policy-search", "baseline-only"],
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

    print("\n" + "=" * 90)
    print("CLOUDTOOLS OPTIMAL SCHEDULING POLICY: AVERAGE BATCH SIZE OPTIMIZER")
    print("=" * 90)
    print("Optimization Objective : Maximize Average Batch Size bar{B} = D_total / T")
    print("Mathematical Invariant : Maximizing bar{B} is strictly equivalent to makespan minimization T_min")
    print(f"Hardware Constraints   : KV Cache Capacity = {args.capacity:,} tokens | Max Concurrency = {args.max_batch_size}")
    print(f"Batch Contract         : Cohort Window M = {args.batch_window} | Inter-Batch Precedence Strictly Enforced")
    print("=" * 90 + "\n")

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

    # Configure suite options
    run_ilp = args.method in ["all", "ilp"]
    run_bnb = args.method in ["all", "bnb"]
    run_policy_search = args.method in ["all", "policy-search"]

    suite = BatchSizeOptimizationSuite(hw=hw, batch_window_m=args.batch_window)
    print("\n[Optimizer] Running optimization suite across analytical and algorithmic paradigms...")
    results = suite.run_suite(
        requests=requests,
        run_ilp=run_ilp,
        run_bnb=run_bnb,
        run_policy_search=run_policy_search,
        ilp_time_limit_s=args.time_limit,
        bnb_time_limit_s=args.time_limit,
        step_delta=args.step_delta,
    )

    # Print comparative results
    print(suite.format_comparison_table(results))

    # Print markdown table if requested
    if args.markdown:
        print("### Markdown Summary Table:\n")
        print("| Strategy / Policy | Method | Avg Batch Size | Makespan | Goodput (tok/s) | Preemptions | Theoretical Efficiency |")
        print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
        print(f"| **Theoretical Upper Bound** | *Analytic Ceiling* | **{bounds.theoretical_max_avg_batch:.2f}** | **{bounds.theoretical_min_makespan:,}** | — | 0 | 100.0% |")
        for m in results["evaluations"]:
            print(f"| {m.policy_name} | {m.method} | {m.avg_batch_size:.2f} | {m.makespan_steps:,} | {m.goodput_tok_per_s:,.1f} | {m.preemptions} | {m.theoretical_efficiency_pct:.1f}% |")
        print("\n")

    # Export results if requested
    if args.output:
        out_data = {
            "bounds": asdict(bounds),
            "evaluations": [m.to_dict() for m in results["evaluations"]],
            "best_policy": results["best_schedule"].to_dict() if results["best_schedule"] else None,
            "total_runtime_s": results["total_runtime_s"],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"[Export] Saved complete optimization report and metrics to '{args.output}'")

    print("[SUCCESS] Batch size optimization completed successfully.")


if __name__ == "__main__":
    main()
