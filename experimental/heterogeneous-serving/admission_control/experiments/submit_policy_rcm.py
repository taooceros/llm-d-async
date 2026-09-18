#!/usr/bin/env python3
"""
Reverse Control Mechanism (RCM) Cloud Policy Benchmark Submitter.
================================================================
Submits and benchmarks scheduling policies on Cloud TPU v5e (TP=8, 2x4 mesh)
using the Ray Job Submission SDK (JobSubmissionClient).

Delegates execution to the unified `admission_control.experiments` framework,
preserving 100% backward compatibility with CLI options and RCM in-flight steering.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Optional

# Ensure project root is in sys.path
EXPERIMENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from admission_control.experiments.config import ExperimentConfig
from admission_control.experiments.runner import PolicyExperimentRunner
from admission_control.experiments.plotting import generate_all_plots
from admission_control.experiments.reporting import build_html_report, build_markdown_report


DEFAULT_RAY_URL = "http://127.0.0.1:8265"
DEFAULT_BATCH_SIZE = 64
DEFAULT_KV_CACHE_CAPACITY = 120_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reverse Control Mechanism (RCM) Cloud Policy Benchmark Submitter",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="knapsack",
        help="Scheduling policy to submit and evaluate (e.g. 'knapsack', 'predictive', 'wave', 'greedy', 'oracle', '3_policies', 'all', 'both')",
    )
    parser.add_argument(
        "--workload-json",
        type=str,
        default=None,
        help="Path to empirical workload JSON trace (e.g. data/workloads/sharegpt_sampled_500.json)",
    )
    parser.add_argument(
        "--target-concurrency",
        type=int,
        default=96,
        help="Target concurrency for oversubscribed execution into BS=128 XLA bucket (e.g. 64, 88, 96, 104, 112, 128)",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Sustained multi-batch benchmark duration in seconds (e.g. 120.0 for 2-minute sustained run)",
    )
    parser.add_argument(
        "--ray-url",
        type=str,
        default=DEFAULT_RAY_URL,
        help="Ray cluster dashboard/job submission API URL",
    )
    parser.add_argument(
        "--force-reinit",
        action="store_true",
        help="Force destroy existing persistent TPU worker and placement group to re-initialize from scratch",
    )
    parser.add_argument(
        "--silence-timeout-s",
        type=float,
        default=600.0,
        help="RCM watchdog silence timeout threshold in seconds",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EXPERIMENT_DIR / "results"),
        help="Output directory for empirical JSON measurement artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("\n" + "=" * 85, flush=True)
    print(">>> OMP REVERSE CONTROL MECHANISM (RCM) CLOUD POLICY SUBMITTER <<<", flush=True)
    print(f"    Policy Candidate:   {args.policy}", flush=True)
    print(f"    Target Concurrency: {args.target_concurrency}", flush=True)
    print(f"    Ray Cluster URL:    {args.ray_url}", flush=True)
    print(f"    Workload Trace:     {args.workload_json or 'Canonical Empirical Trace'}", flush=True)
    print(f"    Sustained Duration: {args.duration_s}s", flush=True)
    print("=" * 85 + "\n", flush=True)

    out_dir = pathlib.Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"tp8_preemption_benchmark_results.json"
    figures_dir = EXPERIMENT_DIR / "reports" / "figures"
    out_md = EXPERIMENT_DIR / "reports" / "tp8_preemption_benchmark_report.md"
    out_html = EXPERIMENT_DIR / "reports" / "tp8_preemption_benchmark_report.html"

    workload = args.workload_json or "data/workloads/sharegpt_sampled_500.json"

    cfg = ExperimentConfig(
        policy=args.policy,
        workload=workload,
        batch_size=DEFAULT_BATCH_SIZE,
        capacity_tokens=DEFAULT_KV_CACHE_CAPACITY,
        target_concurrency=args.target_concurrency,
        max_duration_s=args.duration_s if args.duration_s > 0 else None,
        execution_mode="cluster",
        output_json=str(out_json),
        ray_url=args.ray_url,
        force_reinit=args.force_reinit,
        silence_timeout_s=args.silence_timeout_s,
        generate_plots=True,
        figures_dir=str(figures_dir),
        output_report=str(out_md),
    )

    runner = PolicyExperimentRunner(config=cfg)
    try:
        results = runner.run()
        print(f"[RCM Submitter] Cluster experiment completed successfully.", flush=True)
        return 0
    except Exception as e:
        print(f"[RCM Submitter] Experiment execution error: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
