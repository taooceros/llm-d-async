#!/usr/bin/env python3
"""
Canonical Scheduling Policy Benchmark Harness.
=============================================
Evaluates scheduling policies on real empirical workloads under hardware-exact
Cloud TPU v5e constraints:
- Batch window M=64 requests (strictly partitioned cohorts, preserving inter-batch precedence).
- Reordering permitted only within individual batches of 64 requests.
- Strict preemption immunity: requests drawn from B_{k+1} must never cause preemption in B_k.
- Static XLA compilation buckets (16, 32, 64, 128, 256) with empirical hardware step latencies.
- Absolute preemption immunity: num_preemptions must equal 0.
"""

import os
import sys
import json
import time
import pathlib
import argparse
from typing import Dict, Any, List

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from admission_control.core.models import HardwareSpec, Request
from admission_control.algorithms.policies import (
    get_policy,
    get_active_policy,
    DynamicKnapsackPolicy,
    BasePolicy,
)
from admission_control.algorithms.executor import PolicyExecutor, SimulatedExecutionEngine


def run_benchmark(
    workload_path: str,
    batch_size: int = 64,
    capacity_tokens: int = 120_000,
    policy_name: str = "active",
    json_output: str = None,
) -> Dict[str, Any]:
    from admission_control.experiments.config import ExperimentConfig
    from admission_control.experiments.runner import PolicyExperimentRunner

    cfg = ExperimentConfig(
        policy=policy_name,
        workload=workload_path,
        batch_size=batch_size,
        capacity_tokens=capacity_tokens,
        baseline_policy="greedy",
        output_json=json_output,
    )
    runner = PolicyExperimentRunner(cfg)
    payload = runner.run()

    metrics = payload.get("metrics", {})
    if not metrics.get("preemption_immunity_verified", 1):
        print("ERROR: Policy violated preemption immunity! Preemptions occurred.", file=sys.stderr)
        sys.exit(1)
    if not metrics.get("starvation_freedom_verified", 1):
        print("ERROR: Policy violated starvation freedom! Not all requests/batches completed.", file=sys.stderr)
        sys.exit(1)

    return payload

def main():
    parser = argparse.ArgumentParser(description="Canonical Scheduling Policy Benchmark Harness")
    parser.add_argument(
        "--workload",
        type=str,
        default="data/workloads/sharegpt_sampled_500.json",
        help="Path to workload JSON file",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch cohort window size M (default 64)")
    parser.add_argument("--capacity", type=int, default=120_000, help="KV cache token capacity (default 120,000)")
    parser.add_argument("--policy", type=str, default="active", help="Policy name to evaluate (default: 'active')")
    parser.add_argument(
        "--output-json",
        type=str,
        default="experiments/tp8_preemption_benchmark/results/policy_benchmark_results.json",
        help="Path to save results JSON",
    )
    args = parser.parse_args()

    run_benchmark(
        workload_path=args.workload,
        batch_size=args.batch_size,
        capacity_tokens=args.capacity,
        policy_name=args.policy,
        json_output=args.output_json,
    )


if __name__ == "__main__":
    main()
