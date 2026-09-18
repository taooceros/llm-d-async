#!/usr/bin/env python3
"""
Policy Testing & Comparison CLI.
================================
Quickly test, compare, and sweep scheduling policies under hardware invariants.

Usage:
  # 1. Test a single policy
  python3 scripts/test_policy.py --policy active
  python3 scripts/test_policy.py --policy my_policy.yaml

  # 2. Compare multiple policies side by side
  python3 scripts/test_policy.py --compare greedy knapsack predictive

  # 3. Parameter sweep
  python3 scripts/test_policy.py --sweep cap --values 64 80 96 103 112 128

  # 4. Quick smoke evaluation
  python3 scripts/test_policy.py --quick --policy knapsack
"""

import os
import sys
import argparse
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from admission_control.algorithms.policies import get_policy, PolicyBuilder, Priority, Guard, BatchTransition
from admission_control.experiments.testing import (
    test_policy,
    compare_policies,
    sweep_policy,
    quick_eval,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test, compare, and sweep scheduling policies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", type=str, default="active", help="Policy to test (name, .yaml file, or 'active')")
    parser.add_argument("--compare", nargs="+", default=None, help="List of policies to compare side by side")
    parser.add_argument("--sweep", type=str, default=None, choices=["cap", "weight"], help="Parameter to sweep")
    parser.add_argument("--values", nargs="+", default=None, help="Values for parameter sweep")
    parser.add_argument("--workload", type=str, default="data/workloads/sharegpt_sampled_500.json", help="Workload JSON file or standard name")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch cohort window size M")
    parser.add_argument("--capacity", type=int, default=120_000, help="KV cache token capacity")
    parser.add_argument("--quick", action="store_true", help="Run quick 64-request evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick:
        print(f"[Testing] Running quick evaluation of '{args.policy}'...")
        res = quick_eval(args.policy, capacity_tokens=args.capacity)
        passed = "PASSED" if res["passed"] else "FAILED"
        print(f"Result: {passed} | Goodput: {res['goodput_tok_per_s']:,.1f} tok/s | Preemptions: {res['num_preemptions']}")
        sys.exit(0 if res["passed"] else 1)

    if args.compare:
        print(f"[Testing] Comparing policies: {args.compare}...")
        res = compare_policies(
            policies=args.compare,
            workload=args.workload,
            batch_size=args.batch_size,
            capacity_tokens=args.capacity,
        )
        sys.exit(0)

    if args.sweep:
        if not args.values:
            print("ERROR: --values required when using --sweep", file=sys.stderr)
            sys.exit(1)

        if args.sweep == "cap":
            caps = [int(v) for v in args.values]
            def factory(c):
                return (
                    PolicyBuilder(f"Knapsack-Cap-{c}")
                    .with_priority(Priority.slack_packed_lpt(weight=0.0010))
                    .admit_if(Guard.concurrency_cap(c))
                    .admit_if(Guard.preemption_immune(capacity=args.capacity, block_size=16))
                    .draw_next_batch_if(BatchTransition.preemption_immune(capacity=args.capacity, target_concurrency=c, block_size=16))
                    .build()
                )
            sweep_policy(factory, "concurrency_cap", caps, workload=args.workload, capacity_tokens=args.capacity)
        elif args.sweep == "weight":
            weights = [float(v) for v in args.values]
            def factory(w):
                return (
                    PolicyBuilder(f"Knapsack-W-{w}")
                    .with_priority(Priority.slack_packed_lpt(weight=w))
                    .admit_if(Guard.concurrency_cap(103))
                    .admit_if(Guard.preemption_immune(capacity=args.capacity, block_size=16))
                    .draw_next_batch_if(BatchTransition.preemption_immune(capacity=args.capacity, target_concurrency=103, block_size=16))
                    .build()
                )
            sweep_policy(factory, "priority_weight", weights, workload=args.workload, capacity_tokens=args.capacity)
        sys.exit(0)

    # Default: test single policy
    print(f"[Testing] Evaluating policy '{args.policy}'...")
    res = test_policy(
        policy=args.policy,
        workload=args.workload,
        batch_size=args.batch_size,
        capacity_tokens=args.capacity,
    )
    passed = "PASSED" if res["passed"] else "FAILED"
    print(f"\nFinal Result: {passed}")
    print(f"Policy Name : {res['policy_name']}")
    print(f"Goodput     : {res['goodput_tok_per_s']:,.1f} tok/s ({res['speedup']:.4f}x vs baseline)")
    print(f"Preemptions : {res['num_preemptions']} (Immunity: {'YES' if res['preemption_immune'] else 'NO'})")
    print(f"Starvation  : {'FREE' if res['starvation_free'] else 'VIOLATED'}\n")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
