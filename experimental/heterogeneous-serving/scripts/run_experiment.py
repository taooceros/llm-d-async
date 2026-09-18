#!/usr/bin/env python3
"""
Unified Experiment Runner CLI.
==============================
Executes scheduling policy benchmarks across discrete simulation or Cloud TPU cluster
using the Reverse Control Mechanism (RCM).

Usage:
  python3 scripts/run_experiment.py --policy active
  python3 scripts/run_experiment.py --policy knapsack --workload data/workloads/sharegpt_sampled_500.json
  python3 scripts/run_experiment.py --policy my_policy.yaml --trace-output trace.json
  python3 scripts/run_experiment.py --policy knapsack --mode cluster --target-concurrency 96
"""

import os
import sys
import pathlib

# Ensure project root is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from admission_control.experiments.runner import main

if __name__ == "__main__":
    main()
