#!/usr/bin/env python3
"""
Real Hardware Policy Evaluation Runner on Cloud TPU v5e (TP=8).
=============================================================
Runs on the Ray cluster head node, attaching to the hot persistent
TPU worker actor holding Gemma-4 31B weights in HBM.
Evaluates:
1. Greedy Reactive Preemption (Baseline)
2. QuantizedKnapsack-103 / Slotted Pipeline (C=103)
3. Knapsack-128 / Slotted Pipeline (C=128)
Records empirical latency, goodput, preemptions, and token delivery.
"""

import os
import sys
import time
import json
import socket
import argparse
from typing import Dict, Any, List

import ray

def main():
    parser = argparse.ArgumentParser(description="Real Hardware Benchmark Runner")
    parser.add_argument("--workload", type=str, default="data/workloads/sharegpt_sampled_500.json")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--caps", type=str, default="103,128")
    parser.add_argument("--output-json", type=str, default="/tmp/real_hardware_policy_benchmark.json")
    args = parser.parse_args()

    print(f"[Real HW Bench] Initializing Ray connection...", flush=True)
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    print(f"[Real HW Bench] Connecting to persistent TPU worker actor...", flush=True)
    worker = ray.get_actor("persistent_tpu_benchmark_worker", namespace="omp_benchmark")
    status = ray.get(worker.get_status.remote(), timeout=15)
    print(f"[Real HW Bench] Worker Status: {status}", flush=True)

    # Load workload
    workload_path = args.workload
    if not os.path.exists(workload_path):
        # check alternative locations
        alt = os.path.join(os.getcwd(), workload_path)
        if os.path.exists(alt):
            workload_path = alt
    print(f"[Real HW Bench] Loading workload from {workload_path}...", flush=True)
    with open(workload_path, "r") as f:
        workload = json.load(f)[:args.num_requests]
    print(f"[Real HW Bench] Loaded {len(workload)} requests. Total demand: {sum(r.get('prompt_len', 256) + r.get('decode_len', 2048) for r in workload)} tokens.", flush=True)

    results = {
        "timestamp": time.time(),
        "hardware": status,
        "workload": {
            "path": workload_path,
            "requests": len(workload),
        },
        "benchmarks": {}
    }

    # 1. Run Greedy Reactive Baseline
    print("\n" + "=" * 80, flush=True)
    print("[Real HW Bench] >>> 1. Running Policy A: Greedy Baseline on Physical TPU v5e <<<", flush=True)
    print("=" * 80, flush=True)
    ray.get(worker.reset_kv_cache.remote())
    time.sleep(1.0)
    t0 = time.perf_counter()
    res_a = ray.get(worker.run_policy_a_greedy.remote(
        num_requests=len(workload),
        prompt_len=256,
        max_tokens=2048,
        custom_requests=workload,
    ))
    dur_a = time.perf_counter() - t0
    res_a["measured_wall_time_s"] = round(dur_a, 3)
    results["benchmarks"]["greedy_baseline"] = res_a
    print(f"[Real HW Bench] Greedy Baseline completed in {dur_a:.2f}s: {json.dumps(res_a, indent=2)}", flush=True)

    # 2. Run Candidate Concurrencies
    caps = [int(c.strip()) for c in args.caps.split(",") if c.strip()]
    for cap in caps:
        print("\n" + "=" * 80, flush=True)
        print(f"[Real HW Bench] >>> 2. Running Slotted Pipeline / Knapsack (Cap={cap}) on Physical TPU v5e <<<", flush=True)
        print("=" * 80, flush=True)
        ray.get(worker.reset_kv_cache.remote())
        time.sleep(1.0)
        t0 = time.perf_counter()
        res_cap = ray.get(worker.run_policy_f_continuous_slotted_pipeline.remote(
            num_requests=len(workload),
            prompt_len=256,
            max_tokens=2048,
            target_concurrency=cap,
            custom_requests=workload,
        ))
        dur_cap = time.perf_counter() - t0
        res_cap["measured_wall_time_s"] = round(dur_cap, 3)
        results["benchmarks"][f"knapsack_cap_{cap}"] = res_cap
        print(f"[Real HW Bench] Cap={cap} completed in {dur_cap:.2f}s: {json.dumps(res_cap, indent=2)}", flush=True)

    # Output final summary
    print("\n" + "=" * 80, flush=True)
    print("REAL CLOUD TPU v5e EMPIRICAL BENCHMARK RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(json.dumps(results, indent=2), flush=True)

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Real HW Bench] Wrote results to {args.output_json}", flush=True)

if __name__ == "__main__":
    main()
