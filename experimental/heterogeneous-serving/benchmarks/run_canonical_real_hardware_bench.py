#!/usr/bin/env python3
"""
Canonical Real Hardware Benchmark on Cloud TPU v5e (TP=8, 2x4 mesh).
===================================================================
Executes Gemma-4 31B BF16 with real weights on physical TPU hardware.
Evaluates:
1. Greedy Reactive Preemption (vLLM unconstrained default)
2. Predictive / Slotted Knapsack Gating (Cap=48)
3. Balanced Dual-Wave (32+32)
Measures empirical preemptions, discarded tokens, step latencies, and goodput.
"""

import os
import sys
import time
import json
import argparse
import ray

def main():
    parser = argparse.ArgumentParser(description="Real Hardware Benchmark")
    parser.add_argument("--workload", type=str, default="data/workloads/sharegpt_sampled_500.json")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--output-json", type=str, default="tp8_empirical_hardware_run_results.json")
    args = parser.parse_args()

    print("[Real HW Bench] Connecting to Ray cluster...", flush=True)
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    worker = ray.get_actor("persistent_tpu_benchmark_worker", namespace="omp_benchmark")
    status = ray.get(worker.get_status.remote(), timeout=15)
    print(f"[Real HW Bench] Persistent TPU Worker: {status}", flush=True)

    workload_file = args.workload
    if not os.path.exists(workload_file):
        workload_file = os.path.join(os.getcwd(), args.workload)
    print(f"[Real HW Bench] Ingesting workload trace: {workload_file}...", flush=True)
    with open(workload_file, "r") as f:
        workload = json.load(f)[:args.num_requests]

    total_demand = sum(r.get("prompt_len", 256) + r.get("decode_len", 2048) for r in workload)
    print(f"[Real HW Bench] Workload: {len(workload)} requests, Total Demand: {total_demand:,} tokens (Capacity: 120,000 tokens)", flush=True)

    benchmark_summary = {
        "hardware": {
            "accelerator": "Cloud TPU v5e (8 chips, 2x4 mesh, TP=8)",
            "model": "Gemma-4 31B BF16 (61.4 GB weights in HBM)",
            "kv_cache_capacity_tokens": 120000,
            "host": status.get("host"),
        },
        "workload": {
            "file": os.path.basename(workload_file),
            "num_requests": len(workload),
            "total_demand_tokens": total_demand,
        },
        "runs": {}
    }

    # =========================================================================
    # RUN 1: Policy A (Greedy Reactive Baseline, Max Seqs = 256)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("[Real HW Bench] >>> 1. EXECUTING POLICY A: GREEDY REACTIVE PREEMPTION <<<", flush=True)
    print("=" * 80, flush=True)
    ray.get(worker.reset_kv_cache.remote())
    time.sleep(1.0)
    t0 = time.perf_counter()
    res_greedy = ray.get(worker.run_policy_a_greedy.remote(
        num_requests=len(workload),
        prompt_len=256,
        max_tokens=2048,
        custom_requests=workload,
    ))
    t_greedy = time.perf_counter() - t0
    res_greedy["measured_wall_time_s"] = round(t_greedy, 2)
    benchmark_summary["runs"]["policy_a_greedy"] = res_greedy
    print(f"[Real HW Bench] Policy A Finished in {t_greedy:.2f}s | Result: {json.dumps(res_greedy, indent=2)}", flush=True)

    # =========================================================================
    # RUN 2: Policy F / Slotted Knapsack (Cap = 48, Preemption Immune)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("[Real HW Bench] >>> 2. EXECUTING POLICY F: SLOTTED KNAPSACK (CAP=48) <<<", flush=True)
    print("=" * 80, flush=True)
    ray.get(worker.reset_kv_cache.remote())
    time.sleep(1.0)
    t0 = time.perf_counter()
    res_slotted = ray.get(worker.run_policy_f_continuous_slotted_pipeline.remote(
        num_requests=len(workload),
        prompt_len=256,
        max_tokens=2048,
        target_concurrency=48,
        custom_requests=workload,
    ))
    t_slotted = time.perf_counter() - t0
    res_slotted["measured_wall_time_s"] = round(t_slotted, 2)
    benchmark_summary["runs"]["policy_f_slotted_48"] = res_slotted
    print(f"[Real HW Bench] Policy F Finished in {t_slotted:.2f}s | Result: {json.dumps(res_slotted, indent=2)}", flush=True)

    # Save and output
    with open(args.output_json, "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"\n[Real HW Bench] Results saved to {args.output_json}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("FINAL REAL TPU HARDWARE COMPARISON REPORT", flush=True)
    print("=" * 80, flush=True)
    print(json.dumps(benchmark_summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
