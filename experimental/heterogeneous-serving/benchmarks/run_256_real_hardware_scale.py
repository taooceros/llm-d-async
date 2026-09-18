#!/usr/bin/env python3
"""
256-Request Real Hardware Benchmark on Cloud TPU v5e (TP=8).
===========================================================
Executes all 256 requests from sharegpt_sampled_500.json
evaluating different batch sizes and concurrency caps:
1. Batch Size M=64 (4 cohorts of 64 requests)
2. Batch Size M=128 (2 cohorts of 128 requests - Larger Batch Size)
3. Slotted Continuous Pipeline (Cap=103 - QuantizedKnapsack-103)
4. Slotted Continuous Pipeline (Cap=128 - BS=128 Bucket Ceiling)
"""

import os
import sys
import time
import json
import argparse
import ray

def main():
    parser = argparse.ArgumentParser(description="256-Request Real Hardware Benchmark")
    parser.add_argument("--workload", type=str, default="data/workloads/sharegpt_sampled_500.json")
    parser.add_argument("--output-json", type=str, default="tp8_256_real_hardware_results.json")
    args = parser.parse_args()

    print("[Real HW 256] Connecting to Ray cluster...", flush=True)
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    worker = ray.get_actor("persistent_tpu_benchmark_worker", namespace="omp_benchmark")
    status = ray.get(worker.get_status.remote(), timeout=15)
    print(f"[Real HW 256] Persistent TPU Worker: {status}", flush=True)

    workload_file = args.workload
    if not os.path.exists(workload_file):
        workload_file = os.path.join(os.getcwd(), args.workload)
    print(f"[Real HW 256] Ingesting 256-request workload trace: {workload_file}...", flush=True)
    with open(workload_file, "r") as f:
        workload = json.load(f)[:256]

    total_demand = sum(r.get("prompt_len", 256) + r.get("decode_len", 2048) for r in workload)
    print(f"[Real HW 256] Workload: {len(workload)} requests, Total Demand: {total_demand:,} tokens (Capacity: 120,000 tokens)", flush=True)

    results = {
        "timestamp": time.time(),
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
    # CONFIG 1: Cohort Batching with M=64 (4 cohorts of 64 requests)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("[Real HW 256] >>> CONFIG 1: Cohort Batch Size M=64 (4 cohorts x 64) <<<", flush=True)
    print("=" * 80, flush=True)
    batches_64 = [workload[i:i+64] for i in range(0, len(workload), 64)]
    ray.get(worker.reset_kv_cache.remote())
    time.sleep(1.0)
    t0 = time.perf_counter()
    res_m64 = ray.get(worker.run_sustained_policy_a_greedy.remote(
        batch_stream=batches_64,
        max_duration_s=300.0,
    ))
    dur_m64 = time.perf_counter() - t0
    res_m64["measured_wall_time_s"] = round(dur_m64, 2)
    res_m64["batch_size"] = 64
    res_m64["num_batches"] = len(batches_64)
    results["runs"]["cohort_batch_size_64"] = res_m64
    print(f"[Real HW 256] Config 1 (M=64) Finished in {dur_m64:.2f}s | Result: {json.dumps(res_m64, indent=2)}", flush=True)

    # =========================================================================
    # CONFIG 2: Cohort Batch Size M=128 (2 cohorts of 128 requests - LARGER BATCH)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("[Real HW 256] >>> CONFIG 2: Cohort Batch Size M=128 (2 cohorts x 128 - LARGER BATCH) <<<", flush=True)
    print("=" * 80, flush=True)
    batches_128 = [workload[i:i+128] for i in range(0, len(workload), 128)]
    ray.get(worker.reset_kv_cache.remote())
    time.sleep(1.0)
    t0 = time.perf_counter()
    res_m128 = ray.get(worker.run_sustained_policy_a_greedy.remote(
        batch_stream=batches_128,
        max_duration_s=300.0,
    ))
    dur_m128 = time.perf_counter() - t0
    res_m128["measured_wall_time_s"] = round(dur_m128, 2)
    res_m128["batch_size"] = 128
    res_m128["num_batches"] = len(batches_128)
    results["runs"]["cohort_batch_size_128"] = res_m128
    print(f"[Real HW 256] Config 2 (M=128) Finished in {dur_m128:.2f}s | Result: {json.dumps(res_m128, indent=2)}", flush=True)

    # =========================================================================
    # CONFIG 3: Slotted Continuous Pipeline Cap=103 (QuantizedKnapsack-103)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("[Real HW 256] >>> CONFIG 3: Slotted Continuous Pipeline (Cap=103 - QuantizedKnapsack-103) <<<", flush=True)
    print("=" * 80, flush=True)
    ray.get(worker.reset_kv_cache.remote())
    time.sleep(1.0)
    t0 = time.perf_counter()
    res_c103 = ray.get(worker.run_policy_f_continuous_slotted_pipeline.remote(
        num_requests=len(workload),
        prompt_len=256,
        max_tokens=2048,
        target_concurrency=103,
        custom_requests=workload,
    ))
    dur_c103 = time.perf_counter() - t0
    res_c103["measured_wall_time_s"] = round(dur_c103, 2)
    res_c103["cap"] = 103
    results["runs"]["slotted_pipeline_cap_103"] = res_c103
    print(f"[Real HW 256] Config 3 (Cap=103) Finished in {dur_c103:.2f}s | Result: {json.dumps(res_c103, indent=2)}", flush=True)

    # =========================================================================
    # CONFIG 4: Slotted Continuous Pipeline Cap=128 (Knapsack-128 / BS=128 Ceiling)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("[Real HW 256] >>> CONFIG 4: Slotted Continuous Pipeline (Cap=128 - BS=128 Ceiling) <<<", flush=True)
    print("=" * 80, flush=True)
    ray.get(worker.reset_kv_cache.remote())
    time.sleep(1.0)
    t0 = time.perf_counter()
    res_c128 = ray.get(worker.run_policy_f_continuous_slotted_pipeline.remote(
        num_requests=len(workload),
        prompt_len=256,
        max_tokens=2048,
        target_concurrency=128,
        custom_requests=workload,
    ))
    dur_c128 = time.perf_counter() - t0
    res_c128["measured_wall_time_s"] = round(dur_c128, 2)
    res_c128["cap"] = 128
    results["runs"]["slotted_pipeline_cap_128"] = res_c128
    print(f"[Real HW 256] Config 4 (Cap=128) Finished in {dur_c128:.2f}s | Result: {json.dumps(res_c128, indent=2)}", flush=True)

    # Save results
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Real HW 256] All results successfully saved to {args.output_json}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("FINAL 256-REQUEST TPU v5e BENCHMARK SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(json.dumps(results, indent=2), flush=True)

if __name__ == "__main__":
    main()
