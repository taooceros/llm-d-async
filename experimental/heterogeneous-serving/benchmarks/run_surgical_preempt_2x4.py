#!/usr/bin/env python3
"""
Continuous Surgical Preemption 2x4 Benchmark on Cloud TPU v5e (TP=8).
===================================================================
Executes real conversational workload evaluation on physical Cloud TPU v5e hardware
(2x4 subslice: 192.168.102.18 and 192.168.102.19, TP=8) using Gemma-4 31B BF16
holding 61.4 GB weights in HBM.

Evaluates 3 real-hardware policies on 256 requests (total demand 375,173 tokens):
1. Baseline Cohort Batching (C=64, M=64)
2. Slotted Continuous Pipeline (C=103, QuantizedKnapsack-103)
3. Continuous Surgical Preempt with Memory Cushion (Delta M_pad = 2,500 tokens, Target C=128, Max-Yield)

Outputs empirical latency, goodput, preemption counts, and peak memory to
results/surgical_preempt_2x4_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
from typing import Any, Dict, List, Optional

import ray
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)

from admission_control.tpu.placement import cleanup_stale_placement_groups
from admission_control.serving.worker import TPUPreemptionBenchmarkWorker, emit_omp_event


PERSISTENT_ACTOR_NAME = "persistent_tpu_benchmark_worker"
PERSISTENT_NAMESPACE = "omp_benchmark"
TARGET_HOST_IPS = ["192.168.102.18", "192.168.102.19"]


def setup_or_get_tpu_worker(
    target_ips: List[str] = TARGET_HOST_IPS,
    model_path: str = "/models/gemma-4-31b",
    tp_size: int = 8,
    force_reinit: bool = False,
) -> Any:
    """Acquires existing hot persistent worker or provisions node-pinned 2x4 worker."""
    if not force_reinit:
        try:
            worker = ray.get_actor(PERSISTENT_ACTOR_NAME, namespace=PERSISTENT_NAMESPACE)
            status = ray.get(worker.get_status.remote(), timeout=15)
            if status.get("ready"):
                print(f"[Benchmark 2x4] Attached to existing hot TPU worker: {status}", flush=True)
                return worker
        except Exception:
            pass

    print("[Benchmark 2x4] Cleaning up stale placement groups...", flush=True)
    cleanup_stale_placement_groups()

    print(f"[Benchmark 2x4] Provisioning node-pinned placement group across {target_ips}...", flush=True)
    bundles = [
        {"TPU": 4.0, "CPU": 1.0, f"node:{ip}": 0.001}
        for ip in target_ips
    ]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready(), timeout=120)
    print("[Benchmark 2x4] Placement group ready and pinned to 2x4 subslice.", flush=True)

    # Randomize coordinator port to avoid 'Address already in use'
    coord_port = random.randint(12800, 12950)
    os.environ["TPU_COORDINATOR_PORT"] = str(coord_port)
    print(f"[Benchmark 2x4] TPU Coordinator Port: {coord_port}", flush=True)

    worker = TPUPreemptionBenchmarkWorker.options(
        name=PERSISTENT_ACTOR_NAME,
        namespace=PERSISTENT_NAMESPACE,
        lifetime="detached",
        placement_group=pg,
        placement_group_bundle_index=0,
    ).remote(
        model_path=model_path,
        tp_size=tp_size,
        target_ips=target_ips,
    )

    print("[Benchmark 2x4] Initializing vLLM engine with real weights in HBM...", flush=True)
    ray.get(worker.initialize_engine.remote(max_num_seqs=256, watermark=0.0), timeout=1200)
    print("[Benchmark 2x4] vLLM engine initialized successfully on physical TPU hardware.", flush=True)
    return worker


def main():
    parser = argparse.ArgumentParser(description="2x4 Continuous Surgical Preempt Benchmark")
    parser.add_argument("--workload", type=str, default="data/workloads/sharegpt_sampled_500.json")
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="results/surgical_preempt_2x4_results.json")
    parser.add_argument("--force-reinit", action="store_true")
    args = parser.parse_args()

    print("[Benchmark 2x4] Initializing Ray connection...", flush=True)
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    worker = setup_or_get_tpu_worker(force_reinit=args.force_reinit)
    status = ray.get(worker.get_status.remote(), timeout=15)

    # Ingest workload
    workload_path = args.workload
    if not os.path.isabs(workload_path):
        workload_path = os.path.join(os.getcwd(), workload_path)

    print(f"[Benchmark 2x4] Loading workload trace from {workload_path}...", flush=True)
    with open(workload_path, "r") as f:
        workload = json.load(f)[:args.num_requests]

    total_prompt_tok = sum(r.get("prompt_len", 0) for r in workload)
    total_decode_tok = sum(r.get("decode_len", 0) for r in workload)
    total_demand = total_prompt_tok + total_decode_tok

    print(
        f"[Benchmark 2x4] Workload: {len(workload)} requests | "
        f"Prompts={total_prompt_tok:,} | Decodes={total_decode_tok:,} | Total Demand={total_demand:,} tokens",
        flush=True,
    )

    results = {
        "timestamp": time.time(),
        "hardware": {
            "accelerator": "Cloud TPU v5e (8 chips, 2x4 mesh, TP=8)",
            "hosts": TARGET_HOST_IPS,
            "coordinator_host": TARGET_HOST_IPS[0],
            "model": "Gemma-4 31B BF16 (61.4 GB weights in HBM)",
            "kv_cache_capacity_tokens": 120000,
            "padding_cushion_tokens": 2500,
        },
        "workload": {
            "file": os.path.basename(workload_path),
            "num_requests": len(workload),
            "prompt_tokens": total_prompt_tok,
            "decode_tokens": total_decode_tok,
            "total_demand_tokens": total_demand,
        },
        "runs": {},
    }

    # =========================================================================
    # CONFIG 1: Baseline Cohort Batching (C=64, M=64)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("[Benchmark 2x4] >>> RUN 1: Baseline Cohort Batching (M=64 cohorts) <<<", flush=True)
    print("=" * 80, flush=True)
    batches_64 = [workload[i:i+64] for i in range(0, len(workload), 64)]
    ray.get(worker.reset_kv_cache.remote())
    time.sleep(1.0)
    t0 = time.perf_counter()
    res_base = ray.get(worker.run_sustained_policy_a_greedy.remote(
        batch_stream=batches_64,
        max_duration_s=600.0,
    ))
    dur_base = time.perf_counter() - t0
    res_base["measured_wall_time_s"] = round(dur_base, 2)
    res_base["batch_size"] = 64
    res_base["num_batches"] = len(batches_64)
    results["runs"]["baseline_cohort_64"] = res_base
    print(f"[Benchmark 2x4] Run 1 Finished in {dur_base:.2f}s | Result: {json.dumps(res_base, indent=2)}", flush=True)

    # =========================================================================
    # CONFIG 2: Slotted Continuous Pipeline (Cap=103 - QuantizedKnapsack-103)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("[Benchmark 2x4] >>> RUN 2: Slotted Continuous Pipeline (Cap=103) <<<", flush=True)
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
    print(f"[Benchmark 2x4] Run 2 Finished in {dur_c103:.2f}s | Result: {json.dumps(res_c103, indent=2)}", flush=True)

    # =========================================================================
    # CONFIG 3: Continuous Surgical Preempt with Memory Cushion (Delta M_pad = 2.5k, Target C=128)
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("[Benchmark 2x4] >>> RUN 3: Continuous Surgical Preempt (Cushion=2.5k, Target C=128) <<<", flush=True)
    print("=" * 80, flush=True)
    ray.get(worker.reset_kv_cache.remote())
    time.sleep(1.0)
    t0 = time.perf_counter()
    res_surg = ray.get(worker.run_policy_continuous_surgical_preempt.remote(
        num_requests=len(workload),
        prompt_len=256,
        max_tokens=2048,
        target_concurrency=128,
        capacity_tokens=120_000,
        padding_cushion_tokens=2_500,
        custom_requests=workload,
        max_duration_s=600.0,
    ))
    dur_surg = time.perf_counter() - t0
    res_surg["measured_wall_time_s"] = round(dur_surg, 2)
    res_surg["target_concurrency"] = 128
    res_surg["padding_cushion_tokens"] = 2500
    results["runs"]["continuous_surgical_preempt_128"] = res_surg
    print(f"[Benchmark 2x4] Run 3 Finished in {dur_surg:.2f}s | Result: {json.dumps(res_surg, indent=2)}", flush=True)

    # Write results
    out_file = args.output_json
    if not os.path.isabs(out_file):
        out_file = os.path.join(os.getcwd(), out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print(f"[Benchmark 2x4] All runs complete! Saved empirical results to {out_file}", flush=True)
    print("=" * 80, flush=True)
    print("[RESULTS_JSON_START]")
    print(json.dumps(results, indent=2))
    print("[RESULTS_JSON_END]", flush=True)
    emit_omp_event(
        "BENCHMARK_2X4_COMPLETED",
        output_file=out_file,
        baseline_time_s=res_base.get("total_time_s"),
        slotted_time_s=res_c103.get("total_time_s"),
        surgical_time_s=res_surg.get("total_time_s"),
        surgical_goodput=res_surg.get("goodput_tok_per_s"),
    )


if __name__ == "__main__":
    main()
