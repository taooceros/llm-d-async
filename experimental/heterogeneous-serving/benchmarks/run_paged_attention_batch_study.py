#!/usr/bin/env python3
"""
Empirical PagedAttention Batch Size & Memory Bandwidth Utilization Benchmark on TPU v5e
Evaluates the physical impact of Batch Size and Context Length on vLLM PagedAttention
across 2x4 (TP=8) and 4x4 (TP=16) topologies for Gemma-4 31B (GQA: 32 Q heads, 16 KV heads).
"""

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

# Add repository root to path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omp_rcm import (
    DiagnosticTrap,
    JointTimingLogEngine,
    StageRunner,
    StateStore,
    stage,
)

# Gemma-4 31B Architecture Parameters
MODEL_NAME = "gemma-4-31b"
HIDDEN_SIZE = 5376
NUM_Q_HEADS = 32
NUM_KV_HEADS = 16
HEAD_DIM = 128
NUM_LAYERS = 48
PAGE_SIZE = 16
TPU_PEAK_HBM_BW_GB_S = 819.0  # Google Cloud TPU v5e specification

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512]
CONTEXT_LENGTHS = [256, 512, 1024, 2048, 4096]
DEFAULT_FIXED_LENGTH = 1024
DEFAULT_FIXED_BATCH = 32

DEFAULT_RAY_URL = "http://127.0.0.1:8265"
DEFAULT_OUTPUT_JSON = "results/paged_attention_batch_size.json"


def emit_omp_event(event_type: str, **kwargs: Any) -> str:
    """Emits structured OMP lifecycle event signal on stdout."""
    payload_items = []
    for k, v in kwargs.items():
        if isinstance(v, (dict, list)):
            payload_items.append(f"{k}={json.dumps(v, separators=(',', ':'))}")
        elif isinstance(v, str) and (" " in v or "=" in v):
            payload_items.append(f'{k}="{v}"')
        else:
            payload_items.append(f"{k}={v}")
    payload_str = " ".join(payload_items)
    line = f"\n[OMP_EVENT: {event_type} {payload_str}]"
    print(line, flush=True)
    return line


# =========================================================================
# Cluster-side Distributed Benchmark Implementation (Ray / JAX / TPU)
# =========================================================================

def execute_cluster_paged_attention_benchmark(topology: str, output_file: str) -> Dict[str, Any]:
    """Executes the empirical PagedAttention benchmark on the Ray TPU cluster."""
    import numpy as np
    import ray
    from ray.util.placement_group import (
        placement_group,
        placement_group_table,
        remove_placement_group,
        PlacementGroup,
    )
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    print(f"\n[Cluster Worker] Starting PagedAttention physical benchmark for topology: {topology}", flush=True)
    nodes = ray.nodes()
    alive_tpu_nodes = [
        n for n in nodes if n.get("Alive") and n.get("Resources", {}).get("TPU", 0) > 0
    ]
    alive_tpu_nodes = sorted(alive_tpu_nodes, key=lambda n: n["NodeManagerAddress"])
    print(f"[Cluster Worker] Discovered {len(alive_tpu_nodes)} active TPU worker nodes.", flush=True)

    if len(alive_tpu_nodes) < 4:
        raise RuntimeError(f"All 4 TPU worker nodes are required (discovered {len(alive_tpu_nodes)}).")

    tp_target = 16 if topology == "4x4" else 8
    local_q_heads = NUM_Q_HEADS // tp_target
    local_kv_heads = NUM_KV_HEADS // tp_target

    # Clean up stale placement groups
    table = placement_group_table()
    for pg_id, pg_info in table.items():
        if pg_info.get("state") in ["CREATED", "PENDING", "RESCHEDULING"]:
            try:
                remove_placement_group(PlacementGroup(ray.PlacementGroupID(bytes.fromhex(pg_id))))
            except Exception:
                pass
    time.sleep(1.0)

    # Deterministic GKE task ordering
    tpu_nodes = sorted(alive_tpu_nodes, key=lambda n: 0 if n["NodeManagerAddress"] == "192.168.102.18" else 1)
    target_ips = [n["NodeManagerAddress"] for n in tpu_nodes]
    print(f"[Cluster Worker] Configured {topology} (TP={tp_target}) across hosts: {target_ips}", flush=True)

    bundles = [{"TPU": 4.0, "CPU": 1.0, f"node:{ip}": 0.001} for ip in target_ips]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready(), timeout=120)
    print(f"[Cluster Worker] Node-pinned placement group ready for {topology}!", flush=True)
    emit_omp_event("PLACEMENT_GROUP_LOCKED", topology=topology, hosts=target_ips, tp_size=tp_target)

    # Remote PagedAttention Worker
    @ray.remote(num_cpus=0)
    class PagedAttentionTpuWorker:
        def __init__(self, rank: int, world_size: int):
            self.rank = rank
            self.world_size = world_size

        def initialize(self, coordinator_addr: str):
            import os
            os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
            os.environ["JAX_PLATFORMS"] = "tpu,cpu"
            os.environ["ENABLE_PJRT_COMPATIBILITY"] = "true"
            os.environ["JAX_COMPILATION_CACHE_DIR"] = "/models/jax_cache"

            import jax
            jax.distributed.initialize(
                coordinator_address=coordinator_addr,
                num_processes=self.world_size,
                process_id=self.rank,
                initialization_timeout=180,
            )
            return True

        def measure_paged_attention(
            self,
            batch_size: int,
            context_len: int,
            num_q_heads: int,
            num_kv_heads: int,
            head_dim: int = HEAD_DIM,
            page_size: int = PAGE_SIZE,
            warmup_iters: int = 3,
            timed_iters: int = 15,
        ) -> Dict[str, Any]:
            import jax
            import jax.numpy as jnp
            from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
                ragged_paged_attention,
                get_kv_cache_shape
            )
            from tpu_inference.kernels.ragged_paged_attention.v3.util import cdiv

            dtype = jnp.bfloat16
            B = batch_size
            L = context_len

            pages_per_seq = cdiv(L, page_size)
            total_num_pages = max(1024, B * pages_per_seq + 64)
            kv_cache_shape = get_kv_cache_shape(total_num_pages, page_size, num_kv_heads, head_dim, dtype)

            kv_lens = jnp.full((B,), L, dtype=jnp.int32)
            cu_q_lens = jnp.arange(B + 1, dtype=jnp.int32)
            distribution = jnp.array([B, 0, 0], dtype=jnp.int32)

            page_indices = []
            for i in range(B):
                needed = pages_per_seq
                start_p = i * pages_per_seq
                page_indices.extend(range(start_p, start_p + needed))
            page_indices = jnp.array(page_indices, dtype=jnp.int32)

            # Warmup runs (allocate fresh arrays to handle JAX buffer donation)
            for _ in range(warmup_iters):
                q_w = jnp.zeros((B, num_q_heads, head_dim), dtype=dtype)
                k_w = jnp.zeros((B, num_kv_heads, head_dim), dtype=dtype)
                v_w = jnp.zeros((B, num_kv_heads, head_dim), dtype=dtype)
                kv_w = jnp.zeros(kv_cache_shape, dtype=dtype)
                out, _ = ragged_paged_attention(
                    queries=q_w,
                    keys=k_w,
                    values=v_w,
                    kv_cache=kv_w,
                    kv_lens=kv_lens,
                    page_indices=page_indices,
                    cu_q_lens=cu_q_lens,
                    distribution=distribution,
                    update_kv_cache=False,
                )
                out.block_until_ready()

            # Pre-allocate buffer list for timed iterations to measure pure execution
            q_list = [jnp.zeros((B, num_q_heads, head_dim), dtype=dtype) for _ in range(timed_iters)]
            k_list = [jnp.zeros((B, num_kv_heads, head_dim), dtype=dtype) for _ in range(timed_iters)]
            v_list = [jnp.zeros((B, num_kv_heads, head_dim), dtype=dtype) for _ in range(timed_iters)]
            kv_list = [jnp.zeros(kv_cache_shape, dtype=dtype) for _ in range(timed_iters)]

            iter_ms = []
            for i in range(timed_iters):
                t0 = time.perf_counter()
                out, _ = ragged_paged_attention(
                    queries=q_list[i],
                    keys=k_list[i],
                    values=v_list[i],
                    kv_cache=kv_list[i],
                    kv_lens=kv_lens,
                    page_indices=page_indices,
                    cu_q_lens=cu_q_lens,
                    distribution=distribution,
                    update_kv_cache=False,
                )
                out.block_until_ready()
                t1 = time.perf_counter()
                iter_ms.append((t1 - t0) * 1000.0)

            iter_ms.sort()
            layer_ms_p50 = float(np.percentile(iter_ms, 50))
            layer_ms_p90 = float(np.percentile(iter_ms, 90))
            layer_ms_mean = float(np.mean(iter_ms))

            full_model_ms = layer_ms_p50 * NUM_LAYERS
            tput_tok_s = float(B / (full_model_ms / 1000.0)) if full_model_ms > 0 else 0.0
            tpot_ms = full_model_ms

            # Bandwidth & MBU calculations:
            # Each token requires reading 2 tensors (K and V) of size (num_kv_heads * head_dim * 2 bytes)
            bytes_read_per_chip = 2 * B * L * num_kv_heads * head_dim * 2
            eff_bw_gb_s = float((bytes_read_per_chip / (layer_ms_p50 / 1000.0)) / 1e9) if layer_ms_p50 > 0 else 0.0
            mbu_pct = float((eff_bw_gb_s / TPU_PEAK_HBM_BW_GB_S) * 100.0)

            return {
                "batch_size": B,
                "context_len": L,
                "num_q_heads": num_q_heads,
                "num_kv_heads": num_kv_heads,
                "layer_ms_p50": round(layer_ms_p50, 4),
                "layer_ms_p90": round(layer_ms_p90, 4),
                "layer_ms_mean": round(layer_ms_mean, 4),
                "full_model_ms": round(full_model_ms, 3),
                "decode_throughput_tok_s": round(tput_tok_s, 1),
                "tpot_ms": round(tpot_ms, 3),
                "bytes_read_per_chip": bytes_read_per_chip,
                "effective_bw_gb_s": round(eff_bw_gb_s, 2),
                "mbu_pct": round(mbu_pct, 2),
                "raw_samples_ms": [round(x, 4) for x in iter_ms],
            }

    coord_port = 12800
    coord_addr = f"{target_ips[0]}:{coord_port}"
    workers = [
        PagedAttentionTpuWorker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            )
        ).remote(i, len(target_ips))
        for i in range(len(target_ips))
    ]

    print(f"[Cluster Worker] Initializing JAX distributed workers at {coord_addr}...", flush=True)
    init_futures = [w.initialize.remote(coord_addr) for w in workers]
    ray.get(init_futures, timeout=180)
    print(f"[Cluster Worker] All {len(workers)} PagedAttention workers initialized successfully.", flush=True)

    # 1. Sweep Batch Size at Fixed Context Length (L = 1024)
    print("\n" + "=" * 95, flush=True)
    print(f"  PAGED_ATTENTION BATCH SIZE SWEEP (L={DEFAULT_FIXED_LENGTH}) ON {topology} (TP={tp_target})", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Batch Size':<12} | {'Context Len':<12} | {'Layer (ms)':<14} | {'Model (ms)':<14} | {'Effective BW':<16} | {'MBU %':<10} | {'Throughput':<15}", flush=True)
    print("-" * 95, flush=True)

    batch_sweep_results = []
    for bs in BATCH_SIZES:
        futures = [
            w.measure_paged_attention.remote(
                batch_size=bs,
                context_len=DEFAULT_FIXED_LENGTH,
                num_q_heads=local_q_heads,
                num_kv_heads=local_kv_heads,
            )
            for w in workers
        ]
        all_res = ray.get(futures, timeout=120)
        rank0_res = all_res[0]
        batch_sweep_results.append(rank0_res)
        print(
            f"{bs:<12} | {DEFAULT_FIXED_LENGTH:<12} | {rank0_res['layer_ms_p50']:<14.4f} | "
            f"{rank0_res['full_model_ms']:<14.3f} | {rank0_res['effective_bw_gb_s']:<13.1f} GB/s | "
            f"{rank0_res['mbu_pct']:<8.1f}% | {rank0_res['decode_throughput_tok_s']:<15.1f}",
            flush=True,
        )

    # 2. Sweep Context Length at Fixed Batch Size (B = 32)
    print("\n" + "=" * 95, flush=True)
    print(f"  PAGED_ATTENTION CONTEXT LENGTH SWEEP (B={DEFAULT_FIXED_BATCH}) ON {topology} (TP={tp_target})", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Context Len':<12} | {'Batch Size':<12} | {'Layer (ms)':<14} | {'Model (ms)':<14} | {'Effective BW':<16} | {'MBU %':<10} | {'Throughput':<15}", flush=True)
    print("-" * 95, flush=True)

    context_sweep_results = []
    for ctx_len in CONTEXT_LENGTHS:
        futures = [
            w.measure_paged_attention.remote(
                batch_size=DEFAULT_FIXED_BATCH,
                context_len=ctx_len,
                num_q_heads=local_q_heads,
                num_kv_heads=local_kv_heads,
            )
            for w in workers
        ]
        all_res = ray.get(futures, timeout=120)
        rank0_res = all_res[0]
        context_sweep_results.append(rank0_res)
        print(
            f"{ctx_len:<12} | {DEFAULT_FIXED_BATCH:<12} | {rank0_res['layer_ms_p50']:<14.4f} | "
            f"{rank0_res['full_model_ms']:<14.3f} | {rank0_res['effective_bw_gb_s']:<13.1f} GB/s | "
            f"{rank0_res['mbu_pct']:<8.1f}% | {rank0_res['decode_throughput_tok_s']:<15.1f}",
            flush=True,
        )

    # Cleanup placement group
    try:
        remove_placement_group(pg)
        print(f"[Cluster Worker] Placement group for {topology} removed cleanly.", flush=True)
    except Exception as e:
        print(f"[Cluster Worker] Warning releasing placement group: {e}", flush=True)

    topology_payload = {
        "topology": topology,
        "tp_size": tp_target,
        "local_q_heads": local_q_heads,
        "local_kv_heads": local_kv_heads,
        "target_hosts": target_ips,
        "batch_sweep_l1024": batch_sweep_results,
        "context_sweep_b32": context_sweep_results,
    }

    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(topology_payload, f, indent=2)
        print(f"[Cluster Worker] Wrote {topology} results to {output_file}", flush=True)
    print("\nPAGED_ATTENTION_RESULT_JSON=" + json.dumps(topology_payload), flush=True)

    emit_omp_event(
        "TOPOLOGY_PAGED_ATTENTION_COMPLETE",
        topology=topology,
        tp_size=tp_target,
        batch_sizes=BATCH_SIZES,
        context_lengths=CONTEXT_LENGTHS,
    )
    return topology_payload


# =========================================================================
# Local RCM Stages (Executed via `python3 -m omp_rcm run ...`)
# =========================================================================

@stage(name="stage_discover_cluster", order_idx=1, expected_s=5.0)
def stage_discover_cluster(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Discovers active TPU cluster nodes and checks Ray connectivity."""
    from ray.job_submission import JobSubmissionClient

    timing.log("Executing Stage 1: Cluster Discovery & Connectivity Validation...", level="INFO")
    ray_url = state.get("ray_url") or os.environ.get("RAY_DASHBOARD_URL", DEFAULT_RAY_URL)
    client = JobSubmissionClient(ray_url)
    state["client"] = client
    state["results"] = {
        "metadata": {
            "timestamp": time.time(),
            "model": MODEL_NAME,
            "architecture": {
                "hidden_size": HIDDEN_SIZE,
                "num_q_heads": NUM_Q_HEADS,
                "num_kv_heads": NUM_KV_HEADS,
                "head_dim": HEAD_DIM,
                "num_layers": NUM_LAYERS,
                "page_size": PAGE_SIZE,
            },
            "hardware": {
                "chip": "TPU v5e",
                "hbm_bandwidth_gb_s": TPU_PEAK_HBM_BW_GB_S,
            },
            "batch_sizes": BATCH_SIZES,
            "context_lengths": CONTEXT_LENGTHS,
        },
        "topologies": {},
        "comparison_analysis": {},
    }
    emit_omp_event("CLUSTER_READY", ray_url=ray_url)


def _submit_cluster_paged_attention_stage(
    state: StateStore,
    timing: JointTimingLogEngine,
    topology: str,
    output_path: str,
) -> Dict[str, Any]:
    """Submits and streams a PagedAttention benchmark stage on the cluster."""
    from ray.job_submission import JobSubmissionClient, JobStatus

    client: JobSubmissionClient = state["client"]
    project_root = str(_REPO_ROOT)

    entrypoint = f"python3 benchmarks/run_paged_attention_batch_study.py --cluster-runner --topology {topology} --output-file {output_path}"
    timing.log(f"Submitting {topology} PagedAttention job: {entrypoint}", level="INFO")

    runtime_env = {
        "working_dir": project_root,
        "env_vars": {
            "TPU_MULTIHOST_BACKEND": "ray",
            "SCAN_TPU_CHIPS": "True",
            "TPU_ACCELERATOR_TYPE": "v5e",
            "JAX_PLATFORMS": "tpu,cpu",
            "JAX_COMPILATION_CACHE_DIR": "/models/jax_cache",
            "PYTHONUNBUFFERED": "1",
        },
    }

    submission_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)
    timing.log(f"Job submitted for {topology}. Submission ID: {submission_id}", level="INFO")
    emit_omp_event("CLUSTER_JOB_STARTED", topology=topology, submission_id=submission_id)

    last_log_offset = 0
    poll_interval = 2.0
    trap = DiagnosticTrap()

    while True:
        status = client.get_job_status(submission_id)
        try:
            full_logs = client.get_job_logs(submission_id)
            if len(full_logs) > last_log_offset:
                new_chunk = full_logs[last_log_offset:]
                sys.stdout.write(new_chunk)
                sys.stdout.flush()
                last_log_offset = len(full_logs)
        except Exception:
            pass

        if status.is_terminal():
            break
        time.sleep(poll_interval)

    if status != JobStatus.SUCCEEDED:
        raise RuntimeError(f"Ray job {submission_id} failed with status {status}")

    # Parse output payload
    data = None
    try:
        full_logs = client.get_job_logs(submission_id)
        for line in reversed(full_logs.splitlines()):
            line_str = line.strip()
            if line_str.startswith("PAGED_ATTENTION_RESULT_JSON="):
                json_str = line_str[len("PAGED_ATTENTION_RESULT_JSON="):].strip()
                data = json.loads(json_str)
                break
    except Exception as ex:
        timing.log(f"Notice parsing log payload: {ex}", level="WARN")

    if not data and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if not data:
        raise RuntimeError(f"Could not retrieve benchmark data for {topology}")

    timing.log(f"Successfully collected {topology} PagedAttention measurements.", level="INFO")
    return data


@stage(name="stage_benchmark_paged_attention_2x4", order_idx=2, expected_s=75.0)
def stage_benchmark_paged_attention_2x4(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Runs the 2x4 (TP=8) PagedAttention benchmark across batch sizes and context lengths."""
    timing.log("Executing Stage 2: 2x4 Topology PagedAttention Benchmark...", level="INFO")
    out_path = "/tmp/paged_attention_2x4.json"
    res_2x4 = _submit_cluster_paged_attention_stage(state, timing, "2x4", out_path)
    state["results"]["topologies"]["2x4"] = res_2x4
    emit_omp_event("STAGE_2X4_COMPLETED", batch_count=len(res_2x4["batch_sweep_l1024"]))


@stage(name="stage_benchmark_paged_attention_4x4", order_idx=3, expected_s=75.0)
def stage_benchmark_paged_attention_4x4(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Runs the 4x4 (TP=16) PagedAttention benchmark across batch sizes and context lengths."""
    timing.log("Executing Stage 3: 4x4 Topology PagedAttention Benchmark...", level="INFO")
    out_path = "/tmp/paged_attention_4x4.json"
    res_4x4 = _submit_cluster_paged_attention_stage(state, timing, "4x4", out_path)
    state["results"]["topologies"]["4x4"] = res_4x4
    emit_omp_event("STAGE_4X4_COMPLETED", batch_count=len(res_4x4["batch_sweep_l1024"]))


@stage(name="stage_aggregate_and_report", order_idx=4, expected_s=5.0)
def stage_aggregate_and_report(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Aggregates metrics, calculates MBU scaling and TP comparison, and saves final JSON."""
    timing.log("Executing Stage 4: Aggregation & MBU Topology Comparison...", level="INFO")
    results = state["results"]
    res_2x4 = results["topologies"].get("2x4", {})
    res_4x4 = results["topologies"].get("4x4", {})

    b_sweep_2x4 = {r["batch_size"]: r for r in res_2x4.get("batch_sweep_l1024", [])}
    b_sweep_4x4 = {r["batch_size"]: r for r in res_4x4.get("batch_sweep_l1024", [])}

    print("\n" + "=" * 115, flush=True)
    print(f"  PAGED_ATTENTION BATCH SIZE COMPARISON (L={DEFAULT_FIXED_LENGTH}) (2x4 TP=8 vs 4x4 TP=16)", flush=True)
    print("=" * 115, flush=True)
    print(f"{'Batch Size':<12} | {'2x4 Lat (ms)':<14} | {'2x4 MBU %':<12} | {'4x4 Lat (ms)':<14} | {'4x4 MBU %':<12} | {'Speedup (4x4/2x4)':<20} | {'Throughput (4x4)':<18}", flush=True)
    print("-" * 115, flush=True)

    comparison_batch = {}
    for bs in BATCH_SIZES:
        if bs in b_sweep_2x4 and bs in b_sweep_4x4:
            r2 = b_sweep_2x4[bs]
            r4 = b_sweep_4x4[bs]
            speedup = (r2["layer_ms_p50"] / r4["layer_ms_p50"]) if r4["layer_ms_p50"] > 0 else 0.0
            comparison_batch[str(bs)] = {
                "batch_size": bs,
                "layer_ms_2x4": r2["layer_ms_p50"],
                "mbu_2x4": r2["mbu_pct"],
                "layer_ms_4x4": r4["layer_ms_p50"],
                "mbu_4x4": r4["mbu_pct"],
                "speedup_4x4_vs_2x4": round(speedup, 3),
                "throughput_4x4": r4["decode_throughput_tok_s"],
            }
            print(
                f"{bs:<12} | {r2['layer_ms_p50']:<14.4f} | {r2['mbu_pct']:<10.1f}% | "
                f"{r4['layer_ms_p50']:<14.4f} | {r4['mbu_pct']:<10.1f}% | {speedup:<20.3f} | {r4['decode_throughput_tok_s']:<18.1f}",
                flush=True,
            )

    results["comparison_analysis"]["batch_scaling_l1024"] = comparison_batch

    # Save final aggregated JSON
    out_file = state.get("output_json", DEFAULT_OUTPUT_JSON)
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Report] Wrote final empirical dataset to {out_file}", flush=True)
    emit_omp_event("BENCHMARK_SUCCESS", json=out_file)


# =========================================================================
# CLI Driver
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PagedAttention Batch Size & MBU Benchmark")
    parser.add_argument("--cluster-runner", action="store_true", help="Execute on Ray TPU cluster worker")
    parser.add_argument("--topology", type=str, choices=["2x4", "4x4"], default="2x4", help="Hardware topology")
    parser.add_argument("--output-file", type=str, default="/tmp/res_paged_attn.json", help="Output path for cluster worker")
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON, help="Final aggregated JSON path")
    parser.add_argument("--ray-url", type=str, default=DEFAULT_RAY_URL, help="Ray cluster dashboard/job submission API URL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.cluster_runner:
        execute_cluster_paged_attention_benchmark(args.topology, args.output_file)
        return 0

    runner = StageRunner(silence_timeout_s=300.0)
    runner.state["ray_url"] = args.ray_url
    runner.state["output_json"] = args.output_json

    runner.register(stage_discover_cluster)
    runner.register(stage_benchmark_paged_attention_2x4)
    runner.register(stage_benchmark_paged_attention_4x4)
    runner.register(stage_aggregate_and_report)
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
