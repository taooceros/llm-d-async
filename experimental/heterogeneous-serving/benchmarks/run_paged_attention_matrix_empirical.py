#!/usr/bin/env python3
"""
Empirical 2D PagedAttention Matrix Benchmark on Cloud TPU v5e
Measures the complete 2D grid of (Batch Size x Context Length) for both 2x4 (TP=8)
and 4x4 (TP=16) topologies using the native vLLM PagedAttention kernel on Gemma-4 31B.
Outputs the real hardware Latency Matrix (TPOT) and Throughput Matrix.
"""

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
TPU_PEAK_HBM_BW_GB_S = 819.0

# 2D Grid Dimensions
GRID_BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256, 512, 768, 1024]
GRID_CONTEXT_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]

DEFAULT_RAY_URL = "http://127.0.0.1:8265"
DEFAULT_OUTPUT_JSON = "results/paged_attention_matrix_empirical.json"


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
# Cluster Worker Matrix Executor (Runs across 4 TPU hosts)
# =========================================================================

def execute_cluster_matrix_benchmark(
    topology: str,
    output_file: str,
    batch_sizes: Optional[List[int]] = None,
    context_lengths: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Runs the 2D grid matrix benchmark on the Ray TPU cluster."""
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

    print(f"\n[Cluster Worker] Starting 2D Matrix PagedAttention benchmark for topology: {topology}", flush=True)
    nodes = ray.nodes()
    alive_tpu_nodes = [
        n for n in nodes if n.get("Alive") and n.get("Resources", {}).get("TPU", 0) > 0
    ]
    alive_tpu_nodes = sorted(alive_tpu_nodes, key=lambda n: n["NodeManagerAddress"])
    print(f"[Cluster Worker] Discovered {len(alive_tpu_nodes)} active TPU worker nodes.", flush=True)

    if len(alive_tpu_nodes) < 4:
        raise RuntimeError(f"All 4 TPU worker nodes required (found {len(alive_tpu_nodes)}).")

    tp_target = 16 if topology == "4x4" else 8
    local_q_heads = NUM_Q_HEADS // tp_target
    local_kv_heads = NUM_KV_HEADS // tp_target

    # Clean up old placement groups
    table = placement_group_table()
    for pg_id, pg_info in table.items():
        if pg_info.get("state") in ["CREATED", "PENDING", "RESCHEDULING"]:
            try:
                remove_placement_group(PlacementGroup(ray.PlacementGroupID(bytes.fromhex(pg_id))))
            except Exception:
                pass
    time.sleep(1.0)

    # Rank 0 is 102.18
    tpu_nodes = sorted(alive_tpu_nodes, key=lambda n: 0 if n["NodeManagerAddress"] == "192.168.102.18" else 1)
    target_ips = [n["NodeManagerAddress"] for n in tpu_nodes]
    print(f"[Cluster Worker] Topology {topology} (TP={tp_target}, Q_heads={local_q_heads}, KV_heads={local_kv_heads}) on {target_ips}", flush=True)

    bundles = [{"TPU": 4.0, "CPU": 1.0, f"node:{ip}": 0.001} for ip in target_ips]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready(), timeout=120)
    print(f"[Cluster Worker] Placement group ready for {topology}!", flush=True)
    emit_omp_event("PLACEMENT_GROUP_LOCKED", topology=topology, hosts=target_ips, tp_size=tp_target)

    @ray.remote(num_cpus=0)
    class MatrixAttentionWorker:
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

        def measure_cell(
            self,
            batch_size: int,
            context_len: int,
            num_q_heads: int,
            num_kv_heads: int,
            head_dim: int = HEAD_DIM,
            page_size: int = PAGE_SIZE,
            warmup_iters: int = 2,
            timed_iters: int = 8,
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

            page_indices = jnp.arange(B * pages_per_seq, dtype=jnp.int32)

            # Physical TPU v5e SMEM limit: 1.0 MB scalar memory.
            # ragged_paged_attention prefetches page_indices in SMEM (4 bytes/page).
            # Max tokens ~3.6M before SMEM permanent compile failure.
            total_tokens = B * L
            if total_tokens > 3_600_000:
                return {
                    "batch_size": B,
                    "context_len": L,
                    "layer_ms": None,
                    "full_model_ms": None,
                    "throughput_tok_s": 0.0,
                    "effective_bw_gb_s": 0.0,
                    "mbu_pct": 0.0,
                    "is_oom": True,
                    "error": f"SMEM_EXCEEDED: Total batch tokens {total_tokens} exceeds TPU v5e 1MB SMEM ceiling (~3.6M tokens)",
                }

            try:
                # Warmup: fresh allocations per iteration to satisfy donate_argnames
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
                del q_w, k_w, v_w, kv_w, out
                import gc
                gc.collect()
                q_list = [jnp.zeros((B, num_q_heads, head_dim), dtype=dtype) for _ in range(timed_iters)]
                k_list = [jnp.zeros((B, num_kv_heads, head_dim), dtype=dtype) for _ in range(timed_iters)]
                v_list = [jnp.zeros((B, num_kv_heads, head_dim), dtype=dtype) for _ in range(timed_iters)]
                kv_buf = jnp.zeros(kv_cache_shape, dtype=dtype)

                iter_ms = []
                for i in range(timed_iters):
                    t0 = time.perf_counter()
                    out, kv_buf = ragged_paged_attention(
                        queries=q_list[i],
                        keys=k_list[i],
                        values=v_list[i],
                        kv_cache=kv_buf,
                        kv_lens=kv_lens,
                        page_indices=page_indices,
                        cu_q_lens=cu_q_lens,
                        distribution=distribution,
                        update_kv_cache=False,
                    )
                    out.block_until_ready()
                    t1 = time.perf_counter()
                    iter_ms.append((t1 - t0) * 1000.0)

                del kv_buf, out
                import gc
                gc.collect()

                iter_ms.sort()
                layer_ms_p50 = float(np.percentile(iter_ms, 50))
                full_model_ms = layer_ms_p50 * NUM_LAYERS
                tput_tok_s = float(B / (full_model_ms / 1000.0)) if full_model_ms > 0 else 0.0

                bytes_read_per_chip = 2 * B * L * num_kv_heads * head_dim * 2
                eff_bw_gb_s = float((bytes_read_per_chip / (layer_ms_p50 / 1000.0)) / 1e9) if layer_ms_p50 > 0 else 0.0
                mbu_pct = float((eff_bw_gb_s / TPU_PEAK_HBM_BW_GB_S) * 100.0)

                return {
                    "batch_size": B,
                    "context_len": L,
                    "layer_ms": round(layer_ms_p50, 4),
                    "full_model_ms": round(full_model_ms, 2),
                    "throughput_tok_s": round(tput_tok_s, 1),
                    "effective_bw_gb_s": round(eff_bw_gb_s, 1),
                    "mbu_pct": round(mbu_pct, 1),
                    "is_oom": False,
                }
            except Exception as ex:
                import gc
                gc.collect()
                return {
                    "batch_size": B,
                    "context_len": L,
                    "layer_ms": None,
                    "full_model_ms": None,
                    "throughput_tok_s": 0.0,
                    "effective_bw_gb_s": 0.0,
                    "mbu_pct": 0.0,
                    "is_oom": True,
                    "error": str(ex),
                }

    import random
    coord_port = random.randint(12800, 12950)
    coord_addr = f"{target_ips[0]}:{coord_port}"
    workers = [
        MatrixAttentionWorker.options(
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
    print(f"[Cluster Worker] All {len(workers)} matrix workers initialized successfully.", flush=True)

    # Execute 2D Matrix Sweep (L descending to match table visualization)
    b_sizes = batch_sizes if batch_sizes is not None else GRID_BATCH_SIZES
    c_lens = context_lengths if context_lengths is not None else GRID_CONTEXT_LENGTHS
    matrix_results: Dict[str, Dict[str, Any]] = {}
    total_cells = len(c_lens) * len(b_sizes)
    current_cell = 0

    print("\n" + "=" * 105, flush=True)
    print(f"  EXECUTING REAL 2D MATRIX SWEEP ON {topology} (TP={tp_target})", flush=True)
    print("=" * 105, flush=True)
    print(f"{'Context Len (L)':<18} | {'Batch (B)':<10} | {'Layer (ms)':<14} | {'Model TPOT (ms)':<18} | {'Throughput (tok/s)':<22} | {'MBU %':<10}", flush=True)
    print("-" * 105, flush=True)

    for L in c_lens:
        matrix_results[str(L)] = {}
        for B in b_sizes:
            current_cell += 1
            futures = [
                w.measure_cell.remote(
                    batch_size=B,
                    context_len=L,
                    num_q_heads=local_q_heads,
                    num_kv_heads=local_kv_heads,
                )
                for w in workers
            ]
            all_res = ray.get(futures, timeout=120)
            r0 = all_res[0]
            matrix_results[str(L)][str(B)] = r0
            if r0.get("is_oom"):
                err_label = "SMEM_OOM" if "SMEM" in r0.get("error", "") else "OOM"
                err_msg = r0.get("error", "")[:35]
                print(
                    f"{L:<18} | {B:<10} | {err_label:<14} | {err_label:<18} | {err_msg:<22} | {'0.0%':<10}",
                    flush=True,
                )
            else:
                print(
                    f"{L:<18} | {B:<10} | {r0['layer_ms']:<14.4f} | {r0['full_model_ms']:<18.2f} | "
                    f"{r0['throughput_tok_s']:<22.1f} | {r0['mbu_pct']:<10.1f}%",
                    flush=True,
                )

    try:
        remove_placement_group(pg)
        print(f"[Cluster Worker] Placement group for {topology} released.", flush=True)
    except Exception:
        pass

    topology_payload = {
        "topology": topology,
        "tp_size": tp_target,
        "local_q_heads": local_q_heads,
        "local_kv_heads": local_kv_heads,
        "batch_sizes": b_sizes,
        "context_lengths": c_lens,
        "matrix": matrix_results,
    }

    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(topology_payload, f, indent=2)
        print(f"[Cluster Worker] Wrote {topology} matrix results to {output_file}", flush=True)
    print("\nMATRIX_RESULT_JSON=" + json.dumps(topology_payload), flush=True)

    emit_omp_event(
        "MATRIX_BENCHMARK_COMPLETE",
        topology=topology,
        batch_sizes=b_sizes,
        context_lengths=c_lens,
    )
    return topology_payload


# =========================================================================
# Local RCM Stages
# =========================================================================

@stage(name="stage_discover_cluster", order_idx=1, expected_s=5.0)
def stage_discover_cluster(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Discovers active TPU cluster nodes and checks Ray connectivity."""
    from ray.job_submission import JobSubmissionClient

    timing.log("Executing Stage 1: Cluster Discovery...", level="INFO")
    ray_url = state.get("ray_url") or os.environ.get("RAY_DASHBOARD_URL", DEFAULT_RAY_URL)
    client = JobSubmissionClient(ray_url)
    state["client"] = client

    out_file = state.get("output_json", DEFAULT_OUTPUT_JSON)
    existing_topologies = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                existing_topologies = old_data.get("topologies", {})
        except Exception:
            pass

    bs_list = state.get("batch_sizes", GRID_BATCH_SIZES)
    cl_list = state.get("context_lengths", GRID_CONTEXT_LENGTHS)
    state["results"] = {
        "metadata": {
            "timestamp": time.time(),
            "model": MODEL_NAME,
            "grid_batch_sizes": bs_list,
            "grid_context_lengths": cl_list,
        },
        "topologies": existing_topologies,
        "comparison_matrix": {},
    }
    emit_omp_event("CLUSTER_READY", ray_url=ray_url)


def _submit_cluster_matrix_stage(
    state: StateStore,
    timing: JointTimingLogEngine,
    topology: str,
    output_path: str,
) -> Dict[str, Any]:
    """Submits and streams a 2D matrix benchmark stage on the cluster."""
    from ray.job_submission import JobSubmissionClient, JobStatus

    client: JobSubmissionClient = state["client"]
    project_root = str(_REPO_ROOT)

    bs_arg = ",".join(map(str, state.get("batch_sizes", GRID_BATCH_SIZES)))
    ctx_arg = ",".join(map(str, state.get("context_lengths", GRID_CONTEXT_LENGTHS)))
    entrypoint = (
        f"python3 benchmarks/run_paged_attention_matrix_empirical.py --cluster-runner "
        f"--topology {topology} --output-file {output_path} "
        f"--batch-sizes {bs_arg} --context-lengths {ctx_arg}"
    )
    timing.log(f"Submitting {topology} Matrix job: {entrypoint}", level="INFO")

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

    data = None
    try:
        full_logs = client.get_job_logs(submission_id)
        for line in reversed(full_logs.splitlines()):
            line_str = line.strip()
            if line_str.startswith("MATRIX_RESULT_JSON="):
                json_str = line_str[len("MATRIX_RESULT_JSON="):].strip()
                data = json.loads(json_str)
                break
    except Exception as ex:
        timing.log(f"Notice parsing log payload: {ex}", level="WARN")

    if not data and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if not data:
        raise RuntimeError(f"Could not retrieve matrix data for {topology}")

    timing.log(f"Successfully collected {topology} Matrix empirical measurements.", level="INFO")
    return data


@stage(name="stage_benchmark_matrix_2x4", order_idx=2, expected_s=360.0)
def stage_benchmark_matrix_2x4(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Runs the 2D matrix benchmark on 2x4 (TP=8)."""
    timing.log("Executing Stage 2: 2x4 Matrix Benchmark...", level="INFO")
    out_path = "/tmp/paged_attn_matrix_2x4.json"
    res_2x4 = _submit_cluster_matrix_stage(state, timing, "2x4", out_path)
    state["results"]["topologies"]["2x4"] = res_2x4
    emit_omp_event("STAGE_2X4_COMPLETED")


@stage(name="stage_benchmark_matrix_4x4", order_idx=3, expected_s=360.0)
def stage_benchmark_matrix_4x4(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Runs the 2D matrix benchmark on 4x4 (TP=16)."""
    timing.log("Executing Stage 3: 4x4 Matrix Benchmark...", level="INFO")
    out_path = "/tmp/paged_attn_matrix_4x4.json"
    res_4x4 = _submit_cluster_matrix_stage(state, timing, "4x4", out_path)
    state["results"]["topologies"]["4x4"] = res_4x4
    emit_omp_event("STAGE_4X4_COMPLETED")


@stage(name="stage_aggregate_and_render_matrix", order_idx=4, expected_s=5.0)
def stage_aggregate_and_render_matrix(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Renders the final ASCII comparison matrices and saves JSON."""
    timing.log("Executing Stage 4: Matrix Rendering & Comparison...", level="INFO")
    results = state["results"]
    m_2x4 = results["topologies"].get("2x4", {}).get("matrix", {})
    m_4x4 = results["topologies"].get("4x4", {}).get("matrix", {})
    b_sizes = results["metadata"].get("grid_batch_sizes", GRID_BATCH_SIZES)
    c_lens = results["metadata"].get("grid_context_lengths", GRID_CONTEXT_LENGTHS)

    has_2x4 = bool(m_2x4)
    has_4x4 = bool(m_4x4)

    # 1. Latency Matrix (TPOT in ms)
    col_w = 17 if (has_2x4 and has_4x4) else 11
    total_w = 16 + len(b_sizes) * (col_w + 3)
    print("\n" + "=" * total_w, flush=True)
    if has_2x4 and has_4x4:
        title = "EMPIRICAL LATENCY MATRIX (TPOT in ms: 2x4 / 4x4)"
    elif has_4x4:
        title = "EMPIRICAL LATENCY MATRIX ON 4x4 (TPOT in ms)"
    else:
        title = "EMPIRICAL LATENCY MATRIX ON 2x4 (TPOT in ms)"
    print(f"  {title}", flush=True)
    print("=" * total_w, flush=True)
    header_str = f"{'Context (L)':<12} | " + " | ".join([f"B={b:<{col_w-2}}" for b in b_sizes])
    print(header_str, flush=True)
    print("-" * total_w, flush=True)

    latency_matrix = {}
    throughput_matrix = {}

    for L in reversed(c_lens):
        row_str = f"L={L:<10} | "
        l_key = str(L)
        latency_matrix[l_key] = {}
        throughput_matrix[l_key] = {}

        for B in b_sizes:
            b_key = str(B)
            r2 = m_2x4.get(l_key, {}).get(b_key, {})
            r4 = m_4x4.get(l_key, {}).get(b_key, {})
            lat2 = r2.get("full_model_ms", None)
            lat4 = r4.get("full_model_ms", None)
            tput2 = r2.get("throughput_tok_s", 0.0)
            tput4 = r4.get("throughput_tok_s", 0.0)

            latency_matrix[l_key][b_key] = {"2x4": lat2, "4x4": lat4}
            throughput_matrix[l_key][b_key] = {"2x4": tput2, "4x4": tput4}
            oom2 = r2.get("is_oom", False) or lat2 is None or lat2 == 0.0
            oom4 = r4.get("is_oom", False) or lat4 is None or lat4 == 0.0
            s2 = "OOM" if oom2 else f"{lat2:.1f}"
            s4 = "OOM" if oom4 else f"{lat4:.1f}"
            if has_2x4 and has_4x4:
                cell_str = f"{s2}/{s4}"
            elif has_4x4:
                cell_str = f"{s4}"
            else:
                cell_str = f"{s2}"
            row_str += f"{cell_str:<{col_w}} | "
        print(row_str, flush=True)

    # 2. Throughput Matrix (Tokens/sec)
    t_col_w = 17 if (has_2x4 and has_4x4) else 12
    total_tw = 16 + len(b_sizes) * (t_col_w + 3)
    print("\n" + "=" * total_tw, flush=True)
    if has_2x4 and has_4x4:
        title_t = "EMPIRICAL THROUGHPUT MATRIX (tok/s: 2x4 / 4x4)"
    elif has_4x4:
        title_t = "EMPIRICAL THROUGHPUT MATRIX ON 4x4 (tok/s)"
    else:
        title_t = "EMPIRICAL THROUGHPUT MATRIX ON 2x4 (tok/s)"
    print(f"  {title_t}", flush=True)
    print("=" * total_tw, flush=True)
    header_tput = f"{'Context (L)':<12} | " + " | ".join([f"B={b:<{t_col_w-2}}" for b in b_sizes])
    print(header_tput, flush=True)
    print("-" * total_tw, flush=True)

    for L in reversed(c_lens):
        row_str = f"L={L:<10} | "
        l_key = str(L)
        for B in b_sizes:
            b_key = str(B)
            tput2 = throughput_matrix[l_key][b_key]["2x4"]
            tput4 = throughput_matrix[l_key][b_key]["4x4"]
            r2 = m_2x4.get(l_key, {}).get(b_key, {})
            r4 = m_4x4.get(l_key, {}).get(b_key, {})
            oom2 = r2.get("is_oom", False) or tput2 == 0.0
            oom4 = r4.get("is_oom", False) or tput4 == 0.0
            s2 = "OOM" if oom2 else f"{int(tput2)}"
            s4 = "OOM" if oom4 else f"{int(tput4)}"
            if has_2x4 and has_4x4:
                cell_str = f"{s2}/{s4}"
            elif has_4x4:
                cell_str = f"{s4}"
            else:
                cell_str = f"{s2}"
            row_str += f"{cell_str:<{t_col_w}} | "
        print(row_str, flush=True)
    results["comparison_matrix"]["latency_tpot_ms"] = latency_matrix
    results["comparison_matrix"]["throughput_tok_s"] = throughput_matrix

    out_file = state.get("output_json", DEFAULT_OUTPUT_JSON)
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Report] Wrote empirical matrix dataset to {out_file}", flush=True)
    emit_omp_event("BENCHMARK_SUCCESS", json=out_file)


# =========================================================================
# CLI Entrypoint
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D PagedAttention Matrix Benchmark")
    parser.add_argument("--cluster-runner", action="store_true", help="Execute on Ray TPU cluster worker")
    parser.add_argument("--topology", type=str, choices=["2x4", "4x4"], default="2x4", help="Hardware topology")
    parser.add_argument("--output-file", type=str, default="/tmp/res_matrix.json", help="Output path for worker")
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON, help="Final JSON path")
    parser.add_argument("--ray-url", type=str, default=DEFAULT_RAY_URL, help="Ray API URL")
    parser.add_argument("--batch-sizes", type=str, default=",".join(map(str, GRID_BATCH_SIZES)), help="Comma-separated batch sizes")
    parser.add_argument("--context-lengths", type=str, default=",".join(map(str, GRID_CONTEXT_LENGTHS)), help="Comma-separated context lengths")
    parser.add_argument("--topologies", type=str, default="2x4,4x4", help="Comma-separated topologies to benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    context_lengths = [int(x.strip()) for x in args.context_lengths.split(",") if x.strip()]

    if args.cluster_runner:
        execute_cluster_matrix_benchmark(
            args.topology,
            args.output_file,
            batch_sizes=batch_sizes,
            context_lengths=context_lengths,
        )
        return 0

    topologies = [x.strip() for x in args.topologies.split(",") if x.strip()]

    runner = StageRunner(silence_timeout_s=600.0)
    runner.state["ray_url"] = args.ray_url
    runner.state["output_json"] = args.output_json
    runner.state["batch_sizes"] = batch_sizes
    runner.state["context_lengths"] = context_lengths

    runner.register(stage_discover_cluster)
    if "2x4" in topologies:
        runner.register(stage_benchmark_matrix_2x4)
    if "4x4" in topologies:
        runner.register(stage_benchmark_matrix_4x4)
    runner.register(stage_aggregate_and_render_matrix)
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
