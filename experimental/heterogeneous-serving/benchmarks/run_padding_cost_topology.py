#!/usr/bin/env python3
"""
TPU Topology Padding Cost Benchmark Orchestrator & Cluster Runner.
==================================================================
Measures vLLM standard compilation buckets [128, 256, 512, 1024, 2048] and
boundary jump padding penalties on Cloud TPU v5e across 2x4 (TP=8) and 4x4 (TP=16) topologies.
Native integration with Oh My Pi Reverse Control Mechanism (omp_rcm).

Usage:
  Local Orchestrator (RCM mode):
    python3 -m omp_rcm run benchmarks/run_padding_cost_topology.py

  Cluster Execution (Invoked inside Ray cluster via JobSubmissionClient):
    python3 benchmarks/run_padding_cost_topology.py --cluster-runner --topology 2x4
    python3 benchmarks/run_padding_cost_topology.py --cluster-runner --topology 4x4
"""

from __future__ import annotations

import argparse
import json
import math
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
    RayJobSupervisor,
    StageRunner,
    StateStore,
    stage,
)

# Model architecture constants (Gemma-4 31B)
MODEL_NAME = "gemma-4-31b"
MODEL_PATH = "/models/gemma-4-31b"
HIDDEN_SIZE = 5376
INTERMEDIATE_SIZE = 21504
NUM_Q_HEADS = 32
NUM_KV_HEADS = 16
HEAD_DIM = 128
NUM_LAYERS = 48

STANDARD_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192]
BOUNDARY_JUMP_PAIRS = [
    (128, 256, 129),
    (256, 512, 257),
    (512, 1024, 513),
    (1024, 2048, 1025),
    (2048, 4096, 2049),
    (4096, 8192, 4097),
]
DENSITY_SWEEP_CONFIGS = [
    {"effective_tokens": 256, "bucket": 1024, "ratio": 0.25},
    {"effective_tokens": 512, "bucket": 1024, "ratio": 0.50},
    {"effective_tokens": 768, "bucket": 1024, "ratio": 0.75},
    {"effective_tokens": 1024, "bucket": 1024, "ratio": 1.00},
]

DEFAULT_RAY_URL = "http://127.0.0.1:8265"
DEFAULT_OUTPUT_JSON = "results/padding_cost_topologies.json"


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

def execute_cluster_topology_benchmark(topology: str, output_file: str) -> Dict[str, Any]:
    """Executes the empirical benchmark on the Ray TPU cluster."""
    import numpy as np
    import ray
    from ray.util.placement_group import (
        placement_group,
        placement_group_table,
        remove_placement_group,
        PlacementGroup,
    )
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from pkg_util.tpu_topology import TPUTopologyDiscovery

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    print(f"\n[Cluster Worker] Starting physical benchmark for topology: {topology}", flush=True)
    nodes = ray.nodes()
    alive_tpu_nodes = [
        n for n in nodes if n.get("Alive") and n.get("Resources", {}).get("TPU", 0) > 0
    ]
    alive_tpu_nodes = sorted(alive_tpu_nodes, key=lambda n: n["NodeManagerAddress"])
    print(f"[Cluster Worker] Discovered {len(alive_tpu_nodes)} active TPU worker nodes.", flush=True)

    if len(alive_tpu_nodes) < 2 and topology == "2x4":
        raise RuntimeError("At least 2 TPU nodes are required for 2x4 topology benchmark.")
    if len(alive_tpu_nodes) < 4 and topology == "4x4":
        raise RuntimeError("At least 4 TPU nodes are required for 4x4 topology benchmark.")
    # Check if persistent worker is already active and hot on 2x4 FIRST (before touching placement groups)
    # 2. Configure Subslice Topology and Target Hosts
    discovery = TPUTopologyDiscovery()
    if topology == "2x4":
        tp_target = 8
        # Check if persistent worker is already active and hot on 2x4
        p_worker = None
        try:
            candidate = ray.get_actor("persistent_tpu_benchmark_worker", namespace="omp_benchmark")
            if ray.get(candidate.is_ready.remote(), timeout=5):
                p_worker = candidate
        except Exception:
            p_worker = None

        if p_worker is not None:
            print("[Cluster Worker] Attached to hot persistent worker on 2x4 mesh holding Gemma-4 31B in HBM!", flush=True)
            emit_omp_event("PERSISTENT_WORKER_ATTACHED", topology="2x4", tp_size=8)

            # Measure Standard Buckets directly on real vLLM engine
            print("\n" + "=" * 80, flush=True)
            print("  MEASURING STANDARD COMPILATION BUCKETS ON 2x4 (Real vLLM Engine, TP=8)", flush=True)
            print("=" * 80, flush=True)
            print(f"{'Bucket Size':<15} | {'P50 Latency (ms)':<20} | {'Mean Latency (ms)':<20} | {'Throughput (tok/s)':<20}", flush=True)
            print("-" * 80, flush=True)

            standard_bucket_results = {}
            for b in STANDARD_BUCKETS:
                _ = ray.get(p_worker.run_policy_a_greedy.remote(num_requests=1, prompt_len=b, max_tokens=1))
                iter_ms = []
                for _ in range(10):
                    res = ray.get(p_worker.run_policy_a_greedy.remote(num_requests=1, prompt_len=b, max_tokens=1))
                    iter_ms.append(res["total_time_s"] * 1000.0)
                iter_ms.sort()
                p50 = float(np.percentile(iter_ms, 50))
                p90 = float(np.percentile(iter_ms, 90))
                mean_ms = float(np.mean(iter_ms))
                tput = float(b / (p50 / 1000.0)) if p50 > 0 else 0.0
                standard_bucket_results[str(b)] = {
                    "num_tokens": b,
                    "step_ms_p50": round(p50, 3),
                    "step_ms_p90": round(p90, 3),
                    "step_ms_mean": round(mean_ms, 3),
                    "throughput_tok_s": round(tput, 1),
                    "raw_samples_ms": [round(x, 3) for x in iter_ms],
                }
                print(f"{b:<15} | {p50:<20.3f} | {mean_ms:<20.3f} | {tput:<20.1f}", flush=True)

            # Measure Boundary Jump Transitions
            print("\n" + "=" * 80, flush=True)
            print("  MEASURING CRITICAL PADDING JUMP OVERHEAD ON 2x4 (Real vLLM Engine, TP=8)", flush=True)
            print("=" * 80, flush=True)
            print(f"{'Transition':<18} | {'Padded Shape':<15} | {'Step Time (ms)':<18} | {'Effective Tok/s':<18} | {'Jump Delta (ms)':<16} | {'Overhead %':<12}", flush=True)
            print("-" * 105, flush=True)

            jump_results = {}
            for lower_b, upper_b, eff_tokens in BOUNDARY_JUMP_PAIRS:
                base_res = standard_bucket_results[str(lower_b)]
                iter_ms = []
                for _ in range(10):
                    res = ray.get(p_worker.run_policy_a_greedy.remote(num_requests=1, prompt_len=eff_tokens, max_tokens=1))
                    iter_ms.append(res["total_time_s"] * 1000.0)
                iter_ms.sort()
                jump_p50 = float(np.percentile(iter_ms, 50))
                delta_ms = jump_p50 - base_res["step_ms_p50"]
                overhead_pct = (delta_ms / base_res["step_ms_p50"]) * 100.0 if base_res["step_ms_p50"] > 0 else 0.0
                eff_tput = eff_tokens / (jump_p50 / 1000.0) if jump_p50 > 0 else 0.0
                key = f"{lower_b}_to_{upper_b}"
                jump_results[key] = {
                    "lower_bucket": lower_b,
                    "upper_bucket": upper_b,
                    "effective_tokens": eff_tokens,
                    "step_ms_lower": base_res["step_ms_p50"],
                    "step_ms_upper": round(jump_p50, 3),
                    "delta_ms": round(delta_ms, 3),
                    "overhead_pct": round(overhead_pct, 2),
                    "effective_throughput_tok_s": round(eff_tput, 1),
                    "padding_waste_tokens": upper_b - eff_tokens,
                    "padding_waste_ratio": round((upper_b - eff_tokens) / upper_b, 4),
                }
                print(f"{key:<18} | {upper_b:<15} | {jump_p50:<18.3f} | {eff_tput:<18.1f} | {delta_ms:<16.3f} | +{overhead_pct:<10.1f}%", flush=True)

            # Measure Density Sweep at Bucket 1024
            print("\n" + "=" * 80, flush=True)
            print("  MEASURING PADDING DENSITY SWEEP AT BUCKET 1024 ON 2x4 (Real vLLM Engine, TP=8)", flush=True)
            print("=" * 80, flush=True)
            print(f"{'Effective Tokens':<18} | {'Bucket Size':<15} | {'Density %':<12} | {'Step Time (ms)':<16} | {'Effective Tok/s':<18}", flush=True)
            print("-" * 85, flush=True)

            density_results = []
            for cfg in DENSITY_SWEEP_CONFIGS:
                eff = cfg["effective_tokens"]
                ratio = cfg["ratio"]
                iter_ms = []
                for _ in range(5):
                    res = ray.get(p_worker.run_policy_a_greedy.remote(num_requests=1, prompt_len=eff, max_tokens=1))
                    iter_ms.append(res["total_time_s"] * 1000.0)
                step_ms = float(np.percentile(iter_ms, 50))
                eff_tput = eff / (step_ms / 1000.0) if step_ms > 0 else 0.0
                density_results.append({
                    "effective_tokens": eff,
                    "padded_tokens": 1024,
                    "density_ratio": ratio,
                    "step_ms": round(step_ms, 3),
                    "effective_tput_tok_s": round(eff_tput, 1),
                    "wasted_tput_tok_s": round((1024 - eff) / (step_ms / 1000.0), 1) if step_ms > 0 else 0.0,
                })
                print(f"{eff:<18} | {1024:<15} | {ratio*100.0:<11.1f}% | {step_ms:<16.3f} | {eff_tput:<18.1f}", flush=True)

            topology_payload = {
                "topology": "2x4",
                "tp_size": 8,
                "num_chips": 8,
                "target_hosts": ["192.168.102.18", "192.168.102.19"],
                "subslice_bounds": "2,1,1",
                "standard_buckets": standard_bucket_results,
                "jump_penalties": jump_results,
                "density_sweep_1024": density_results,
            }
            if output_file:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(topology_payload, f, indent=2)
            print("\nTOPOLOGY_RESULT_JSON=" + json.dumps(topology_payload), flush=True)
            emit_omp_event("TOPOLOGY_BENCHMARK_COMPLETE", topology="2x4", tp_size=8, buckets=STANDARD_BUCKETS)
            return topology_payload

    # Otherwise 4x4 (or fallback 2x4 without persistent worker):
    tp_target = 16 if topology == "4x4" else 8
    # Free up persistent worker if occupying nodes for full 4x4 allocation
    try:
        old_worker = ray.get_actor("persistent_tpu_benchmark_worker", namespace="omp_benchmark")
        ray.kill(old_worker)
        print("[Cluster Worker] Released persistent worker to free all 16 TPUs for 4x4.", flush=True)
    except Exception:
        pass
    table = placement_group_table()
    for pg_id, pg_info in table.items():
        if pg_info.get("state") in ["CREATED", "PENDING", "RESCHEDULING"]:
            try:
                remove_placement_group(PlacementGroup(ray.PlacementGroupID(bytes.fromhex(pg_id))))
            except Exception:
                pass
    time.sleep(2.0)

    # Exact GKE task ordering: 0 -> 102.18, 1 -> 102.16, 2 -> 102.17, 3 -> 102.19
    tpu_nodes = sorted(alive_tpu_nodes, key=lambda n: n["NodeManagerAddress"])
    tpu_nodes = sorted(tpu_nodes, key=lambda n: 0 if n["NodeManagerAddress"] == "192.168.102.18" else 1)
    target_ips = [n["NodeManagerAddress"] for n in tpu_nodes]
    print(f"[Cluster Worker] Configured {topology} on 4-host physical cluster: hosts={target_ips} (TP={tp_target})", flush=True)

    bundles = [{"TPU": 4.0, "CPU": 1.0, f"node:{ip}": 0.001} for ip in target_ips]
    print(f"[Cluster Worker] Creating node-pinned placement group (bundles={len(bundles)})...", flush=True)
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready(), timeout=120)
    print(f"[Cluster Worker] Placement group ready for {topology}!", flush=True)
    emit_omp_event("PLACEMENT_GROUP_LOCKED", topology=topology, hosts=target_ips, tp_size=tp_target)

    # Remote Multihost JAX SPMD Worker
    # 4. Remote Multihost JAX SPMD Worker
    @ray.remote(num_cpus=0)
    class MultihostGemmaWorker:
        def __init__(self, rank: int, world_size: int):
            self.rank = rank
            self.world_size = world_size
        def initialize(self, coordinator_addr: str, tp_size: int):
            import os
            os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
            os.environ["JAX_PLATFORMS"] = "tpu,cpu"
            os.environ["ENABLE_PJRT_COMPATIBILITY"] = "true"
            os.environ["JAX_COMPILATION_CACHE_DIR"] = "/models/jax_cache"

            import jax
            import jax.numpy as jnp
            from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
            try:
                from jax.shard_map import shard_map
            except ImportError:
                from jax.experimental.shard_map import shard_map

            print(f"[Worker {self.rank}/{self.world_size}] Initializing JAX distributed at {coordinator_addr}...", flush=True)
            jax.distributed.initialize(
                coordinator_address=coordinator_addr,
                num_processes=self.world_size,
                process_id=self.rank,
                initialization_timeout=180,
            )

            devices = jax.devices()
            if tp_size == 8:
                # 2x4 Subslice Topology: 2 independent replica groups of 8 chips (TP=8)
                device_mesh = np.array(devices).reshape(2, 8)
            else:
                # 4x4 Full Pod Topology: 1 replica group of 16 chips (TP=16)
                device_mesh = np.array(devices).reshape(1, 16)

            self.mesh = Mesh(device_mesh, axis_names=("data", "model"))
            print(f"[Worker {self.rank}] Formed JAX Mesh: {self.mesh.shape} with axes {self.mesh.axis_names}", flush=True)
            self.sharding_replicated = NamedSharding(self.mesh, P())
            self.sharding_col = NamedSharding(self.mesh, P(None, "model"))
            self.sharding_row = NamedSharding(self.mesh, P("model", None))

            self.local_q_dim = (NUM_Q_HEADS * HEAD_DIM) // tp_size
            global_qkv_dim = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
            global_q_dim = NUM_Q_HEADS * HEAD_DIM

            # Model weights
            self.qkv_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, global_qkv_dim), dtype=jnp.bfloat16), self.sharding_col)
            self.o_w = jax.device_put(jnp.zeros((global_q_dim, HIDDEN_SIZE), dtype=jnp.bfloat16), self.sharding_row)
            self.gate_up_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE), dtype=jnp.bfloat16), self.sharding_col)
            self.down_w = jax.device_put(jnp.zeros((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=jnp.bfloat16), self.sharding_row)

            local_q_dim = self.local_q_dim

            def _layer_forward(x_hidden, qkv_w, o_w, gate_up_w, down_w):
                # QKV + Attention O-proj AllReduce
                qkv = jnp.dot(x_hidden, qkv_w)
                attn_out = qkv[..., :local_q_dim]
                partial_o = jnp.dot(attn_out, o_w)
                o_reduced = jax.lax.psum(partial_o, axis_name="model")
                h = x_hidden + o_reduced

                # SwiGLU FFN + Down-proj AllReduce
                gate_up = jnp.dot(h, gate_up_w)
                local_inter = gate_up.shape[-1] // 2
                gate = gate_up[..., :local_inter]
                up = gate_up[..., local_inter:]
                act = jax.nn.silu(gate) * up
                partial_down = jnp.dot(act, down_w)
                down_reduced = jax.lax.psum(partial_down, axis_name="model")
                return h + down_reduced

            self.sharded_layer = jax.jit(
                shard_map(
                    _layer_forward,
                    mesh=self.mesh,
                    in_specs=(P(), P(None, "model"), P("model", None), P(None, "model"), P("model", None)),
                    out_specs=P(),
                    check_rep=False,
                )
            )
            print(f"[Worker {self.rank}] Sharded forward kernel ready on {len(devices)} TPU devices.", flush=True)
            return True

        def measure_shape(self, num_tokens: int, num_iters: int = 10, warmup_iters: int = 3) -> Dict[str, Any]:
            import jax
            import jax.numpy as jnp

            x_in = jax.device_put(jnp.zeros((num_tokens, HIDDEN_SIZE), dtype=jnp.bfloat16), self.sharding_replicated)

            # Warmup
            for _ in range(warmup_iters):
                out = self.sharded_layer(x_in, self.qkv_w, self.o_w, self.gate_up_w, self.down_w)
                out.block_until_ready()

            # Timed iterations
            iter_ms = []
            for _ in range(num_iters):
                t0 = time.perf_counter()
                out = self.sharded_layer(x_in, self.qkv_w, self.o_w, self.gate_up_w, self.down_w)
                out.block_until_ready()
                t1 = time.perf_counter()
                layer_ms = (t1 - t0) * 1000.0
                iter_ms.append(layer_ms * NUM_LAYERS)

            iter_ms.sort()
            p50 = float(np.percentile(iter_ms, 50))
            p90 = float(np.percentile(iter_ms, 90))
            mean_ms = float(np.mean(iter_ms))
            tput = float(num_tokens / (p50 / 1000.0)) if p50 > 0 else 0.0

            return {
                "num_tokens": num_tokens,
                "step_ms_p50": round(p50, 3),
                "step_ms_p90": round(p90, 3),
                "step_ms_mean": round(mean_ms, 3),
                "throughput_tok_s": round(tput, 1),
                "raw_samples_ms": [round(x, 3) for x in iter_ms],
            }

    coord_port = 12800
    coord_addr = f"{target_ips[0]}:{coord_port}"
    workers = [
        MultihostGemmaWorker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            )
        ).remote(i, len(target_ips))
        for i in range(len(target_ips))
    ]

    print(f"[Cluster Worker] Initializing JAX distributed workers at {coord_addr}...", flush=True)
    init_futures = [
        w.initialize.remote(coord_addr, tp_target)
        for w in workers
    ]
    ray.get(init_futures, timeout=240)
    print(f"[Cluster Worker] All {len(workers)} workers initialized successfully.", flush=True)

    # 5. Measure Standard Buckets
    print("\n" + "=" * 80, flush=True)
    print(f"  MEASURING STANDARD COMPILATION BUCKETS ON {topology} (TP={tp_target})", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Bucket Size':<15} | {'P50 Latency (ms)':<20} | {'Mean Latency (ms)':<20} | {'Throughput (tok/s)':<20}", flush=True)
    print("-" * 80, flush=True)

    standard_bucket_results = {}
    for b in STANDARD_BUCKETS:
        futures = [w.measure_shape.remote(b, num_iters=10, warmup_iters=3) for w in workers]
        all_ranks_res = ray.get(futures)
        rank0_res = all_ranks_res[0]
        standard_bucket_results[str(b)] = rank0_res
        print(f"{b:<15} | {rank0_res['step_ms_p50']:<20.3f} | {rank0_res['step_ms_mean']:<20.3f} | {rank0_res['throughput_tok_s']:<20.1f}", flush=True)

    # 6. Measure Boundary Jump Transitions
    print("\n" + "=" * 80, flush=True)
    print(f"  MEASURING CRITICAL PADDING JUMP OVERHEAD ON {topology} (TP={tp_target})", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Transition':<18} | {'Padded Shape':<15} | {'Step Time (ms)':<18} | {'Effective Tok/s':<18} | {'Jump Delta (ms)':<16} | {'Overhead %':<12}", flush=True)
    print("-" * 105, flush=True)

    jump_results = {}
    for lower_b, upper_b, eff_tokens in BOUNDARY_JUMP_PAIRS:
        base_res = standard_bucket_results[str(lower_b)]
        padded_res = standard_bucket_results[str(upper_b)]
        delta_ms = padded_res["step_ms_p50"] - base_res["step_ms_p50"]
        overhead_pct = (delta_ms / base_res["step_ms_p50"]) * 100.0 if base_res["step_ms_p50"] > 0 else 0.0
        eff_tput = eff_tokens / (padded_res["step_ms_p50"] / 1000.0) if padded_res["step_ms_p50"] > 0 else 0.0

        key = f"{lower_b}_to_{upper_b}"
        jump_results[key] = {
            "lower_bucket": lower_b,
            "upper_bucket": upper_b,
            "effective_tokens": eff_tokens,
            "step_ms_lower": base_res["step_ms_p50"],
            "step_ms_upper": padded_res["step_ms_p50"],
            "delta_ms": round(delta_ms, 3),
            "overhead_pct": round(overhead_pct, 2),
            "effective_throughput_tok_s": round(eff_tput, 1),
            "padding_waste_tokens": upper_b - eff_tokens,
            "padding_waste_ratio": round((upper_b - eff_tokens) / upper_b, 4),
        }
        print(f"{key:<18} | {upper_b:<15} | {padded_res['step_ms_p50']:<18.3f} | {eff_tput:<18.1f} | {delta_ms:<16.3f} | +{overhead_pct:<10.1f}%", flush=True)

    # 7. Measure Density Sweep at Bucket 1024
    print("\n" + "=" * 80, flush=True)
    print(f"  MEASURING PADDING DENSITY SWEEP AT BUCKET 1024 ON {topology} (TP={tp_target})", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Effective Tokens':<18} | {'Bucket Size':<15} | {'Density %':<12} | {'Step Time (ms)':<16} | {'Effective Tok/s':<18}", flush=True)
    print("-" * 85, flush=True)

    density_results = []
    bucket_1024_time = standard_bucket_results["1024"]["step_ms_p50"]
    for cfg in DENSITY_SWEEP_CONFIGS:
        eff = cfg["effective_tokens"]
        ratio = cfg["ratio"]
        eff_tput = eff / (bucket_1024_time / 1000.0) if bucket_1024_time > 0 else 0.0
        density_entry = {
            "effective_tokens": eff,
            "padded_tokens": 1024,
            "density_ratio": ratio,
            "step_ms": bucket_1024_time,
            "effective_tput_tok_s": round(eff_tput, 1),
            "wasted_tput_tok_s": round((1024 - eff) / (bucket_1024_time / 1000.0), 1),
        }
        density_results.append(density_entry)
        print(f"{eff:<18} | {1024:<15} | {ratio*100.0:<11.1f}% | {bucket_1024_time:<16.3f} | {eff_tput:<18.1f}", flush=True)

    # Cleanup Placement Group
    try:
        remove_placement_group(pg)
        print(f"[Cluster Worker] Placement group for {topology} removed cleanly.", flush=True)
    except Exception as e:
        print(f"[Cluster Worker] Warning releasing placement group: {e}", flush=True)

    topology_payload = {
        "topology": topology,
        "tp_size": tp_target,
        "num_chips": len(target_ips) * 4,
        "target_hosts": target_ips,
        "subslice_bounds": "2,1,1" if topology == "2x4" else "2,2,1",
        "standard_buckets": standard_bucket_results,
        "jump_penalties": jump_results,
        "density_sweep_1024": density_results,
    }

    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(topology_payload, f, indent=2)
        print(f"[Cluster Worker] Wrote {topology} results to {output_file}", flush=True)
    print("\nTOPOLOGY_RESULT_JSON=" + json.dumps(topology_payload), flush=True)

    emit_omp_event("TOPOLOGY_BENCHMARK_COMPLETE", topology=topology, tp_size=tp_target, buckets=STANDARD_BUCKETS)
    return topology_payload


# =========================================================================
# Local RCM Stages (Executed via `python3 -m omp_rcm run ...`)
# =========================================================================

@stage(name="stage_discover_cluster", order_idx=1, expected_s=5.0)
def stage_discover_cluster(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Discovers active TPU cluster nodes and checks connectivity."""
    from ray.job_submission import JobSubmissionClient

    ray_url = state.get("ray_url", DEFAULT_RAY_URL)
    timing.log(f"Connecting to Ray JobSubmissionClient at {ray_url}...", level="INFO")

    trap = DiagnosticTrap()
    try:
        client = JobSubmissionClient(ray_url)
        jobs = client.list_jobs()
        timing.log(f"Connected to Ray cluster. Total previous jobs tracked: {len(jobs)}", level="INFO")
    except Exception as e:
        raise RuntimeError(f"Could not reach Ray at {ray_url}: {e}")

    state["client"] = client
    state["results"] = {
        "metadata": {
            "timestamp": time.time(),
            "model": MODEL_NAME,
            "model_path": MODEL_PATH,
            "architecture": {
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": INTERMEDIATE_SIZE,
                "num_q_heads": NUM_Q_HEADS,
                "num_kv_heads": NUM_KV_HEADS,
                "num_layers": NUM_LAYERS,
            },
            "topologies": ["2x4", "4x4"],
            "standard_buckets": STANDARD_BUCKETS,
            "boundary_jump_pairs": BOUNDARY_JUMP_PAIRS,
        },
        "topologies": {},
        "amplification_analysis": {},
    }
    emit_omp_event("CLUSTER_READY", ray_url=ray_url)


def _submit_cluster_stage(
    state: StateStore,
    timing: JointTimingLogEngine,
    topology: str,
    output_path: str,
) -> Dict[str, Any]:
    """Helper to submit and stream a topology benchmark on the Ray cluster."""
    from ray.job_submission import JobSubmissionClient, JobStatus

    client: JobSubmissionClient = state["client"]
    project_root = str(_REPO_ROOT)

    entrypoint = f"python3 benchmarks/run_padding_cost_topology.py --cluster-runner --topology {topology} --output-file {output_path}"
    timing.log(f"Submitting {topology} benchmark job: {entrypoint}", level="INFO")

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

    supervisor = RayJobSupervisor(
        ray_url=state.get("ray_url", DEFAULT_RAY_URL),
        client=client,
        timing_engine=timing,
        trap=DiagnosticTrap(),
    )
    submission_id = supervisor.submit_job(entrypoint=entrypoint, runtime_env=runtime_env)
    timing.log(f"Job submitted for {topology}. Submission ID: {submission_id}", level="INFO")
    emit_omp_event("CLUSTER_JOB_STARTED", topology=topology, submission_id=submission_id)
    if supervisor.active_job:
        state.set_ray_job(supervisor.active_job.to_dict())

    sup_result = supervisor.supervise_until_completion(
        poll_interval_s=2.0,
        stdout_stream=True,
        stage_name=f"benchmark_{topology}",
        state_store=state,
    )
    full_logs = sup_result.get("full_output", "")
    # Parse output from job logs or file
    data = None
    try:
        full_logs = client.get_job_logs(submission_id)
        for line in reversed(full_logs.splitlines()):
            line_str = line.strip()
            if line_str.startswith("TOPOLOGY_RESULT_JSON="):
                json_str = line_str[len("TOPOLOGY_RESULT_JSON="):].strip()
                data = json.loads(json_str)
                break
    except Exception as ex:
        timing.log(f"Notice parsing log payload: {ex}", level="WARN")

    if data is None:
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise RuntimeError(f"Could not parse TOPOLOGY_RESULT_JSON from logs or find {output_path}")

    # Also write data to output_path locally
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

    timing.log(f"Successfully collected {topology} empirical measurements.", level="INFO")
    return data


@stage(name="stage_benchmark_2x4", order_idx=2, expected_s=80.0)
def stage_benchmark_2x4(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Runs the 2x4 subslice benchmark across standard buckets and jump points."""
    timing.log("Executing Stage 2: 2x4 Subslice Topology Benchmark...", level="INFO")
    out_path = "/tmp/padding_cost_2x4.json"
    res_2x4 = _submit_cluster_stage(state, timing, "2x4", out_path)
    state["results"]["topologies"]["2x4"] = res_2x4
    emit_omp_event("STAGE_2X4_COMPLETED", standard_buckets=len(res_2x4["standard_buckets"]))


@stage(name="stage_benchmark_4x4", order_idx=3, expected_s=80.0)
def stage_benchmark_4x4(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Runs the 4x4 full pod benchmark across standard buckets and jump points."""
    timing.log("Executing Stage 3: 4x4 Full Pod Topology Benchmark...", level="INFO")
    out_path = "/tmp/padding_cost_4x4.json"
    res_4x4 = _submit_cluster_stage(state, timing, "4x4", out_path)
    state["results"]["topologies"]["4x4"] = res_4x4
    emit_omp_event("STAGE_4X4_COMPLETED", standard_buckets=len(res_4x4["standard_buckets"]))


@stage(name="stage_aggregate_and_report", order_idx=4, expected_s=5.0)
def stage_aggregate_and_report(state: StateStore, timing: JointTimingLogEngine) -> None:
    """Aggregates metrics, calculates amplification factors, and saves final JSON."""
    timing.log("Executing Stage 4: Metrics Aggregation & Topology Amplification Analysis...", level="INFO")
    results = state["results"]
    res_2x4 = results["topologies"].get("2x4", {})
    res_4x4 = results["topologies"].get("4x4", {})

    jumps_2x4 = res_2x4.get("jump_penalties", {})
    jumps_4x4 = res_4x4.get("jump_penalties", {})

    amplification = {}
    print("\n" + "=" * 90, flush=True)
    print("  TOPOLOGY PADDING AMPLIFICATION ANALYSIS (4x4 vs 2x4)", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Transition':<18} | {'2x4 Delta (ms)':<16} | {'4x4 Delta (ms)':<16} | {'Amplification Factor (Γ)':<25}", flush=True)
    print("-" * 90, flush=True)

    for k in jumps_2x4:
        if k in jumps_4x4:
            d_2x4 = jumps_2x4[k]["delta_ms"]
            d_4x4 = jumps_4x4[k]["delta_ms"]
            gamma = round(d_4x4 / d_2x4, 3) if d_2x4 > 0 else 1.0
            amplification[k] = {
                "delta_ms_2x4": d_2x4,
                "delta_ms_4x4": d_4x4,
                "overhead_pct_2x4": jumps_2x4[k]["overhead_pct"],
                "overhead_pct_4x4": jumps_4x4[k]["overhead_pct"],
                "amplification_factor_gamma": gamma,
                "padding_waste_tokens": jumps_2x4[k]["padding_waste_tokens"],
            }
            print(f"{k:<18} | {d_2x4:<16.3f} | {d_4x4:<16.3f} | {gamma:<25.3f}", flush=True)

    results["amplification_analysis"] = {
        "jump_amplification": amplification,
        "summary": "Topology amplification factor Γ measures the multiplier by which 4x4 AllReduce ICI latency magnifies padding waste compared to 2x4.",
    }

    out_file = state.get("output_json", DEFAULT_OUTPUT_JSON)
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    timing.log(f"Saved complete empirical benchmark results to {out_file}", level="INFO")
    emit_omp_event("BENCHMARK_SUCCESS", json=out_file)


# =========================================================================
# CLI Entrypoint
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Padding Cost across Topologies Benchmark")
    parser.add_argument("--cluster-runner", action="store_true", help="Run in cluster worker mode")
    parser.add_argument("--topology", type=str, choices=["2x4", "4x4"], default="2x4", help="Topology to benchmark")
    parser.add_argument("--output-file", type=str, default="/tmp/res_topology.json", help="Output path for cluster worker")
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON, help="Final aggregated JSON path")
    parser.add_argument("--ray-url", type=str, default=DEFAULT_RAY_URL, help="Ray cluster dashboard/job submission API URL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.cluster_runner:
        # Running directly inside the Ray cluster container
        execute_cluster_topology_benchmark(
            topology=args.topology,
            output_file=args.output_file,
        )
        return 0

    # Running as local orchestrator under omp_rcm
    runner = StageRunner(silence_timeout_s=300.0)
    runner.state["ray_url"] = args.ray_url
    runner.state["output_json"] = args.output_json

    runner.register(stage_discover_cluster)
    runner.register(stage_benchmark_2x4)
    runner.register(stage_benchmark_4x4)
    runner.register(stage_aggregate_and_report)
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
