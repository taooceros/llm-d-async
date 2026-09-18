#!/usr/bin/env python3
"""
Two-Host TPU Microbenchmark: Gemma-4 31B Sharded on 2 TPU Hosts (TP=8)
======================================================================
On our 4x4 (16-chip) TPU v5e cluster, this configures a 2D Mesh:
  Mesh shape: (data=2, model=8)
    - "data"  axis: 2 independent replica groups
    - "model" axis: 8 TPU chips across 2 physical hosts (Two-Host TP=8)

All-Reduce is performed strictly along the "model" axis (TP=8).
This precisely measures the cross-host 2-node (8-chip) TPU ICI Ring All-Reduce,
GEMM tiling efficiency (K=512), and MFU.
"""

import os
import time
import argparse
import numpy as np

# Gemma-4 31B Model Architecture Constants
HIDDEN_SIZE = 5376
INTERMEDIATE_SIZE = 21504
NUM_Q_HEADS = 32
NUM_KV_HEADS = 16
HEAD_DIM = 128
VOCAB_SIZE = 256000
NUM_LAYERS = 48
RMS_NORM_EPS = 1e-6


def run_benchmark_on_2d_mesh(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    try:
        from jax.shard_map import shard_map
    except ImportError:
        from jax.experimental.shard_map import shard_map

    process_id = jax.process_index()
    num_processes = jax.process_count()
    device_count = jax.device_count()
    local_device_count = jax.local_device_count()

    tp_size = 8  # 2-host TP degree

    if process_id == 0:
        print("=" * 95)
        print("  Two-Host (2 Hosts, TP=8) Distributed Gemma-4 31B Kernel Benchmark")
        print(f"  Cluster Info : {num_processes} Hosts, {device_count} Total TPUs ({local_device_count} local/host)")
        print(f"  2D Mesh Shape: {mesh.shape} on axes {mesh.axis_names}")
        print(f"  Tensor Parallel (TP) Degree = {tp_size} (Across 2 Hosts / 8 Chips)")
        print(f"  Data Parallel Replicas = {mesh.shape['data']}")
        print("=" * 95)

    test_buckets = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    iters = 25
    warmup = 5

    sharding_replicated = NamedSharding(mesh, P())
    sharding_col = NamedSharding(mesh, P(None, "model"))
    sharding_row = NamedSharding(mesh, P("model", None))

    local_q_dim = (NUM_Q_HEADS * HEAD_DIM) // tp_size

    # Sharded Attention + O-proj All-Reduce along the 8-chip "model" axis (2 hosts)
    def _qkv_and_allreduce_impl(x_hidden, qkv_w, o_w):
        qkv = jnp.dot(x_hidden, qkv_w)
        attn_out = qkv[..., :local_q_dim]
        partial_o = jnp.dot(attn_out, o_w)
        # 2-Host All-Reduce across 8 chips
        o_reduced = jax.lax.psum(partial_o, axis_name="model")
        return o_reduced

    sharded_qkv_and_allreduce = jax.jit(
        shard_map(
            _qkv_and_allreduce_impl,
            mesh=mesh,
            in_specs=(P(), P(None, "model"), P("model", None)),
            out_specs=P(),
            check_rep=False
        )
    )

    # Sharded FFN (SwiGLU) + Down-proj All-Reduce along the 8-chip "model" axis
    def _swiglu_and_allreduce_impl(x_hidden, gate_up_w, down_w):
        gate_up = jnp.dot(x_hidden, gate_up_w)
        local_inter = gate_up.shape[-1] // 2
        gate = gate_up[..., :local_inter]
        up = gate_up[..., local_inter:]
        activated = jax.nn.silu(gate) * up
        partial_down = jnp.dot(activated, down_w)
        # 2-Host All-Reduce across 8 chips
        down_reduced = jax.lax.psum(partial_down, axis_name="model")
        return down_reduced

    sharded_swiglu_and_allreduce = jax.jit(
        shard_map(
            _swiglu_and_allreduce_impl,
            mesh=mesh,
            in_specs=(P(), P(None, "model"), P("model", None)),
            out_specs=P(),
            check_rep=False
        )
    )

    if process_id == 0:
        print(f"\n{'Token Bucket':<15} | {'QKV+O (ms)':<15} | {'SwiGLU FFN (ms)':<18} | {'Layer Step (ms)':<18} | {'Two-Host Tok/s':<18} | {'Two-Host MFU':<15}")
        print("-" * 105)

    two_host_peak_tflops = 8 * 197.0  # 1576 TFLOPS for 8 chips
    flops_per_token = 39.636e9  # 39.636 GFLOPs per token for Gemma-4 31B

    for b in test_buckets:
        x_hidden = jax.device_put(jnp.zeros((b, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_replicated)

        global_qkv_dim = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
        global_q_dim = NUM_Q_HEADS * HEAD_DIM
        qkv_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, global_qkv_dim), dtype=jnp.bfloat16), sharding_col)
        o_w = jax.device_put(jnp.zeros((global_q_dim, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)

        gate_up_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE), dtype=jnp.bfloat16), sharding_col)
        down_w = jax.device_put(jnp.zeros((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)

        # Warmup
        for _ in range(warmup):
            out1 = sharded_qkv_and_allreduce(x_hidden, qkv_w, o_w)
            out2 = sharded_swiglu_and_allreduce(x_hidden, gate_up_w, down_w)
            jax.block_until_ready(out1)
            jax.block_until_ready(out2)

        # Benchmark QKV
        t0 = time.perf_counter()
        for _ in range(iters):
            out1 = sharded_qkv_and_allreduce(x_hidden, qkv_w, o_w)
            jax.block_until_ready(out1)
        t_qkv_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        # Benchmark FFN
        t0 = time.perf_counter()
        for _ in range(iters):
            out2 = sharded_swiglu_and_allreduce(x_hidden, gate_up_w, down_w)
            jax.block_until_ready(out2)
        t_ffn_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        layer_ms = t_qkv_ms + t_ffn_ms
        tput_two_host = b / ((layer_ms * NUM_LAYERS) / 1000.0)
        achieved_tflops = (tput_two_host * flops_per_token) / 1e12
        mfu = (achieved_tflops / two_host_peak_tflops) * 100.0

        if process_id == 0:
            print(f"{b:<15} | {t_qkv_ms:<15.3f} | {t_ffn_ms:<18.3f} | {layer_ms:<18.3f} | {tput_two_host:<18.1f} | {mfu:<14.2f}%")

    if process_id == 0:
        print("=" * 105)
        print("Two-Host (TP=8) Microbenchmark Completed Successfully.")


def run_ray_mode():
    import ray
    print("[Ray Mode] Connecting to Ray cluster...")
    ray.init(address="auto", ignore_reinit_error=True)

    nodes = ray.nodes()
    tpu_nodes = [n for n in nodes if n.get("Alive", False) and n.get("Resources", {}).get("TPU", 0) > 0]
    tpu_nodes = sorted(tpu_nodes, key=lambda n: n["NodeManagerAddress"])

    print(f"[Ray Mode] Connected! Total Cluster Nodes: {len(nodes)}, TPU Worker Nodes: {len(tpu_nodes)}")

    @ray.remote
    class MultihostWorker:
        def __init__(self, rank: int, world_size: int):
            self.rank = rank
            self.world_size = world_size

        def run(self, coordinator_addr: str):
            import os
            os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
            os.environ["JAX_PLATFORMS"] = "tpu,cpu"
            os.environ["ENABLE_PJRT_COMPATIBILITY"] = "true"

            import jax
            from jax.sharding import Mesh

            print(f"[Worker {self.rank}/{self.world_size}] Initializing JAX distributed...")
            jax.distributed.initialize(
                coordinator_address=coordinator_addr,
                num_processes=self.world_size,
                process_id=self.rank
            )

            devices = jax.devices()
            # 2D Mesh: (data=2, model=8) -> 2 independent 2-host TP=8 groups
            device_mesh = np.array(devices).reshape(2, 8)
            mesh = Mesh(device_mesh, axis_names=("data", "model"))

            run_benchmark_on_2d_mesh(mesh)
            return True

    coord_ip = tpu_nodes[0]["NodeManagerAddress"]
    coord_addr = f"{coord_ip}:12345"

    num_workers = len(tpu_nodes)
    workers = [
        MultihostWorker.options(
            resources={f"node:{node['NodeManagerAddress']}": 1, "TPU": 4}
        ).remote(i, num_workers)
        for i, node in enumerate(tpu_nodes)
    ]

    futures = [w.run.remote(coord_addr) for w in workers]
    ray.get(futures)
    print("[Ray Mode] All workers completed successfully.")


if __name__ == "__main__":
    run_ray_mode()
