#!/usr/bin/env python3
"""
Single-Host TPU Microbenchmark: Gemma-4 31B Sharded on 4 Local TPU v5e Chips (TP=4)
==================================================================================
Runs on a single TPU VM host utilizing the 4 local TPU chips without cross-host
communication overhead. Compares local TP=4 kernel performance across batch sizes.
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


def run_single_host_benchmark(tp_size: int = 4):
    os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
    os.environ["JAX_PLATFORMS"] = "tpu,cpu"
    os.environ["ENABLE_PJRT_COMPATIBILITY"] = "true"

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    try:
        from jax.shard_map import shard_map
    except ImportError:
        from jax.experimental.shard_map import shard_map

    local_devices = jax.local_devices()
    num_devices = len(local_devices)
    print("=" * 80)
    print("  Single-Host Local TPU Kernel Benchmark (Gemma-4 31B)")
    print(f"  Local TPU Devices : {num_devices} ({local_devices})")
    print(f"  Tensor Parallel Size: {tp_size}")
    print("=" * 80)

    # Build local 1D Mesh
    device_mesh = np.array(local_devices[:tp_size]).reshape(1, -1)
    mesh = Mesh(device_mesh, axis_names=("data", "model"))

    test_buckets = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    iters = 25
    warmup = 5

    sharding_replicated = NamedSharding(mesh, P())
    sharding_col = NamedSharding(mesh, P(None, "model"))
    sharding_row = NamedSharding(mesh, P("model", None))

    local_q_dim = (NUM_Q_HEADS * HEAD_DIM) // tp_size

    # 1. Sharded Attention + O-proj All-Reduce (Intra-Host)
    def _qkv_and_allreduce_impl(x_hidden, qkv_w, o_w):
        qkv = jnp.dot(x_hidden, qkv_w)
        attn_out = qkv[..., :local_q_dim]
        partial_o = jnp.dot(attn_out, o_w)
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

    # 2. Sharded FFN (SwiGLU) + Down-proj All-Reduce (Intra-Host)
    def _swiglu_and_allreduce_impl(x_hidden, gate_up_w, down_w):
        gate_up = jnp.dot(x_hidden, gate_up_w)
        local_inter = gate_up.shape[-1] // 2
        gate = gate_up[..., :local_inter]
        up = gate_up[..., local_inter:]
        activated = jax.nn.silu(gate) * up
        partial_down = jnp.dot(activated, down_w)
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

    print(f"\n{'Token Bucket':<15} | {'QKV+O (ms)':<15} | {'SwiGLU FFN (ms)':<18} | {'Layer Step (ms)':<18} | {'Tokens/sec':<15} | {'MFU (%)':<10}")
    print("-" * 95)

    # Single-chip peak = 197 TFLOPS -> 4 chips = 788 TFLOPS
    cluster_peak_tflops = 4 * 197.0
    # Gemma-4 31B single token FLOPs = 39.636 GFLOPs across 48 layers
    flops_per_token = (48 * ((2 * HIDDEN_SIZE * (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM) +
                             (2 * (NUM_Q_HEADS * HEAD_DIM) * HIDDEN_SIZE) +
                             (6 * HIDDEN_SIZE * INTERMEDIATE_SIZE)))

    for b in test_buckets:
        x_hidden = jax.device_put(jnp.zeros((b, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_replicated)

        global_qkv_dim = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
        global_q_dim = NUM_Q_HEADS * HEAD_DIM
        qkv_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, global_qkv_dim), dtype=jnp.bfloat16), sharding_col)
        o_w = jax.device_put(jnp.zeros((global_q_dim, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)

        gate_up_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE), dtype=jnp.bfloat16), sharding_col)
        down_w = jax.device_put(jnp.zeros((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)

        # Warm-up
        for _ in range(warmup):
            out1 = sharded_qkv_and_allreduce(x_hidden, qkv_w, o_w)
            out2 = sharded_swiglu_and_allreduce(x_hidden, gate_up_w, down_w)
            jax.block_until_ready(out1)
            jax.block_until_ready(out2)

        # Benchmark QKV + AllReduce
        t0 = time.perf_counter()
        for _ in range(iters):
            out1 = sharded_qkv_and_allreduce(x_hidden, qkv_w, o_w)
            jax.block_until_ready(out1)
        t_qkv_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        # Benchmark FFN + AllReduce
        t0 = time.perf_counter()
        for _ in range(iters):
            out2 = sharded_swiglu_and_allreduce(x_hidden, gate_up_w, down_w)
            jax.block_until_ready(out2)
        t_ffn_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        layer_ms = t_qkv_ms + t_ffn_ms
        tput = b / ((layer_ms * NUM_LAYERS) / 1000.0)
        achieved_tflops = (tput * flops_per_token) / 1e12
        mfu = (achieved_tflops / cluster_peak_tflops) * 100.0

        print(f"{b:<15} | {t_qkv_ms:<15.3f} | {t_ffn_ms:<18.3f} | {layer_ms:<18.3f} | {tput:<15.1f} | {mfu:<10.2f}%")

    print("=" * 95)
    print("Single-Host Microbenchmark Completed Successfully.")


def run_ray_single_host_worker():
    """Submits the single-host benchmark to run on a single TPU worker node via Ray."""
    import ray
    print("[Ray Mode] Connecting to Ray cluster...")
    ray.init(address="auto", ignore_reinit_error=True)

    nodes = ray.nodes()
    tpu_nodes = [n for n in nodes if n.get("Alive", False) and n.get("Resources", {}).get("TPU", 0) > 0]
    target_node = tpu_nodes[0]
    print(f"[Ray Mode] Running Single-Host Benchmark on Target Node: {target_node['NodeManagerAddress']}")

    @ray.remote(num_cpus=8, resources={f"node:{target_node['NodeManagerAddress']}": 1, "TPU": 4})
    def single_host_runner():
        run_single_host_benchmark(tp_size=4)
        return True

    ray.get(single_host_runner.remote())
    print("[Ray Mode] Single-host execution finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Host Gemma-4 31B Benchmark")
    parser.add_argument("--mode", choices=["direct", "ray"], default="ray", help="Run mode")
    args = parser.parse_args()

    if args.mode == "ray":
        run_ray_single_host_worker()
    else:
        run_single_host_benchmark(tp_size=4)
