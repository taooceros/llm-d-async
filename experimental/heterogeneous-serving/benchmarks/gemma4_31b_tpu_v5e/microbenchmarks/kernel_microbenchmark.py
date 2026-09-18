#!/usr/bin/env python3
"""
Single-Host vs Two-Host vs Four-Host Kernel Microbenchmark (Gemma-4 31B on TPU v5e)
===================================================================================
Extracts the core Transformer Forward Step:
  - QKV Projection GEMM
  - Row Parallel O-Projection GEMM + ICI All-Reduce
  - SwiGLU FFN (Gate + Up) GEMM + Down Projection GEMM + ICI All-Reduce

Evaluates:
  - TP=4 (Single-Host, Intra-Host ICI)
  - TP=8 (Two-Host, 2-Host ICI)
  - TP=16 (Four-Host, 4-Host ICI)
  - Batch sizes: [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
"""

import os
import time
import argparse
import numpy as np

HIDDEN_SIZE = 5376
INTERMEDIATE_SIZE = 21504
NUM_Q_HEADS = 32
NUM_KV_HEADS = 16
HEAD_DIM = 128
NUM_LAYERS = 48


def run_benchmark_on_mesh(mesh, tp_size: int = 16):
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

    if process_id == 0:
        print("=" * 95)
        print(f"  Gemma-4 31B Kernel Microbenchmark (TP={tp_size})")
        print(f"  Cluster: {num_processes} Hosts, {device_count} Total TPUs, Mesh: {mesh.shape}")
        print("=" * 95)

    test_buckets = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    iters = 20
    warmup = 3

    sharding_replicated = NamedSharding(mesh, P())
    sharding_col = NamedSharding(mesh, P(None, "model"))
    sharding_row = NamedSharding(mesh, P("model", None))

    local_q_dim = (NUM_Q_HEADS * HEAD_DIM) // tp_size

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

    if process_id == 0:
        print(f"\n{'Token Bucket':<15} | {'QKV+O (ms)':<15} | {'SwiGLU FFN (ms)':<18} | {'Layer Step (ms)':<18} | {'Tokens/sec':<15} | {'MFU (%)':<10}")
        print("-" * 105)

    peak_tflops = tp_size * 197.0
    flops_per_token = 39.636e9

    global_qkv_dim = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    global_q_dim = NUM_Q_HEADS * HEAD_DIM
    qkv_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, global_qkv_dim), dtype=jnp.bfloat16), sharding_col)
    o_w = jax.device_put(jnp.zeros((global_q_dim, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)
    gate_up_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE), dtype=jnp.bfloat16), sharding_col)
    down_w = jax.device_put(jnp.zeros((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)

    for b in test_buckets:
        x_hidden = jax.device_put(jnp.zeros((b, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_replicated)

        for _ in range(warmup):
            out1 = sharded_qkv_and_allreduce(x_hidden, qkv_w, o_w)
            out2 = sharded_swiglu_and_allreduce(x_hidden, gate_up_w, down_w)
            jax.block_until_ready(out1)
            jax.block_until_ready(out2)

        t0 = time.perf_counter()
        for _ in range(iters):
            out1 = sharded_qkv_and_allreduce(x_hidden, qkv_w, o_w)
            jax.block_until_ready(out1)
        t_qkv_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        t0 = time.perf_counter()
        for _ in range(iters):
            out2 = sharded_swiglu_and_allreduce(x_hidden, gate_up_w, down_w)
            jax.block_until_ready(out2)
        t_ffn_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        layer_ms = t_qkv_ms + t_ffn_ms
        tput = b / ((layer_ms * NUM_LAYERS) / 1000.0)
        achieved_tflops = (tput * flops_per_token) / 1e12
        mfu = (achieved_tflops / peak_tflops) * 100.0

        if process_id == 0:
            print(f"{b:<15} | {t_qkv_ms:<15.3f} | {t_ffn_ms:<18.3f} | {layer_ms:<18.3f} | {tput:<15.1f} | {mfu:<10.2f}%")


def run_ray_job(tp_size: int = 16):
    import ray
    ray.init(address="auto", ignore_reinit_error=True)
    nodes = ray.nodes()
    tpu_nodes = [n for n in nodes if n.get("Alive", False) and n.get("Resources", {}).get("TPU", 0) > 0]
    tpu_nodes = sorted(tpu_nodes, key=lambda n: n["NodeManagerAddress"])

    @ray.remote
    class MultihostWorker:
        def __init__(self, rank: int, world_size: int):
            self.rank = rank
            self.world_size = world_size

        def run(self, coordinator_addr: str, tp_target: int):
            import os
            os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
            os.environ["JAX_PLATFORMS"] = "tpu,cpu"
            os.environ["ENABLE_PJRT_COMPATIBILITY"] = "true"

            import jax
            from jax.sharding import Mesh

            jax.distributed.initialize(
                coordinator_address=coordinator_addr,
                num_processes=self.world_size,
                process_id=self.rank
            )

            devices = jax.devices()
            if tp_target == 16:
                device_mesh = np.array(devices).reshape(1, -1)
            elif tp_target == 8:
                device_mesh = np.array(devices).reshape(2, 8)
            elif tp_target == 4:
                device_mesh = np.array(devices).reshape(4, 4)
            mesh = Mesh(device_mesh, axis_names=("data", "model"))

            run_benchmark_on_mesh(mesh, tp_size=tp_target)
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

    futures = [w.run.remote(coord_addr, tp_size) for w in workers]
    ray.get(futures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=16, choices=[4, 8, 16])
    args = parser.parse_args()
    run_ray_job(tp_size=args.tp_size)
