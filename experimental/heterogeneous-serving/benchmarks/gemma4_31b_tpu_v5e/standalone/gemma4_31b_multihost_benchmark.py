#!/usr/bin/env python3
"""
Multi-Host Gemma-4 31B Kernel Microbenchmark Suite (Ray & JAX Distributed)
==========================================================================
This script benchmarks the core kernels of Gemma-4 31B in a Multi-Host TPU cluster.
It can run in two modes:
  1. Ray Multi-Host Cluster Mode (vLLM native distributed runtime)
  2. Standalone JAX Multi-Host Mesh Mode (jax.distributed + NamedSharding)

Usage:
  # Mode 1: Run via Ray on existing multi-host cluster (Head Node):
  python3 gemma4_31b_multihost_benchmark.py --mode=ray --tp-size=16

  # Mode 2: Run via JAX distributed across workers:
  python3 gemma4_31b_multihost_benchmark.py --mode=jax --coordinator-address=10.128.0.10:1234 --num-processes=2 --process-id=0
"""

import os
import time
import argparse
import numpy as np

# -----------------------------------------------------------------------------
# Gemma-4 31B Model Architecture Constants
# -----------------------------------------------------------------------------
HIDDEN_SIZE = 5376
INTERMEDIATE_SIZE = 21504
NUM_Q_HEADS = 32
NUM_KV_HEADS = 16
HEAD_DIM = 128
VOCAB_SIZE = 256000
NUM_LAYERS = 48
RMS_NORM_EPS = 1e-6


def run_benchmark_on_mesh(mesh, tp_size: int):
    """Executes sharded kernel microbenchmarks on the distributed JAX TPU Mesh."""
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

    if process_id == 0:
        print("=" * 80)
        print("  Multi-Host Distributed Gemma-4 31B Kernel Benchmark")
        print(f"  Cluster Info : {num_processes} Hosts, {device_count} Total TPUs ({local_device_count} local/host)")
        print(f"  Mesh Topology: {mesh.shape} on axes {mesh.axis_names}")
        print(f"  Tensor Parallel Size: {tp_size}")
        print("=" * 80)

    test_buckets = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    iters = 20
    warmup = 3

    # 1. Sharding Specifications
    sharding_replicated = NamedSharding(mesh, P())
    sharding_col = NamedSharding(mesh, P(None, "model"))
    sharding_row = NamedSharding(mesh, P("model", None))

    # Inside shard_map, each device sees its LOCAL slice of sharded arrays!
    # Global qkv_w: [HIDDEN_SIZE, (Q+2*KV)*HEAD_DIM] -> local qkv_w: [HIDDEN_SIZE, ((Q+2*KV)*HEAD_DIM)//tp_size]
    # Global o_w: [NUM_Q_HEADS*HEAD_DIM, HIDDEN_SIZE] -> local o_w: [(NUM_Q_HEADS*HEAD_DIM)//tp_size, HIDDEN_SIZE]
    local_q_dim = (NUM_Q_HEADS * HEAD_DIM) // tp_size

    def _qkv_and_allreduce_impl(x_hidden, qkv_w, o_w):
        # x_hidden: [B, H] (replicated)
        # qkv_w: [H, local_qkv_dim] (column sharded)
        qkv = jnp.dot(x_hidden, qkv_w) # [B, local_qkv_dim]
        
        # Local Q slice for projection
        attn_out = qkv[..., :local_q_dim]
        
        # O-Proj Row Parallel GEMM: [B, local_q_dim] @ [local_q_dim, H] -> [B, H]
        partial_o = jnp.dot(attn_out, o_w)
        
        # Cross-Host All-Reduce over TPU ICI Network
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

    # Sharded FFN (SwiGLU) Block
    def _swiglu_and_allreduce_impl(x_hidden, gate_up_w, down_w):
        # gate_up_w: [H, 2 * local_inter_dim]
        gate_up = jnp.dot(x_hidden, gate_up_w)
        local_inter = gate_up.shape[-1] // 2
        gate = gate_up[..., :local_inter]
        up = gate_up[..., local_inter:]
        activated = jax.nn.silu(gate) * up
        
        # Down proj row GEMM: [B, local_inter] @ [local_inter, H] -> [B, H]
        partial_down = jnp.dot(activated, down_w)
        # Cross-Host All-Reduce
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
        print(f"\n{'Token Bucket':<15} | {'QKV+O (ms)':<15} | {'SwiGLU FFN (ms)':<18} | {'Layer Step (ms)':<18} | {'Tokens/sec':<15}")
        print("-" * 85)

    for b in test_buckets:
        # Create sharded mock tensors
        x_hidden = jax.device_put(jnp.zeros((b, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_replicated)
        
        # Full global dimensions (JAX device_put + sharding will shard them across the 16 TPUs)
        global_qkv_dim = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
        global_q_dim = NUM_Q_HEADS * HEAD_DIM
        qkv_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, global_qkv_dim), dtype=jnp.bfloat16), sharding_col)
        o_w = jax.device_put(jnp.zeros((global_q_dim, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)

        gate_up_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE), dtype=jnp.bfloat16), sharding_col)
        down_w = jax.device_put(jnp.zeros((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)

        # Warm-up (triggers JAX AOT compilation on the Mesh)
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

        if process_id == 0:
            print(f"{b:<15} | {t_qkv_ms:<15.3f} | {t_ffn_ms:<18.3f} | {layer_ms:<18.3f} | {tput:<15.1f}")

    if process_id == 0:
        print("=" * 80)
        print("Multi-Host Microbenchmark Completed Successfully.")


# -----------------------------------------------------------------------------
# Mode 1: Ray Multi-Host Cluster Driver
# -----------------------------------------------------------------------------
def run_ray_mode(tp_size: int = 16):
    """Deploys the microbenchmark across the existing Ray cluster's TPU nodes."""
    import ray
    print(f"[Ray Mode] Connecting to existing Ray cluster...")
    ray.init(address="auto", ignore_reinit_error=True)
    
    nodes = ray.nodes()
    tpu_nodes = [n for n in nodes if n.get("Alive", False) and n.get("Resources", {}).get("TPU", 0) > 0]
    # Sort by node IP for deterministic rank assignment
    tpu_nodes = sorted(tpu_nodes, key=lambda n: n["NodeManagerAddress"])
    
    print(f"[Ray Mode] Connected! Total Cluster Nodes: {len(nodes)}, TPU Worker Nodes: {len(tpu_nodes)}")
    for idx, n in enumerate(tpu_nodes):
        print(f"  Worker {idx}: Node IP={n['NodeManagerAddress']}, TPUs={n['Resources'].get('TPU')}")

    if not tpu_nodes:
        raise RuntimeError("No active TPU worker nodes found in the Ray cluster!")

    @ray.remote
    class MultihostWorker:
        def __init__(self, rank: int, world_size: int):
            self.rank = rank
            self.world_size = world_size

        def run(self, coordinator_addr: str, target_tp_size: int):
            import os
            os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
            os.environ["JAX_PLATFORMS"] = "tpu,cpu"
            os.environ["ENABLE_PJRT_COMPATIBILITY"] = "true"
            
            import jax
            from jax.sharding import Mesh
            
            print(f"[Worker {self.rank}/{self.world_size}] Initializing JAX distributed (coord={coordinator_addr})...")
            jax.distributed.initialize(
                coordinator_address=coordinator_addr,
                num_processes=self.world_size,
                process_id=self.rank
            )
            
            devices = jax.devices()
            local_devices = jax.local_devices()
            print(f"[Worker {self.rank}] JAX initialized successfully. Total devices: {len(devices)}, Local devices: {len(local_devices)}")
            
            device_mesh = np.array(devices).reshape(1, -1)
            mesh = Mesh(device_mesh, axis_names=("data", "model"))
            
            run_benchmark_on_mesh(mesh, tp_size=target_tp_size)
            return True

    # Coordinator is Worker 0 IP
    coord_ip = tpu_nodes[0]["NodeManagerAddress"]
    coord_addr = f"{coord_ip}:12345"

    num_workers = len(tpu_nodes)
    print(f"[Ray Mode] Using coordinator address: {coord_addr}")
    print(f"[Ray Mode] Spawning {num_workers} multi-host worker actors...")
    
    workers = [
        MultihostWorker.options(
            resources={f"node:{node['NodeManagerAddress']}": 1, "TPU": 4}
        ).remote(i, num_workers)
        for i, node in enumerate(tpu_nodes)
    ]

    futures = [w.run.remote(coord_addr, tp_size) for w in workers]
    ray.get(futures)
    print("[Ray Mode] All multi-host workers completed successfully.")

# -----------------------------------------------------------------------------
# Mode 2: Direct JAX Distributed Coordinator
# -----------------------------------------------------------------------------
def run_jax_mode(coordinator_address: str, num_processes: int, process_id: int, tp_size: int):
    """Initializes standard JAX distributed coordinator and runs benchmark."""
    import jax
    from jax.sharding import Mesh

    print(f"[JAX Mode] Initializing process {process_id}/{num_processes} via coordinator {coordinator_address}...")
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id
    )

    devices = jax.devices()
    device_mesh = np.array(devices).reshape(1, -1)
    mesh = Mesh(device_mesh, axis_names=("data", "model"))

    run_benchmark_on_mesh(mesh, tp_size=tp_size)


def main():
    parser = argparse.ArgumentParser(description="Multi-Host Gemma-4 31B Benchmark")
    parser.add_argument("--mode", choices=["ray", "jax"], default="ray", help="Cluster orchestration mode")
    parser.add_argument("--tp-size", type=int, default=16, help="Total Tensor Parallel degree (e.g. 16 for 2 hosts of 8 chips)")
    parser.add_argument("--coordinator-address", type=str, default="127.0.0.1:12345", help="JAX coordinator address:port")
    parser.add_argument("--num-processes", type=int, default=1, help="Total number of hosts/processes")
    parser.add_argument("--process-id", type=int, default=0, help="Current host process rank (0 to num_processes-1)")
    args = parser.parse_args()

    if args.mode == "ray":
        run_ray_mode(tp_size=args.tp_size)
    else:
        run_jax_mode(args.coordinator_address, args.num_processes, args.process_id, args.tp_size)


if __name__ == "__main__":
    main()
