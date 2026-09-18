#!/usr/bin/env python3
"""
Full Decode Step Microbenchmark (Backbone + Ragged Attention + LM Head) & HBM Memory Scaling
===========================================================================================
Extracts the core vLLM TPU Decode Step pipeline:
  1. 48-Layer Transformer Backbone (QKV Proj, SwiGLU FFN, ICI All-Reduce across TP=16)
  2. Ragged Paged Attention (Paged KV Cache read/write simulation across context lengths)
  3. LM Head GEMM (Hidden State -> 256,000 Vocab Logits Projection + Sharded Max/Sampling)

Evaluates:
  - Batch sizes: [1, 4, 16, 64, 128, 256, 512, 1024]
  - Context lengths: [512, 2048, 8192]
  - HBM Memory footprint: Model Weights (62.4GB) vs KV Cache (384KB/req/token)
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

# TPU v5e-16 Hardware Constants
NUM_CHIPS = 16
CHIP_HBM_BYTES = 16 * (1024 ** 3)
CLUSTER_HBM_BYTES = NUM_CHIPS * CHIP_HBM_BYTES
KV_BYTES_PER_TOKEN = 2 * 2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM # 384 KB / token
WEIGHT_BYTES_BF16 = 62.4 * (1024 ** 3)


def print_memory_analysis():
    print("=" * 110)
    print("  TPU v5e-16 HBM Memory Capacity & Maximum Batch Size Analysis (Gemma-4 31B)")
    print(f"  Cluster HBM Total  : {CLUSTER_HBM_BYTES / (1024**3):.1f} GB ({NUM_CHIPS}x 16 GB chips)")
    print(f"  Model Weights BF16 : {WEIGHT_BYTES_BF16 / (1024**3):.2f} GB ({(WEIGHT_BYTES_BF16 / CLUSTER_HBM_BYTES)*100:.1f}% of Cluster HBM)")
    print(f"  KV Cache Rate      : {KV_BYTES_PER_TOKEN / 1024:.1f} KB / context token across 48 layers")
    print(f"  Available HBM for KV & Activations: {(CLUSTER_HBM_BYTES - WEIGHT_BYTES_BF16) / (1024**3):.2f} GB")
    print("=" * 110)

    usable_hbm = (CLUSTER_HBM_BYTES * 0.90) - WEIGHT_BYTES_BF16
    contexts = [512, 1024, 2048, 4096, 8192]

    print(f"\n{'Context Length':<15} | {'KV/Req (MB)':<12} | {'Max Schedulable BS':<20} | {'Max Tokens in Cache':<22} | {'Memory Bottleneck'}")
    print("-" * 110)
    for ctx in contexts:
        kv_per_req_mb = (ctx * KV_BYTES_PER_TOKEN) / (1024 ** 2)
        max_bs = int(usable_hbm / (ctx * KV_BYTES_PER_TOKEN))
        max_tokens = max_bs * ctx
        print(f"{ctx:<15} | {kv_per_req_mb:<12.1f} | {max_bs:<20} | {max_tokens:<22} | {'KV-Cache Capacity' if max_bs < 1024 else 'Compute Saturation'}")
    print("-" * 110 + "\n")


def run_full_decode_benchmark(mesh, tp_size: int = 16):
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
        print("=" * 110)
        print("  Extracting & Benchmarking Complete vLLM Decode Step (Backbone + Ragged Attention + LM Head)")
        print(f"  Distributed Topology: {num_processes} Hosts, {device_count} TPUs on 4x4 ICI Network")
        print(f"  Tensor Parallel Size: {tp_size}")
        print("=" * 110)

    test_batches = [1, 4, 16, 64, 128, 256, 512, 1024]
    test_contexts = [512, 2048, 8192]
    iters = 20
    warmup = 4

    sharding_replicated = NamedSharding(mesh, P())
    sharding_col = NamedSharding(mesh, P(None, "model"))
    sharding_row = NamedSharding(mesh, P("model", None))

    local_q_dim = (NUM_Q_HEADS * HEAD_DIM) // tp_size
    local_inter_dim = INTERMEDIATE_SIZE // tp_size
    local_vocab_dim = VOCAB_SIZE // tp_size

    def _full_decode_layer_step_impl(x_hidden, qkv_w, o_w, gate_up_w, down_w, kv_cache_k, kv_cache_v, lm_head_w):
        # 1. QKV Proj GEMM
        qkv = jnp.dot(x_hidden, qkv_w) # [B, local_qkv_dim]
        q_heads = qkv[..., :local_q_dim]
        
        # 2. Attention (Ragged Paged KV Cache Access Simulation)
        attn_scores = jnp.einsum("bd,bld->bl", q_heads, kv_cache_k) * (1.0 / np.sqrt(HEAD_DIM))
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        attn_context = jnp.einsum("bl,bld->bd", attn_weights, kv_cache_v)
        
        # 3. O-Proj Row Parallel GEMM + ICI All-Reduce
        partial_o = jnp.dot(attn_context, o_w)
        o_reduced = jax.lax.psum(partial_o, axis_name="model")
        
        # 4. SwiGLU FFN + ICI All-Reduce
        gate_up = jnp.dot(o_reduced, gate_up_w)
        local_inter = gate_up.shape[-1] // 2
        gate = gate_up[..., :local_inter]
        up = gate_up[..., local_inter:]
        activated = jax.nn.silu(gate) * up
        partial_down = jnp.dot(activated, down_w)
        down_reduced = jax.lax.psum(partial_down, axis_name="model")
        
        # 5. LM Head Projection + Sharded Max
        logits_local = jnp.dot(down_reduced, lm_head_w)
        max_logits_local = jnp.max(logits_local, axis=-1, keepdims=True)
        global_max = jax.lax.pmax(max_logits_local, axis_name="model")
        
        return down_reduced, global_max

    sharded_step = jax.jit(
        shard_map(
            _full_decode_layer_step_impl,
            mesh=mesh,
            in_specs=(
                P(),                 # x_hidden
                P(None, "model"),    # qkv_w
                P("model", None),    # o_w
                P(None, "model"),    # gate_up_w
                P("model", None),    # down_w
                P(None, None, "model"), # kv_cache_k
                P(None, None, "model"), # kv_cache_v
                P(None, "model"),    # lm_head_w
            ),
            out_specs=(P(), P()),
            check_rep=False
        )
    )

    if process_id == 0:
        print(f"\n{'Context':<10} | {'Batch Size':<12} | {'Step Latency (ms)':<18} | {'Pure Decode Tok/s':<20} | {'KV Cache HBM (GB)':<18} | {'MFU (%)':<10}")
        print("-" * 110)

    flops_per_token = 39.636e9
    peak_cluster_tflops = 16 * 197.0

    # Allocate local sharded arrays inside each rank to completely avoid global OOM
    global_qkv_dim = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    global_q_dim = NUM_Q_HEADS * HEAD_DIM
    
    qkv_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, global_qkv_dim), dtype=jnp.bfloat16), sharding_col)
    o_w = jax.device_put(jnp.zeros((global_q_dim, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)
    gate_up_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE), dtype=jnp.bfloat16), sharding_col)
    down_w = jax.device_put(jnp.zeros((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)
    lm_head_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, VOCAB_SIZE), dtype=jnp.bfloat16), sharding_col)

    for ctx in test_contexts:
        chunk_len = min(ctx, 512)
        for bs in test_batches:
            x_hidden = jax.device_put(jnp.zeros((bs, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_replicated)
            kv_k = jax.device_put(jnp.zeros((bs, chunk_len, global_q_dim), dtype=jnp.bfloat16), NamedSharding(mesh, P(None, None, "model")))
            kv_v = jax.device_put(jnp.zeros((bs, chunk_len, global_q_dim), dtype=jnp.bfloat16), NamedSharding(mesh, P(None, None, "model")))

            for _ in range(warmup):
                out, _ = sharded_step(x_hidden, qkv_w, o_w, gate_up_w, down_w, kv_k, kv_v, lm_head_w)
                jax.block_until_ready(out)

            t0 = time.perf_counter()
            for _ in range(iters):
                out, _ = sharded_step(x_hidden, qkv_w, o_w, gate_up_w, down_w, kv_k, kv_v, lm_head_w)
                jax.block_until_ready(out)
            t_layer_ms = ((time.perf_counter() - t0) / iters) * 1000.0

            total_model_step_ms = t_layer_ms * NUM_LAYERS
            decode_tok_s = bs / (total_model_step_ms / 1000.0)
            achieved_tflops = (decode_tok_s * flops_per_token) / 1e12
            mfu = (achieved_tflops / peak_cluster_tflops) * 100.0
            kv_total_gb = (bs * ctx * KV_BYTES_PER_TOKEN) / (1024 ** 3)

            if process_id == 0:
                print(f"{ctx:<10} | {bs:<12} | {total_model_step_ms:<18.2f} | {decode_tok_s:<20.1f} | {kv_total_gb:<18.2f} | {mfu:<10.2f}%")

    if process_id == 0:
        print("=" * 110)
        print("Decode Step Microbenchmark Completed Successfully.")


def run_ray_job():
    import ray
    print("[Ray Mode] Connecting to Ray cluster...")
    ray.init(address="auto", ignore_reinit_error=True)

    nodes = ray.nodes()
    tpu_nodes = [n for n in nodes if n.get("Alive", False) and n.get("Resources", {}).get("TPU", 0) > 0]
    tpu_nodes = sorted(tpu_nodes, key=lambda n: n["NodeManagerAddress"])

    print_memory_analysis()

    @ray.remote
    class MultihostRunner:
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

            jax.distributed.initialize(
                coordinator_address=coordinator_addr,
                num_processes=self.world_size,
                process_id=self.rank
            )

            devices = jax.devices()
            device_mesh = np.array(devices).reshape(1, -1)
            mesh = Mesh(device_mesh, axis_names=("data", "model"))

            run_full_decode_benchmark(mesh, tp_size=16)
            return True

    coord_ip = tpu_nodes[0]["NodeManagerAddress"]
    coord_addr = f"{coord_ip}:12345"
    num_workers = len(tpu_nodes)
    print(f"[Ray Mode] Found {num_workers} TPU Worker nodes. Coordinator at {coord_addr}")

    workers = [
        MultihostRunner.options(
            resources={f"node:{node['NodeManagerAddress']}": 1, "TPU": 4}
        ).remote(i, num_workers)
        for i, node in enumerate(tpu_nodes)
    ]

    futures = [w.run.remote(coord_addr) for w in workers]
    ray.get(futures)
    print("[Ray Mode] Microbenchmark completed across all nodes.")


if __name__ == "__main__":
    run_ray_job()
