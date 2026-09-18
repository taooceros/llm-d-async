"""Multihost TPU Ray launcher for Decode Step Microbenchmark across 16 chips.
Evaluates single-step decode latency across batch sizes and compares against E2E vLLM serving.
"""
import os
import time
import argparse
import numpy as np
import ray

@ray.remote(resources={"TPU": 4})
def tpu_worker_benchmark_task(coordinator_address: str, process_id: int, num_processes: int):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jax.experimental.shard_map import shard_map

    print(f"[Worker {process_id}/{num_processes}] Initializing JAX distributed (coordinator={coordinator_address})...")
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        initialization_timeout=300
    )

    devices = jax.devices()
    process_devices = jax.local_devices()
    print(f"[Worker {process_id}] Total cluster devices: {len(devices)} | Local devices: {len(process_devices)}")

    mesh_devices = np.array(devices).reshape(4, 4)
    mesh = Mesh(mesh_devices, axis_names=('hosts', 'chips'))

    # Gemma-4 31B specs
    HIDDEN_SIZE = 5376
    INTERMEDIATE_SIZE = 21504
    NUM_Q_HEADS = 32
    NUM_KV_HEADS = 16
    HEAD_DIM = 128
    VOCAB_SIZE = 256000
    NUM_LAYERS = 48

    sharding_col = NamedSharding(mesh, P(None, ("hosts", "chips")))
    sharding_row = NamedSharding(mesh, P(("hosts", "chips"), None))
    sharding_replicated = NamedSharding(mesh, P(None, None))

    global_qkv_dim = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
    global_q_dim = NUM_Q_HEADS * HEAD_DIM

    # Single-step decode execution map
    def decode_layer_fn(x, w_qkv, w_o, w_gate_up, w_down, k_cache, v_cache, w_lm_head):
        # 1. QKV GEMM
        qkv = jnp.dot(x, w_qkv)
        # 2. Paged Attention simulation (Gather KV + Score + Softmax + Reduce)
        q = qkv[:, :global_q_dim // 16]
        scores = jnp.einsum('b h d, b s h d -> b h s', q.reshape(x.shape[0], -1, HEAD_DIM), k_cache)
        attn_out = jnp.einsum('b h s, b s h d -> b h d', jax.nn.softmax(scores, axis=-1), v_cache).reshape(x.shape[0], -1)
        # 3. O-Proj All-Reduce
        o_proj = jax.lax.psum(jnp.dot(attn_out, w_o), axis="model")
        x_post_attn = x + o_proj
        # 4. SwiGLU FFN
        gate_up = jnp.dot(x_post_attn, w_gate_up)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        ffn_act = jax.nn.silu(gate) * up
        # 5. Down-Proj All-Reduce
        down_proj = jax.lax.psum(jnp.dot(ffn_act, w_down), axis="model")
        x_out = x_post_attn + down_proj
        # 6. LM Head Logits (for last layer / sampling)
        logits_shard = jnp.dot(x_out, w_lm_head)
        return x_out, logits_shard

    sharded_step = jax.jit(
        shard_map(
            decode_layer_fn,
            mesh=mesh,
            in_specs=(
                P(None, None), # x
                P(None, ("hosts", "chips")), # w_qkv
                P(("hosts", "chips"), None), # w_o
                P(None, ("hosts", "chips")), # w_gate_up
                P(("hosts", "chips"), None), # w_down
                P(None, None, ("hosts", "chips")), # k_cache
                P(None, None, ("hosts", "chips")), # v_cache
                P(None, ("hosts", "chips")), # w_lm_head
            ),
            out_specs=(P(None, None), P(None, ("hosts", "chips"))),
            check_rep=False,
        )
    )

    qkv_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, global_qkv_dim), dtype=jnp.bfloat16), sharding_col)
    o_w = jax.device_put(jnp.zeros((global_q_dim, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)
    gate_up_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE), dtype=jnp.bfloat16), sharding_col)
    down_w = jax.device_put(jnp.zeros((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)
    lm_head_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, VOCAB_SIZE), dtype=jnp.bfloat16), sharding_col)

    test_batches = [1, 4, 16, 32, 64, 128, 256]
    results = []

    if process_id == 0:
        print("\n" + "=" * 95)
        print("Gemma-4 31B Decode Step Microbenchmark (TP=16, Cloud TPU v5e-16)")
        print(f"{'Batch Size':<12} | {'Step Latency (ms)':<20} | {'Pure Decode Tok/s':<22} | {'Backbone TPOT (ms/tok)':<24}")
        print("-" * 95)

    for bs in test_batches:
        x_hidden = jax.device_put(jnp.zeros((bs, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_replicated)
        kv_k = jax.device_put(jnp.zeros((bs, 512, global_q_dim // 16, HEAD_DIM), dtype=jnp.bfloat16), NamedSharding(mesh, P(None, None, ("hosts", "chips"), None)))
        kv_v = jax.device_put(jnp.zeros((bs, 512, global_q_dim // 16, HEAD_DIM), dtype=jnp.bfloat16), NamedSharding(mesh, P(None, None, ("hosts", "chips"), None)))

        # Warmup
        for _ in range(5):
            out, _ = sharded_step(x_hidden, qkv_w, o_w, gate_up_w, down_w, kv_k, kv_v, lm_head_w)
            jax.block_until_ready(out)

        # Bench
        iters = 30
        t0 = time.perf_counter()
        for _ in range(iters):
            out, _ = sharded_step(x_hidden, qkv_w, o_w, gate_up_w, down_w, kv_k, kv_v, lm_head_w)
            jax.block_until_ready(out)
        t_layer_ms = ((time.perf_counter() - t0) / iters) * 1000.0
        total_model_step_ms = t_layer_ms * NUM_LAYERS
        decode_tok_s = bs / (total_model_step_ms / 1000.0)

        results.append({
            "batch_size": bs,
            "step_latency_ms": total_model_step_ms,
            "decode_tok_s": decode_tok_s,
            "tpot_ms": total_model_step_ms / bs if bs > 0 else 0,
        })

        if process_id == 0:
            print(f"{bs:<12} | {total_model_step_ms:<20.2f} | {decode_tok_s:<22.1f} | {total_model_step_ms / bs:<24.2f}")

    if process_id == 0:
        print("=" * 95)

    return results

def main():
    ray.init(address="auto", ignore_reinit_error=True)
    tpu_nodes = [node for node in ray.nodes() if node.get("alive") and node.get("Resources", {}).get("TPU", 0) >= 4]
    num_processes = len(tpu_nodes)
    coordinator_address = f"{tpu_nodes[0]['NodeManagerAddress']}:8471"

    print(f"Dispatched across {num_processes} TPU host workers. Coordinator: {coordinator_address}")

    futures = []
    for i, node in enumerate(tpu_nodes):
        node_ip = node["NodeManagerAddress"]
        f = tpu_worker_benchmark_task.options(resources={f"node:{node_ip}": 0.001, "TPU": 4}).remote(
            coordinator_address, i, num_processes
        )
        futures.append(f)

    results = ray.get(futures)
    print("Microbenchmark job finished cleanly.")

if __name__ == "__main__":
    main()
