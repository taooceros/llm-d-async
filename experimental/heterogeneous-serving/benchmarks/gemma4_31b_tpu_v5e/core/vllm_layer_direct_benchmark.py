"""Direct TPU Gemma-4 31B Layer & Backbone microbenchmark using vLLM's internal JAX/Flax NNX definitions.
Evaluates single-step decode latency & throughput across batch sizes using multihost Ray workers.
"""
import os
import time
import numpy as np
import ray

@ray.remote(resources={"TPU": 4})
def tpu_worker_vllm_layer_bench(coordinator_address: str, process_id: int, num_processes: int, host_ip: str):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        initialization_timeout=300
    )

    devices = jax.devices()
    mesh_devices = np.array(devices).reshape(1, 16)
    mesh = Mesh(mesh_devices, axis_names=('data', 'model'))

    # Import vLLM config structures and TPU parallel state
    from vllm.config import VllmConfig, ModelConfig, ParallelConfig, CacheConfig, LoadConfig
    from tpu_inference.models.common.model_loader import get_model
    from tpu_inference.runner.tpu_runner import AttentionMetadata
    from tpu_inference.distributed.jax_parallel_state import init_pp_distributed_environment

    # Initialize PP group state (world_size=1, need_pp=False for TP-only)
    init_pp_distributed_environment(
        ip=host_ip,
        rank=0,
        world_size=1,
        device=jax.local_devices()[0],
        need_pp=False
    )

    model_config = ModelConfig(
        model="/models/gemma-4-31b",
        tokenizer="/models/gemma-4-31b",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
        max_model_len=8192,
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=16,
    )
    cache_config = CacheConfig(
        block_size=512,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    load_config = LoadConfig(
        load_format="jax_dummy",
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=cache_config,
        load_config=load_config,
    )

    rng_key = jax.random.PRNGKey(0)

    if process_id == 0:
        print("\n" + "=" * 110)
        print("vLLM Production JAX/Flax NNX Model Multi-Host Single-Step Decode Microbenchmark (TP=16)")
        print(f"JAX backend: {jax.default_backend()} | Cluster devices: {len(devices)} | Mesh: {mesh.shape} {mesh.axis_names}")
        print("=" * 110)

    # Initialize complete model using vLLM/TPU inference framework
    model_bundle = get_model(vllm_config, rng_key, mesh)
    model_fn = model_bundle.model_fn
    state_leaves = model_bundle.state_leaves

    batch_sizes = [1, 4, 16, 32, 64, 128, 256]
    results = []

    if process_id == 0:
        print(f"{'Batch Size':<12} | {'Step Latency (ms)':<22} | {'Direct Model Tok/s':<24} | {'Step TPOT (ms/t)':<20}")
        print("-" * 100)

    for bs in batch_sizes:
        # Replicated input token ids [bs, 1]
        input_ids = jax.device_put(jnp.ones((bs, 1), dtype=jnp.int32), NamedSharding(mesh, P(None, None)))
        positions = jax.device_put(jnp.full((bs, 1), 512, dtype=jnp.int32), NamedSharding(mesh, P(None, None)))

        # Allocate KV cache for 48 layers: list of 48 tensors
        # Each layer tensor: [num_blocks, block_size, num_kv_heads, head_dim]
        # (1024 blocks * 512 = 524,288 tokens capacity across cluster)
        kv_cache_spec = NamedSharding(mesh, P(None, None, 'model', None))
        kv_caches = [
            jax.device_put(
                jnp.zeros((1024, 512, 16 // 16, 128), dtype=jnp.bfloat16),
                kv_cache_spec
            )
            for _ in range(48)
        ]

        attn_metadata = AttentionMetadata(
            is_prompt=False,
            block_tables=jax.device_put(jnp.zeros((bs, 16), dtype=jnp.int32), NamedSharding(mesh, P(None, None))),
            seq_lens=jax.device_put(jnp.full((bs,), 512, dtype=jnp.int32), NamedSharding(mesh, P(None, None))),
            query_start_loc=jax.device_put(jnp.arange(0, bs + 1, dtype=jnp.int32), NamedSharding(mesh, P(None, None))),
        )

        # Call model_fn
        out = model_fn(state_leaves, kv_caches, input_ids, attn_metadata)
        out[1].block_until_ready()

        # Bench iterations
        iters = 30
        start = time.perf_counter()
        for _ in range(iters):
            out = model_fn(state_leaves, kv_caches, input_ids, attn_metadata)
            out[1].block_until_ready()
        elapsed = time.perf_counter() - start

        step_ms = (elapsed / iters) * 1000.0
        direct_tok_s = bs / (step_ms / 1000.0)
        tpot_ms = step_ms / bs

        results.append({
            "batch_size": bs,
            "step_ms": step_ms,
            "direct_tok_s": direct_tok_s,
            "tpot_ms": tpot_ms,
        })
        if process_id == 0:
            print(f"{bs:<12} | {step_ms:<22.2f} | {direct_tok_s:<24.1f} | {tpot_ms:<20.2f}")

    if process_id == 0:
        print("=" * 100)

    return results

def main():
    ray.init(address="auto", ignore_reinit_error=True)
    tpu_nodes = [node for node in ray.nodes() if node.get("alive") and node.get("Resources", {}).get("TPU", 0) >= 4]
    num_processes = len(tpu_nodes)
    coordinator_address = f"{tpu_nodes[0]['NodeManagerAddress']}:8471"

    futures = []
    for i, node in enumerate(tpu_nodes):
        node_ip = node["NodeManagerAddress"]
        f = tpu_worker_vllm_layer_bench.options(resources={f"node:{node_ip}": 0.001, "TPU": 4}).remote(
            coordinator_address, i, num_processes, node_ip
        )
        futures.append(f)

    results = ray.get(futures)
    print("Direct vLLM Model Microbenchmark finished cleanly.")

if __name__ == "__main__":
    main()
