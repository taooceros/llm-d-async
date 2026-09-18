"""
Complete Multi-Topology Offline vLLM E2E Benchmark for Gemma-4 31B on Cloud TPU v5e
Matches kernel_microbenchmark.py node discovery and JAX PJRT initialization exactly.
"""
import os
import time
import json
import argparse
import numpy as np
import ray


@ray.remote
class MultihostVllmWorker:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    def run(
        self,
        coordinator_address: str,
        tp_target: int,
        prompt_len: int = 512,
        batch_sizes: list = None
    ):
        if batch_sizes is None:
            batch_sizes = [1, 4, 16, 32, 64, 128, 256]

        import os
        os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
        os.environ["JAX_PLATFORMS"] = "tpu,cpu"
        os.environ["ENABLE_PJRT_COMPATIBILITY"] = "true"
        os.environ["JAX_COMPILATION_CACHE_DIR"] = "/models/jax_cache"

        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=self.world_size,
            process_id=self.rank,
            initialization_timeout=300
        )

        devices = jax.devices()
        if tp_target == 16:
            device_mesh = np.array(devices).reshape(1, -1)
        elif tp_target == 8:
            device_mesh = np.array(devices).reshape(2, 8)
        elif tp_target == 4:
            device_mesh = np.array(devices).reshape(4, 4)
        mesh = Mesh(device_mesh, axis_names=('data', 'model'))

        from vllm.config import VllmConfig, ModelConfig, ParallelConfig, CacheConfig, LoadConfig
        from tpu_inference.models.common.model_loader import get_model
        from tpu_inference.runner.tpu_runner import AttentionMetadata
        from tpu_inference.distributed.jax_parallel_state import init_pp_distributed_environment

        init_pp_distributed_environment(
            ip="127.0.0.1",
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
            tensor_parallel_size=tp_target,
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

        model_cls = get_model(vllm_config)
        model = model_cls(vllm_config, mesh=mesh)
        state_leaves = model.get_weights()

        kv_cache_spec = NamedSharding(mesh, P(None, None, "model", None))
        kv_caches = [
            jax.device_put(
                jnp.zeros((1024, 512, 16 // tp_target, 128), dtype=jnp.bfloat16),
                kv_cache_spec
            )
            for _ in range(48)
        ]

        @jax.jit
        def run_step(state, kv, input_tokens, meta):
            return model(state, kv, input_tokens, meta)

        if self.rank == 0:
            print("=" * 115, flush=True)
            print(f"  Offline vLLM E2E Benchmark: Gemma-4 31B (TP={tp_target})", flush=True)
            print(f"  Cluster: {self.world_size} Hosts, Mesh: {mesh.shape}", flush=True)
            print(f"  Prompt Length: {prompt_len} tokens", flush=True)
            print("=" * 115, flush=True)
            print(f"{'Phase':<10} | {'Batch':<8} | {'Tokens (N)':<12} | {'Step Time (ms)':<16} | {'Throughput (tok/s)':<22} | {'TPOT (ms/tok)':<15}", flush=True)
            print("-" * 115, flush=True)

        prefill_results = []
        decode_results = []

        # 1. PREFILL PHASE BENCHMARK
        for bs in batch_sizes:
            n_tokens = bs * prompt_len
            if n_tokens > 16384:
                continue

            prompt_input_ids = jax.device_put(jnp.ones((n_tokens, 1), dtype=jnp.int32), NamedSharding(mesh, P(None, None)))
            prompt_meta = AttentionMetadata(
                is_prompt=True,
                block_tables=jax.device_put(jnp.zeros((bs, 16), dtype=jnp.int32), NamedSharding(mesh, P(None, None))),
                seq_lens=jax.device_put(jnp.full((bs,), prompt_len, dtype=jnp.int32), NamedSharding(mesh, P(None, None))),
                query_start_loc=jax.device_put(jnp.arange(0, n_tokens + 1, prompt_len, dtype=jnp.int32), NamedSharding(mesh, P(None, None))),
            )

            # Warmup
            out = run_step(state_leaves, kv_caches, prompt_input_ids, prompt_meta)
            out[1].block_until_ready()

            iters = 10
            t0 = time.perf_counter()
            for _ in range(iters):
                out = run_step(state_leaves, kv_caches, prompt_input_ids, prompt_meta)
                out[1].block_until_ready()
            elapsed = time.perf_counter() - t0

            step_ms = (elapsed / iters) * 1000.0
            tput_tok_s = n_tokens / (step_ms / 1000.0)

            prefill_results.append({
                "batch_size": bs,
                "tokens": n_tokens,
                "step_ms": step_ms,
                "throughput": tput_tok_s
            })
            if self.rank == 0:
                print(f"{'Prefill':<10} | {bs:<8} | {n_tokens:<12} | {step_ms:<16.2f} | {tput_tok_s:<22.1f} | {'—':<15}", flush=True)

        if self.rank == 0:
            print("-" * 115, flush=True)

        # 2. DECODE PHASE BENCHMARK
        for bs in batch_sizes:
            decode_input_ids = jax.device_put(jnp.ones((bs, 1), dtype=jnp.int32), NamedSharding(mesh, P(None, None)))
            decode_meta = AttentionMetadata(
                is_prompt=False,
                block_tables=jax.device_put(jnp.zeros((bs, 16), dtype=jnp.int32), NamedSharding(mesh, P(None, None))),
                seq_lens=jax.device_put(jnp.full((bs,), prompt_len, dtype=jnp.int32), NamedSharding(mesh, P(None, None))),
                query_start_loc=jax.device_put(jnp.arange(0, bs + 1, dtype=jnp.int32), NamedSharding(mesh, P(None, None))),
            )

            # Warmup
            out = run_step(state_leaves, kv_caches, decode_input_ids, decode_meta)
            out[1].block_until_ready()

            iters = 25
            t0 = time.perf_counter()
            for _ in range(iters):
                out = run_step(state_leaves, kv_caches, decode_input_ids, decode_meta)
                out[1].block_until_ready()
            elapsed = time.perf_counter() - t0

            step_ms = (elapsed / iters) * 1000.0
            tput_tok_s = bs / (step_ms / 1000.0)
            tpot_ms = step_ms

            decode_results.append({
                "batch_size": bs,
                "step_ms": step_ms,
                "throughput": tput_tok_s,
                "tpot_ms": tpot_ms
            })
            if self.rank == 0:
                print(f"{'Decode':<10} | {bs:<8} | {bs:<12} | {step_ms:<16.2f} | {tput_tok_s:<22.1f} | {tpot_ms:<15.2f}", flush=True)

        if self.rank == 0:
            print("=" * 115, flush=True)
            print("BENCHMARK_COMPLETE_JSON=" + json.dumps({"prefill": prefill_results, "decode": decode_results}), flush=True)

        return {"prefill": prefill_results, "decode": decode_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=16, choices=[4, 8, 16])
    args = parser.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)
    nodes = ray.nodes()
    tpu_nodes = [n for n in nodes if n.get("Alive", False) and n.get("Resources", {}).get("TPU", 0) > 0]
    tpu_nodes = sorted(tpu_nodes, key=lambda n: n["NodeManagerAddress"])

    num_workers = len(tpu_nodes)
    coord_ip = tpu_nodes[0]["NodeManagerAddress"]
    coord_addr = f"{coord_ip}:12348"

    print(f"Launching TP={args.tp_size} actor benchmark across all {num_workers} cluster hosts: {[n['NodeManagerAddress'] for n in tpu_nodes]}", flush=True)

    workers = [
        MultihostVllmWorker.options(
            resources={f"node:{node['NodeManagerAddress']}": 1, "TPU": 4}
        ).remote(i, num_workers)
        for i, node in enumerate(tpu_nodes)
    ]

    futures = [w.run.remote(coord_addr, args.tp_size) for w in workers]
    results = ray.get(futures)
    print("Benchmark finished cleanly.", flush=True)


if __name__ == "__main__":
    main()
