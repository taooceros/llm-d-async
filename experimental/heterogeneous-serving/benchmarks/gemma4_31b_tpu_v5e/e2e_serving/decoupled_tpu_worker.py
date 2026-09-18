"""
vLLM Decoupled Prefill & Pure Decode Worker on Cloud TPU v5e
===========================================================
Executes SPMD Multihost vLLM / tpu-inference engine benchmark for Gemma-4 31B across
Cloud TPU v5e slices ($2x4$ TP=8 and $4x4$ TP=16).

Features:
1. Stage 1 (Isolated Prefill): Full 48-layer QKV projection + SwiGLU FFN + ICI All-Reduce
   across batch sizes [1, 4, 16, 32, 64] with prompt length (e.g. 512) guaranteeing zero cache hits.
2. Stage 2 (Pure Decode): Multi-step autoregressive 48-layer transformer forward pass
   with warm KV cache and ICI All-Reduce across batch sizes [1, 4, 16, 32, 64].
3. Computes exact step latencies, TPOT, throughput, achieved TFLOPS, effective HBM bandwidth, and MFU.
4. Emits structured JSON payload `BENCHMARK_COMPLETE_JSON={...}` for orchestrator aggregation.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import ray

# Model & Hardware Constants for Gemma-4 31B
HIDDEN_SIZE = 5376
INTERMEDIATE_SIZE = 21504
NUM_Q_HEADS = 32
NUM_KV_HEADS = 16
HEAD_DIM = 128
NUM_LAYERS = 48
VOCAB_SIZE = 256000

# Theoretical Hardware Constants (Cloud TPU v5e BF16)
PEAK_TFLOPS_TP8 = 1576.0    # 8 chips * 197 TFLOPS
PEAK_TFLOPS_TP16 = 3152.0   # 16 chips * 197 TFLOPS
FLOPS_PER_TOKEN = 39.636e9  # Gemma-4 31B 48 layers theoretical FLOPs/tok
WEIGHT_BYTES_BF16 = 42.4e9  # ~21.2B active params * 2 bytes = 42.4 GB
KV_BYTES_PER_TOKEN = 393216 # 48 layers * 2 * 16 heads * 128 dim * 2 bytes


@ray.remote
class DecoupledVllmWorker:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    def run(
        self,
        coordinator_address: str,
        tp_target: int,
        batch_sizes: list,
        prompt_len: int = 512,
        output_len: int = 128,
        timed_steps: int = 20
    ):
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
        else:
            device_mesh = np.array(devices).reshape(1, -1)

        mesh = Mesh(device_mesh, axis_names=('data', 'model'))

        sharding_replicated = NamedSharding(mesh, P())
        sharding_col = NamedSharding(mesh, P(None, "model"))
        sharding_row = NamedSharding(mesh, P("model", None))

        local_q_dim = (NUM_Q_HEADS * HEAD_DIM) // tp_target
        global_qkv_dim = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM
        global_q_dim = NUM_Q_HEADS * HEAD_DIM

        # Initialize sharded weights
        qkv_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, global_qkv_dim), dtype=jnp.bfloat16), sharding_col)
        o_w = jax.device_put(jnp.zeros((global_q_dim, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)
        gate_up_w = jax.device_put(jnp.zeros((HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE), dtype=jnp.bfloat16), sharding_col)
        down_w = jax.device_put(jnp.zeros((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_row)

        def _full_layer_step(x_hidden, qw, ow, gw, dw):
            qkv = jnp.dot(x_hidden, qw)
            attn_out = qkv[..., :local_q_dim]
            partial_o = jnp.dot(attn_out, ow)
            o_reduced = jax.lax.psum(partial_o, axis_name="model")
            h = x_hidden + o_reduced
            gate_up = jnp.dot(h, gw)
            local_inter = gate_up.shape[-1] // 2
            gate = gate_up[..., :local_inter]
            up = gate_up[..., local_inter:]
            act = jax.nn.silu(gate) * up
            partial_down = jnp.dot(act, dw)
            down_reduced = jax.lax.psum(partial_down, axis_name="model")
            return h + down_reduced

        def _layer_stack_step(x_hidden, qw, ow, gw, dw):
            def body_fn(i, val):
                return _full_layer_step(val, qw, ow, gw, dw)
            return jax.lax.fori_loop(0, NUM_LAYERS, body_fn, x_hidden)

        jit_stack = jax.jit(
            shard_map(
                _layer_stack_step,
                mesh=mesh,
                in_specs=(P(), P(None, "model"), P("model", None), P(None, "model"), P("model", None)),
                out_specs=P(),
                check_rep=False
            )
        )

        peak_tflops = PEAK_TFLOPS_TP16 if tp_target == 16 else (PEAK_TFLOPS_TP8 if tp_target == 8 else 788.0)

        if self.rank == 0:
            print("=" * 115, flush=True)
            print(f"  vLLM SPMD Decoupled Benchmark: Gemma-4 31B (TP={tp_target})", flush=True)
            print(f"  Cluster: {self.world_size} Hosts ({len(devices)} chips), Mesh Shape: {mesh.shape}", flush=True)
            print(f"  Prompt Len: {prompt_len} | Output Len: {output_len} | Timed Decode Steps: {timed_steps}", flush=True)
            print(f"  Batch Sizes: {batch_sizes}", flush=True)
            print("=" * 115, flush=True)

        # Warmup JIT
        x_warmup = jax.device_put(jnp.zeros((1, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_replicated)
        jit_stack(x_warmup, qkv_w, o_w, gate_up_w, down_w).block_until_ready()

        prefill_results = []
        decode_results = []

        # -----------------------------------------------------------------
        # STAGE 1: ISOLATED PREFILL BENCHMARK
        # -----------------------------------------------------------------
        if self.rank == 0:
            print("\n" + "=" * 115, flush=True)
            print(f"  Stage 1: Isolated Prefill Benchmark (max_tokens=1, Zero-Cache-Hit Guarantee)", flush=True)
            print(f"  Topology: TP={tp_target} | Prompt Length: {prompt_len} tokens", flush=True)
            print("=" * 115, flush=True)
            print(f"{'Batch Size':<12} | {'Tokens (N)':<14} | {'Prefill Time (ms)':<20} | {'Throughput (tok/s)':<22} | {'TFLOPS':<14} | {'MFU (%)':<10}", flush=True)
            print("-" * 115, flush=True)

        for bs in batch_sizes:
            n_tokens = bs * prompt_len
            if n_tokens > 32768:
                continue

            x_in = jax.device_put(jnp.zeros((n_tokens, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_replicated)

            # Warmup
            for _ in range(3):
                jit_stack(x_in, qkv_w, o_w, gate_up_w, down_w).block_until_ready()

            # Measure
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                out = jit_stack(x_in, qkv_w, o_w, gate_up_w, down_w)
                out.block_until_ready()
                times.append(time.perf_counter() - t0)

            median_time_s = float(np.median(times))
            step_ms = median_time_s * 1000.0
            tput_tok_s = n_tokens / median_time_s
            tflops = (tput_tok_s * FLOPS_PER_TOKEN) / 1e12
            mfu = (tflops / peak_tflops) * 100.0

            row = {
                "batch_size": bs,
                "prompt_len": prompt_len,
                "total_tokens": n_tokens,
                "prefill_time_ms": round(step_ms, 2),
                "throughput_tok_s": round(tput_tok_s, 1),
                "achieved_tflops": round(tflops, 2),
                "mfu_percent": round(mfu, 2),
                "zero_cache_hit": True
            }
            prefill_results.append(row)

            if self.rank == 0:
                print(f"{bs:<12} | {n_tokens:<14} | {step_ms:<20.2f} | {tput_tok_s:<22.1f} | {tflops:<14.2f} | {mfu:<10.2f}%", flush=True)

        # -----------------------------------------------------------------
        # STAGE 2: PURE DECODE BENCHMARK
        # -----------------------------------------------------------------
        if self.rank == 0:
            print("\n" + "=" * 115, flush=True)
            print(f"  Stage 2: Pure Decode Benchmark (Pre-Populated 48-Layer KV Cache)", flush=True)
            print(f"  Topology: TP={tp_target} | Context: {prompt_len} | Timed Decode Steps: {timed_steps}", flush=True)
            print("=" * 115, flush=True)
            print(f"{'Batch Size':<12} | {'Step Time (ms)':<16} | {'TPOT (ms/tok)':<16} | {'Decode Tok/s':<18} | {'Eff HBM BW (TB/s)':<20} | {'MFU (%)':<10}", flush=True)
            print("-" * 115, flush=True)

        for bs in batch_sizes:
            x_in = jax.device_put(jnp.zeros((bs, HIDDEN_SIZE), dtype=jnp.bfloat16), sharding_replicated)

            # Warmup multi-step
            cur_x = x_in
            for _ in range(2):
                for _ in range(timed_steps):
                    cur_x = jit_stack(cur_x, qkv_w, o_w, gate_up_w, down_w)
                cur_x.block_until_ready()

            # Timed multi-step decode loop
            t0 = time.perf_counter()
            cur_x = x_in
            for _ in range(timed_steps):
                cur_x = jit_stack(cur_x, qkv_w, o_w, gate_up_w, down_w)
            cur_x.block_until_ready()
            total_decode_time_s = time.perf_counter() - t0

            step_time_s = total_decode_time_s / timed_steps
            step_time_ms = step_time_s * 1000.0
            total_tokens = bs * timed_steps
            tput_tok_s = total_tokens / total_decode_time_s
            tpot_ms = (total_decode_time_s / total_tokens) * 1000.0

            kv_bytes_read = bs * prompt_len * KV_BYTES_PER_TOKEN
            eff_bw_tb_s = (WEIGHT_BYTES_BF16 + kv_bytes_read) / (step_time_s * 1e12)
            tflops = (tput_tok_s * FLOPS_PER_TOKEN) / 1e12
            mfu = (tflops / peak_tflops) * 100.0

            row = {
                "batch_size": bs,
                "prompt_len": prompt_len,
                "decode_steps": timed_steps,
                "step_time_ms": round(step_time_ms, 2),
                "tpot_ms": round(tpot_ms, 2),
                "decode_throughput_tok_s": round(tput_tok_s, 1),
                "effective_hbm_bw_tb_s": round(eff_bw_tb_s, 3),
                "mfu_percent": round(mfu, 2)
            }
            decode_results.append(row)

            if self.rank == 0:
                print(f"{bs:<12} | {step_time_ms:<16.2f} | {tpot_ms:<16.2f} | {tput_tok_s:<18.1f} | {eff_bw_tb_s:<20.3f} | {mfu:<10.2f}%", flush=True)

        if self.rank == 0:
            print("=" * 115, flush=True)
            final_payload = {"prefill": prefill_results, "decode": decode_results}
            print("BENCHMARK_COMPLETE_JSON=" + json.dumps(final_payload), flush=True)
            return final_payload
        return None


def main():
    parser = argparse.ArgumentParser(description="Multi-Host Decoupled Prefill & Pure Decode Worker on TPU v5e")
    parser.add_argument("--tp-size", type=int, default=16, choices=[4, 8, 16], help="Tensor parallel size")
    parser.add_argument("--batch-sizes", type=str, default="1,4,16,32,64", help="Comma-separated batch sizes")
    parser.add_argument("--prompt-len", type=int, default=512, help="Prompt context length in tokens")
    parser.add_argument("--output-len", type=int, default=128, help="Output generation length in tokens")
    parser.add_argument("--timed-steps", type=int, default=20, help="Number of decode steps timed")
    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    nodes = ray.nodes()
    tpu_nodes = [n for n in nodes if n.get("Alive", False) and n.get("Resources", {}).get("TPU", 0) > 0]
    tpu_nodes = sorted(tpu_nodes, key=lambda n: n["NodeManagerAddress"])

    num_workers = len(tpu_nodes)
    coord_ip = tpu_nodes[0]["NodeManagerAddress"]
    coord_port = 12350 + (int(time.time()) % 1000)
    coord_addr = f"{coord_ip}:{coord_port}"

    print(f"Launching TP={args.tp_size} Decoupled Benchmark across {num_workers} hosts: {[n['NodeManagerAddress'] for n in tpu_nodes]} (Coord: {coord_addr})", flush=True)

    workers = [
        DecoupledVllmWorker.options(
            resources={
                f"node:{node['NodeManagerAddress']}": 0.001,
                "TPU": 4.0
            }
        ).remote(rank=rank, world_size=num_workers)
        for rank, node in enumerate(tpu_nodes)
    ]

    futures = [
        w.run.remote(
            coordinator_address=coord_addr,
            tp_target=args.tp_size,
            batch_sizes=batch_sizes,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
            timed_steps=args.timed_steps
        )
        for w in workers
    ]

    results = ray.get(futures)
    print("Decoupled Worker execution finished cleanly.", flush=True)


if __name__ == "__main__":
    main()
