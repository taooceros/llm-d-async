#!/usr/bin/env python3
"""
Offline End-to-End Batch Inference Throughput & Concurrency Benchmark
=====================================================================
Benchmarks full vLLM pipeline (Tokenizer -> PagedAttention -> Backbone -> Sampler -> Detokenizer)
across custom input/output lengths and concurrency batch sizes.
"""

import os
import time
import argparse
import json

MODEL_PATH = "/models/gemma-4-31b"


def run_e2e_benchmark(tp_size: int = 16, input_len: int = 512, output_len: int = 128, batch_sizes=None):
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]

    if "VLLM_HOST_IP" in os.environ:
        del os.environ["VLLM_HOST_IP"]

    os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
    os.environ["TPU_CHIPS_PER_HOST"] = "4"
    os.environ["SCAN_TPU_CHIPS"] = "True"
    os.environ["TPU_ACCELERATOR_TYPE"] = "v5e"
    os.environ["TPU_TOPOLOGY"] = "4x4"
    os.environ["JAX_PLATFORMS"] = ""
    os.environ["JAX_COMPILATION_CACHE_DIR"] = "/models/jax_cache"

    from vllm import LLM, SamplingParams

    print("=" * 105)
    print(f"  vLLM E2E Offline Batch Inference Benchmark: Gemma-4 31B (TP={tp_size})")
    print(f"  Input Tokens  : {input_len}")
    print(f"  Output Tokens : {output_len}")
    print(f"  Batch Sizes   : {batch_sizes}")
    print("=" * 105)

    print("\n[Step 1] Initializing vLLM Engine...")
    t0_init = time.perf_counter()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tp_size,
        distributed_executor_backend="ray",
        max_model_len=8192,
        max_num_seqs=max(batch_sizes),
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=False,
    )
    print(f"[Step 1] Engine ready in {time.perf_counter() - t0_init:.2f}s.\n")

    dummy_prompt = "explain TPU architecture in detail " * (input_len // 5)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=output_len,
        ignore_eos=True,
    )

    print(f"{'Batch Size':<12} | {'Total Time (s)':<16} | {'Req/s':<12} | {'Output Tok/s':<16} | {'Total Tok/s':<16} | {'TPOT (ms/tok)':<15}")
    print("-" * 105)

    results = []

    for bs in batch_sizes:
        prompts = [dummy_prompt for _ in range(bs)]
        
        if bs == batch_sizes[0]:
            print("[Warmup] Executing warmup JIT pass...")
            _ = llm.generate(["warmup prompt"], sampling_params)
            print("[Warmup] Warmup completed.\n")

        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        t_elapsed = time.perf_counter() - t0

        total_gen_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        total_prompt_tokens = sum(len(out.prompt_token_ids) for out in outputs)
        total_tokens = total_gen_tokens + total_prompt_tokens

        req_per_sec = bs / t_elapsed
        out_tokens_per_sec = total_gen_tokens / t_elapsed
        total_tokens_per_sec = total_tokens / t_elapsed
        tpot_ms = (t_elapsed / output_len) * 1000.0

        print(f"{bs:<12} | {t_elapsed:<16.3f} | {req_per_sec:<12.2f} | {out_tokens_per_sec:<16.2f} | {total_tokens_per_sec:<16.2f} | {tpot_ms:<15.2f}")
        results.append({
            "batch_size": bs,
            "total_time_s": t_elapsed,
            "req_per_sec": req_per_sec,
            "output_tokens_per_sec": out_tokens_per_sec,
            "total_tokens_per_sec": total_tokens_per_sec,
            "tpot_ms": tpot_ms
        })

    print("=" * 105)
    print("E2E Batch Benchmark Completed Successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128, 256])
    args = parser.parse_args()

    run_e2e_benchmark(
        tp_size=args.tp_size,
        input_len=args.input_len,
        output_len=args.output_len,
        batch_sizes=args.batch_sizes
    )
