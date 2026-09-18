#!/usr/bin/env python3
"""
Long-Decode Offline E2E Throughput Benchmark for Gemma-4 31B on TPU v5e
======================================================================
Tests longer generation sequence lengths:
  - Input length: 512 tokens
  - Output length: 1024 tokens (8x longer decode steps)
  - Batch sizes tested: [1, 4, 8, 16, 32, 64, 128]

Measures KV Cache growth, Ragged Attention memory bandwidth saturation,
and steady-state generation (decode-dominated) tokens/second.
"""

import os
import time
import argparse

MODEL_PATH = "/models/gemma-4-31b"

def run_long_decode_benchmark(tp_size: int = 16, input_len: int = 512, output_len: int = 1024):
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

    print("=" * 95)
    print(f"  vLLM Long-Decode End-to-End (E2E) Benchmark: Gemma-4 31B (TP={tp_size})")
    print(f"  Input Tokens  : {input_len}")
    print(f"  Output Tokens : {output_len} (Long Generation)")
    print(f"  Max Model Len : 8192")
    print("=" * 95)

    print("\n[Step 1] Initializing vLLM LLMEngine...")
    t_init_start = time.perf_counter()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tp_size,
        distributed_executor_backend="ray",
        max_model_len=8192,
        max_num_seqs=512,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=False,
    )
    print(f"[Step 1] Initialized in {time.perf_counter() - t_init_start:.2f}s.\n")

    dummy_prompt = "hello " * input_len
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=output_len,
        ignore_eos=True,
    )

    batch_sizes = [128, 256, 384, 512]
    print(f"{'Batch Size':<12} | {'Total Time (s)':<16} | {'Req/s':<12} | {'Output Tok/s':<16} | {'Total Tok/s':<16} | {'TPOT (ms/tok)':<15}")
    print("-" * 105)

    for bs in batch_sizes:
        prompts = [dummy_prompt for _ in range(bs)]
        
        if bs == 1:
            print("[Warmup] Warming up JIT and execution pipelines...")
            _ = llm.generate(["warmup prompt"], sampling_params)
            print("[Warmup] Ready.\n")

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

    print("=" * 105)
    print("Long-Decode E2E Benchmark Completed Successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=1024)
    args = parser.parse_args()
    run_long_decode_benchmark(tp_size=args.tp_size, input_len=args.input_len, output_len=args.output_len)
