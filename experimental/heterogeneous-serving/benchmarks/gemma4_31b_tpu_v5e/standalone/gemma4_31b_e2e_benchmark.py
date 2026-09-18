#!/usr/bin/env python3
"""
Offline E2E Throughput Benchmark for Gemma-4 31B on TPU v5e
==========================================================
Benchmarks end-to-end vLLM offline inference across multiple batch sizes:
  - Input length: 512 tokens
  - Output length: 128 tokens
  - Batch sizes tested: [1, 4, 8, 16, 32, 64, 128, 256]

Calculates:
  - E2E Request Throughput (req/s)
  - E2E Token Generation Throughput (output tokens/s)
  - E2E Total Token Throughput (input+output tokens/s)
"""

import os
import time
import argparse

MODEL_PATH = "/models/gemma-4-31b"

def run_e2e_batch_benchmark(tp_size: int = 16):
    # Clean VLLM_HOST_IP to prevent placement group misconfiguration on head node
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
    print("\n[Step 1] Initializing vLLM LLMEngine on TPU Ray Cluster...")
    t_init_start = time.perf_counter()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tp_size,
        distributed_executor_backend="ray",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=False,
    )
    t_init_end = time.perf_counter()
    print(f"[Step 1] vLLM Engine Initialized in {t_init_end - t_init_start:.2f} seconds.\n")

    # Fixed Prompt & Generation Length
    input_len = 512
    output_len = 128
    dummy_prompt = "hello " * input_len

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=output_len,
        ignore_eos=True,
    )

    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    
    print(f"{'Batch Size':<12} | {'Total Time (s)':<16} | {'Req/s':<12} | {'Output Tok/s':<16} | {'Total Tok/s':<16}")
    print("-" * 85)

    results = []

    for bs in batch_sizes:
        prompts = [dummy_prompt for _ in range(bs)]
        
        # Warm-up / First compile
        if bs == 1:
            print("[Warmup] Triggering initial JIT graph compilation...")
            _ = llm.generate(["warmup prompt"], sampling_params)
            print("[Warmup] JIT warmup complete.\n")

        # Timed run
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        t_elapsed = time.perf_counter() - t0

        total_gen_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        total_prompt_tokens = sum(len(out.prompt_token_ids) for out in outputs)
        total_tokens = total_gen_tokens + total_prompt_tokens

        req_per_sec = bs / t_elapsed
        out_tokens_per_sec = total_gen_tokens / t_elapsed
        total_tokens_per_sec = total_tokens / t_elapsed

        print(f"{bs:<12} | {t_elapsed:<16.3f} | {req_per_sec:<12.2f} | {out_tokens_per_sec:<16.2f} | {total_tokens_per_sec:<16.2f}")
        results.append((bs, t_elapsed, req_per_sec, out_tokens_per_sec, total_tokens_per_sec))

    print("=" * 85)
    print("E2E Offline Benchmark Completed Successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=16)
    args = parser.parse_args()
    run_e2e_batch_benchmark(tp_size=args.tp_size)
