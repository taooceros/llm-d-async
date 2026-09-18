"""
Full Production vLLM End-to-End Offline Latency & Throughput Benchmark
Runs standard vLLM LLM.generate with Tokenizer, Backbone, RPA Attention, and Sampler.
"""
import os
import time
import json
import argparse
import numpy as np

MODEL_PATH = "/models/gemma-4-31b"


def run_vllm_benchmark(tp_size: int = 16, input_len: int = 512, output_len: int = 10, batch_sizes=None):
    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 32, 64, 128, 256]

    os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
    os.environ["TPU_CHIPS_PER_HOST"] = "4"
    os.environ["SCAN_TPU_CHIPS"] = "True"
    os.environ["TPU_ACCELERATOR_TYPE"] = "v5e"
    os.environ["TPU_TOPOLOGY"] = "4x4"
    os.environ["JAX_PLATFORMS"] = ""
    os.environ["JAX_COMPILATION_CACHE_DIR"] = "/models/jax_cache"

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    print("=" * 115, flush=True)
    print(f"  Production vLLM E2E Serving Benchmark: Gemma-4 31B (TP={tp_size})", flush=True)
    print(f"  Prompt Length : {input_len} tokens | Output Length: {output_len} tokens", flush=True)
    print(f"  Batch Sizes   : {batch_sizes}", flush=True)
    print("=" * 115, flush=True)

    print("\n[1/3] Initializing full vLLM Engine (TPU v5e-16)...", flush=True)
    t0_init = time.perf_counter()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tp_size,
        distributed_executor_backend="ray",
        max_model_len=4096,
        max_num_seqs=max(batch_sizes),
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=False,
    )
    print(f"[1/3] Engine ready in {time.perf_counter() - t0_init:.2f}s.\n", flush=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=output_len,
        ignore_eos=True,
        detokenize=False,
    )

    print(f"{'Batch Size':<12} | {'Total Time (s)':<16} | {'Gen Tok/s':<16} | {'Prefill Tok/s':<18} | {'TPOT (ms/tok)':<15}", flush=True)
    print("-" * 115, flush=True)

    results = []

    for bs in batch_sizes:
        # Generate tokenized prompt tokens
        token_ids = [list(range(100, 100 + input_len)) for _ in range(bs)]
        prompts = [TokensPrompt(prompt_token_ids=t) for t in token_ids]

        # Warmup
        if bs == batch_sizes[0]:
            print("[2/3] Performing warmup JIT passes...", flush=True)
            _ = llm.generate([TokensPrompt(prompt_token_ids=token_ids[0])], sampling_params, use_tqdm=False)
            print("[2/3] Warmup complete.\n", flush=True)

        iters = 5 if bs <= 64 else 3
        t_list = []
        for _ in range(iters):
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            t_list.append(time.perf_counter() - t0)

        avg_t = np.mean(t_list)
        total_gen_toks = bs * output_len
        total_prefill_toks = bs * input_len

        gen_tok_s = total_gen_toks / avg_t
        prefill_tok_s = total_prefill_toks / avg_t
        tpot_ms = (avg_t / output_len) * 1000.0

        print(f"{bs:<12} | {avg_t:<16.3f} | {gen_tok_s:<16.1f} | {prefill_tok_s:<18.1f} | {tpot_ms:<15.2f}", flush=True)
        results.append({
            "batch_size": bs,
            "total_time_s": float(avg_t),
            "gen_tok_s": float(gen_tok_s),
            "prefill_tok_s": float(prefill_tok_s),
            "tpot_ms": float(tpot_ms)
        })

    print("=" * 115, flush=True)
    print("FULL_VLLM_BENCHMARK_JSON=" + json.dumps(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=10)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32, 64, 128, 256])
    args = parser.parse_args()

    run_vllm_benchmark(
        tp_size=args.tp_size,
        input_len=args.input_len,
        output_len=args.output_len,
        batch_sizes=args.batch_sizes
    )
