#!/usr/bin/env python3
"""
Benchmark: Truncation vs. Ceil Padding on Real vLLM TPU Attention Kernel
========================================================================
This microbenchmark compares the execution latency and effective token throughput
of vLLM's TPU Ragged Paged Attention (RPA) kernel under two shape-alignment strategies:

  Strategy A (Ceil Padding):
    Input tokens = 350 -> Pad to 512 bucket (execute 512 tokens with 162 padding).

  Strategy B (Smart Truncation):
    Input tokens = 350 -> Truncate to 256 bucket (execute 256 tokens in Step 1, 94 in Step 2).
"""

import time
import bisect
import numpy as np
from typing import List, Tuple

# Try importing TPU / JAX kernels
try:
    import jax
    import jax.numpy as jnp
    from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
        ragged_paged_attention,
        get_kv_cache_shape
    )
    from tpu_inference.kernels.ragged_paged_attention.v3.util import align_to, cdiv, get_dtype_packing
    HAS_TPU_RPA = True
except ImportError:
    HAS_TPU_RPA = False


def create_mock_rpa_inputs(
    num_tokens: int,
    num_seqs: int = 8,
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    num_pages: int = 1024,
    dtype=jnp.bfloat16 if HAS_TPU_RPA else None
):
    """Creates properly formatted input tensors for vLLM's ragged_paged_attention kernel."""
    if not HAS_TPU_RPA:
        return None

    # Distribute tokens across sequences
    tokens_per_seq = max(1, num_tokens // num_seqs)
    q_lens = [tokens_per_seq] * (num_seqs - 1)
    q_lens.append(num_tokens - sum(q_lens))
    kv_lens = [q + 128 for q in q_lens]  # Assuming some historical context

    cu_q_lens = np.pad(np.cumsum(q_lens, dtype=np.int32), (1, 0))
    kv_lens_arr = np.array(kv_lens, dtype=np.int32)
    
    # Generate page indices
    pages_per_seq = max(cdiv(kv, page_size) for kv in kv_lens)
    page_indices = np.zeros((num_seqs, pages_per_seq), dtype=np.int32)
    for i, kv in enumerate(kv_lens):
        needed = cdiv(kv, page_size)
        page_indices[i, :needed] = np.arange(i * 32, i * 32 + needed, dtype=np.int32)
    page_indices = page_indices.reshape(-1)

    # Queries, Keys, Values
    queries = jnp.zeros((num_tokens, num_q_heads, head_dim), dtype=dtype)
    keys = jnp.zeros((num_tokens, num_kv_heads, head_dim), dtype=dtype)
    values = jnp.zeros((num_tokens, num_kv_heads, head_dim), dtype=dtype)

    kv_cache_shape = get_kv_cache_shape(num_pages, page_size, num_kv_heads, head_dim, dtype)
    kv_cache = jnp.zeros(kv_cache_shape, dtype=dtype)

    distribution = np.array([num_seqs, 0, 0], dtype=np.int32)

    return {
        "queries": queries,
        "keys": keys,
        "values": values,
        "kv_cache": kv_cache,
        "kv_lens": jnp.array(kv_lens_arr),
        "page_indices": jnp.array(page_indices),
        "cu_q_lens": jnp.array(cu_q_lens),
        "distribution": jnp.array(distribution),
    }


def run_benchmark():
    print("=" * 70)
    print("  vLLM TPU Kernel Benchmark: Truncation vs. Ceil Padding")
    print("=" * 70)

    if not HAS_TPU_RPA:
        print("[!] TPU RPA kernel not available in CPU test environment.")
        print("[i] Simulating throughput model based on hardware roofline...\n")
        simulate_comparison()
        return

    # Benchmark settings
    warmup = 5
    iters = 20
    test_buckets = [64, 128, 256, 512, 1024, 2048]

    latencies = {}
    print(f"{'Bucket Size':<15} | {'Step Latency (ms)':<20} | {'Peak Token/s':<18}")
    print("-" * 60)

    for b in test_buckets:
        inputs = create_mock_rpa_inputs(num_tokens=b)
        
        # JIT warm-up
        for _ in range(warmup):
            out, _ = ragged_paged_attention(**inputs)
            jax.block_until_ready(out)

        # Timed runs
        t0 = time.perf_counter()
        for _ in range(iters):
            out, _ = ragged_paged_attention(**inputs)
            jax.block_until_ready(out)
        t1 = time.perf_counter()

        avg_ms = ((t1 - t0) / iters) * 1000.0
        tput = b / (avg_ms / 1000.0)
        latencies[b] = avg_ms
        print(f"{b:<15} | {avg_ms:<20.3f} | {tput:<18.1f}")

    print("\n" + "=" * 70)
    print("  Scenario Comparison: Input with 350 Tokens")
    print("=" * 70)
    
    # Strategy A: Ceil to 512
    t_512 = latencies[512]
    effective_tput_ceil = 350 / (t_512 / 1000.0)

    # Strategy B: Truncate to 256 + 94 (94 padded to 128)
    t_256 = latencies[256]
    t_128 = latencies[128]
    t_trunc_total = t_256 + t_128
    effective_tput_trunc = 350 / (t_trunc_total / 1000.0)

    print(f"Strategy A (Ceil Padding to 512):")
    print(f"  - Total Execution Time: {t_512:.3f} ms")
    print(f"  - Effective Throughput: {effective_tput_ceil:.1f} real_tokens/s")
    print(f"  - Padding Overhead    : 162 wasted tokens (31.6%)\n")

    print(f"Strategy B (Truncate to 256 + 128 in 2 Steps):")
    print(f"  - Total Execution Time: {t_trunc_total:.3f} ms ({t_256:.3f}ms + {t_128:.3f}ms)")
    print(f"  - Effective Throughput: {effective_tput_trunc:.1f} real_tokens/s")
    print(f"  - Padding Overhead    : 34 wasted tokens (8.8%)\n")

    if effective_tput_ceil > effective_tput_trunc:
        ratio = (effective_tput_ceil / effective_tput_trunc - 1) * 100
        print(f"[Result] Ceil Padding is {ratio:.1f}% FASTER due to single-step kernel dispatch overhead.")
    else:
        ratio = (effective_tput_trunc / effective_tput_ceil - 1) * 100
        print(f"[Result] Truncation is {ratio:.1f}% FASTER by avoiding padding compute.")


def simulate_comparison():
    """Mathematical comparison under typical TPU v5e kernel dispatch times."""
    print("Simulated Benchmark (Based on TPU v5e RPA Kernel Characteristics):")
    print("  - Base dispatch + HBM latency floor: ~0.40 ms")
    print("  - Compute slope: ~0.0015 ms / token\n")

    def est_latency(b):
        return 0.40 + 0.0015 * b

    lat_128 = est_latency(128) # ~0.59 ms
    lat_256 = est_latency(256) # ~0.78 ms
    lat_512 = est_latency(512) # ~1.17 ms

    # Input: 350 tokens
    # Strategy A: 512 bucket
    t_ceil = lat_512
    tput_ceil = 350 / (t_ceil / 1000.0)

    # Strategy B: 256 + 128 (two steps)
    t_trunc = lat_256 + lat_128
    tput_trunc = 350 / (t_trunc / 1000.0)

    print(f"Input: 350 Tokens")
    print(f"  Strategy A (Ceil -> 512):  Latency = {t_ceil:.2f} ms | Throughput = {tput_ceil:.1f} tok/s")
    print(f"  Strategy B (Trunc -> 256+128): Latency = {t_trunc:.2f} ms | Throughput = {tput_trunc:.1f} tok/s")
    print(f"\n[Key Insight] Because TPU forward pass has a fixed host-dispatch and weight-read overhead (~0.4ms),")
    print(f"running 1 step of 512 ({t_ceil:.2f}ms) is often faster than running 2 steps (256 + 128 = {t_trunc:.2f}ms),")
    print(f"UNLESS the leftover tokens can be naturally piggybacked onto subsequent pending requests.")


if __name__ == "__main__":
    run_benchmark()
