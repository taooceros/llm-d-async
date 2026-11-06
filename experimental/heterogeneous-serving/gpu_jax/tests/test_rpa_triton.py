"""Triton RPA vs tpu-inference's ref_ragged_paged_attention, on a GPU.

    HETERO_JAX_GPU=1 PYTHONPATH=gpu_jax python -m pytest -q gpu_jax/tests/test_rpa_triton.py

The reference runs eagerly in float32 on an equivalent float32 cache (same values,
packing 1); the kernel runs jitted in bfloat16 on the packed bf16 cache, as in serving.
Checks: attention output of every real token, the whole KV cache after the write
(exact), zero output for padding tokens, and garbage in padded sequence slots.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hetero_jax_gpu.rpa_triton import ragged_paged_attention
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    get_kv_cache_shape, ref_ragged_paged_attention)

GEMMA4_SLIDING = dict(num_q_heads=32, num_kv_heads=16, head_dim=256)
GEMMA4_FULL = dict(num_q_heads=32, num_kv_heads=4, head_dim=512)

BATCHES = {  # (q_len, kv_len) per sequence
    "decode": [(1, 1), (1, 17), (1, 300), (1, 1025), (1, 2047), (1, 2900), (1, 64), (1, 513)],
    "prefill": [(700, 700), (300, 300), (37, 37)],
    "mixed": [(1, 900), (1, 2500), (1, 33), (200, 1500), (300, 300), (5, 1030)],
}


def build(seed, seqs, *, num_q_heads, num_kv_heads, head_dim, page_size,
          max_num_seqs, max_tokens, max_len, garbage):
    rng = np.random.default_rng(seed)
    num_seqs = len(seqs)
    q_lens = np.array([q for q, _ in seqs], np.int32)
    kv_lens = np.array([kv for _, kv in seqs], np.int32)
    total = int(q_lens.sum())
    assert total <= max_tokens and num_seqs <= max_num_seqs
    pages_per_seq = math.ceil(max_len / page_size)
    need = [math.ceil(kv / page_size) for kv in kv_lens]
    num_pages = sum(need) + 7
    perm = rng.permutation(num_pages).astype(np.int32)

    page_indices = np.zeros(max_num_seqs * pages_per_seq, np.int32)
    used = 0
    for i, n in enumerate(need):
        page_indices[i * pages_per_seq:i * pages_per_seq + n] = perm[used:used + n]
        used += n
    cu = np.zeros(max_num_seqs + 1, np.int32)
    cu[1:num_seqs + 1] = np.cumsum(q_lens)
    cu[num_seqs + 1:] = total
    lens = np.zeros(max_num_seqs, np.int32)
    lens[:num_seqs] = kv_lens
    if garbage:  # what padded slots may hold in a real step: anything (own RNG: same q/k/v)
        grng = np.random.default_rng(seed + 1000)
        cu[num_seqs + 1:] = grng.integers(0, max_tokens, max_num_seqs - num_seqs)
        lens[num_seqs:] = grng.integers(0, max_len, max_num_seqs - num_seqs)
        tail = page_indices.reshape(max_num_seqs, pages_per_seq)
        tail[num_seqs:] = grng.integers(0, num_pages, tail[num_seqs:].shape)

    def normal(*shape):
        return rng.standard_normal(shape, np.float32)

    q = normal(max_tokens, num_q_heads, head_dim)
    k = normal(max_tokens, num_kv_heads, head_dim)
    v = normal(max_tokens, num_kv_heads, head_dim)
    cache_shape = get_kv_cache_shape(num_pages, page_size, num_kv_heads, head_dim, jnp.bfloat16)
    cache = normal(*cache_shape)  # old context already in the pages
    dist = np.array([0, 0, num_seqs], np.int32)
    return dict(q=q, k=k, v=v, cache=cache, kv_lens=lens, page_indices=page_indices, cu=cu,
                dist=dist, total=total)


def run_case(case, *, sliding_window, soft_cap=None):
    bf = lambda a: jnp.asarray(a, jnp.bfloat16)
    i32 = lambda a: jnp.asarray(a, jnp.int32)
    q, k, v, cache = bf(case["q"]), bf(case["k"]), bf(case["v"]), bf(case["cache"])
    head_dim = q.shape[-1]
    sm_scale = head_dim**-0.5
    fn = jax.jit(functools.partial(ragged_paged_attention, sm_scale=sm_scale,
                                   sliding_window=sliding_window, soft_cap=soft_cap))
    out, new_cache = fn(q, k, v, cache, i32(case["kv_lens"]), i32(case["page_indices"]),
                        i32(case["cu"]), i32(case["dist"]))
    return np.asarray(out.astype(jnp.float32)), np.asarray(new_cache.astype(jnp.float32))


def run_ref(case, *, sliding_window, soft_cap=None):
    f32 = lambda a: jnp.asarray(jnp.asarray(a, jnp.bfloat16), jnp.float32)  # same bf16 values
    q, k, v = f32(case["q"]), f32(case["k"]), f32(case["v"])
    cache_bf = case["cache"].shape
    pages, page_size, heads_per_pack, pack, head_dim = cache_bf
    cache = f32(case["cache"]).reshape(pages, page_size, heads_per_pack * pack, 1, head_dim)
    out, new_cache = ref_ragged_paged_attention(
        q, k, v, cache, jnp.asarray(case["kv_lens"]), jnp.asarray(case["page_indices"]),
        jnp.asarray(case["cu"]), jnp.asarray(case["dist"]),
        sm_scale=q.shape[-1]**-0.5, sliding_window=sliding_window, soft_cap=soft_cap)
    return np.asarray(out), np.asarray(new_cache).reshape(cache_bf)


@pytest.mark.parametrize("shape", ["sliding", "full"])
@pytest.mark.parametrize("batch", list(BATCHES))
@pytest.mark.parametrize("page_size", [16, 256])
def test_matches_reference(shape, batch, page_size):
    dims = GEMMA4_SLIDING if shape == "sliding" else GEMMA4_FULL
    window = 1024 if shape == "sliding" else None
    case = build(0, BATCHES[batch], **dims, page_size=page_size, max_num_seqs=16,
                 max_tokens=1536, max_len=4096, garbage=False)
    out, cache = run_case(case, sliding_window=window)
    ref_out, ref_cache = run_ref(case, sliding_window=window)
    n = case["total"]
    err = np.abs(out[:n] - ref_out)
    assert err.max() < 3e-2 and err.mean() < 3e-3, (err.max(), err.mean())
    np.testing.assert_array_equal(cache, ref_cache)
    assert not np.any(out[n:])


@pytest.mark.parametrize("window", [None, 128])
def test_padded_sequences_and_small_window(window):
    seqs = BATCHES["mixed"][:5]
    clean = build(1, seqs, **GEMMA4_SLIDING, page_size=64, max_num_seqs=32, max_tokens=1024,
                  max_len=4096, garbage=False)
    dirty = build(1, seqs, **GEMMA4_SLIDING, page_size=64, max_num_seqs=32, max_tokens=1024,
                  max_len=4096, garbage=True)
    out, cache = run_case(dirty, sliding_window=window)
    ref_out, ref_cache = run_ref(clean, sliding_window=window)
    n = clean["total"]
    err = np.abs(out[:n] - ref_out)
    assert err.max() < 3e-2 and err.mean() < 3e-3, (err.max(), err.mean())
    np.testing.assert_array_equal(cache, ref_cache)


def test_soft_cap():
    case = build(2, BATCHES["mixed"], **GEMMA4_SLIDING, page_size=16, max_num_seqs=8,
                 max_tokens=1024, max_len=4096, garbage=False)
    out, _ = run_case(case, sliding_window=None, soft_cap=30.0)
    ref_out, _ = run_ref(case, sliding_window=None, soft_cap=30.0)
    err = np.abs(out[:case["total"]] - ref_out)
    assert err.max() < 3e-2 and err.mean() < 3e-3, (err.max(), err.mean())
