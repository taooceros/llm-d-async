"""Ragged paged attention for tpu-inference on NVIDIA GPUs.

Drop-in for ``tpu_inference.kernels.ragged_paged_attention.v3.kernel.
ragged_paged_attention`` (same arguments, same packed KV cache layout, same
``(output, kv_cache)`` result), so the rest of tpu-inference is unchanged.

* KV write: one XLA scatter of the new tokens into their pages (the TPU kernel
  fuses this into the Pallas kernel).
* Attention: the 2-D "unified attention" Triton kernel (prefill + decode in one
  launch, GQA heads packed into one tile, sliding window, soft cap, sinks).
  Source: ejKernel 0.0.80.3 ``kernels/_triton/unified_attention`` (Apache-2.0,
  (c) 2026 Erfan Zare Chavoshi), itself a JAX port of vLLM's
  ``triton_unified_attention``. Changes here: reads K and V from the
  tpu-inference cache layout (K of kv head h at combined head 2h, V at 2h+1),
  ALiBi / query-query bias removed, and the batch size is the static
  ``max_num_seqs`` with padded sequences sanitised to empty queries.
  Launched through jax-triton.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl


@triton.jit
def _find_seq_idx(query_start_len_ptr, target_idx, NUM_SEQS: tl.constexpr,
                  BLOCK_Q: tl.constexpr):
    # Block-index binary search: sequence i owns q blocks
    # [cu[i] // BLOCK_Q + i, cu[i + 1] // BLOCK_Q + i + 1).
    left: tl.int32 = 0
    right: tl.int32 = NUM_SEQS
    while left < right:
        mid = (left + right) // 2
        mid_val = tl.load(query_start_len_ptr + mid) // BLOCK_Q + mid
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _rpa_unified_2d(
    query_ptr,
    kv_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    out_ptr,
    scale: tl.constexpr,
    softcap: tl.constexpr,
    PAGES_PER_SEQ: tl.constexpr,
    NUM_SEQS: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    STRIDE_PAGE: tl.constexpr,  # elements between pages of the KV cache
    STRIDE_SLOT: tl.constexpr,  # elements between tokens of one page
    STRIDE_KV_HEAD: tl.constexpr,  # elements between kv heads (K/V pairs)
    V_OFFSET: tl.constexpr,  # elements from K to V of the same kv head
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = _find_seq_idx(query_start_len_ptr, q_block_global_idx, NUM_SEQS,
                            BLOCK_Q)
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (query_offset_0[:, None] * (num_query_heads * HEAD_SIZE) +
                    query_offset_1[:, None] * HEAD_SIZE + offs_d[None, :])

    dim_mask = (offs_d < HEAD_SIZE).to(tl.int1)
    query_mask_0 = (query_pos < cur_batch_query_len).to(tl.int1)
    query_mask_1 = (query_offset_1 < num_query_heads).to(tl.int1)

    q = tl.load(query_ptr + query_offset,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
                other=0.0)
    if q.dtype != kv_cache_ptr.dtype.element_ty:
        q = q.to(kv_cache_ptr.dtype.element_ty)

    block_table_offset = seq_idx * PAGES_PER_SEQ

    if USE_SINKS:
        m_prev = tl.load(sink_ptr + query_offset_1, mask=query_mask_1,
                         other=float("-inf")).to(tl.float32)
    else:
        m_prev = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_prev = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx).to(tl.int32)
    context_len = seq_len - cur_batch_query_len

    max_seq_prefix_len = (context_len + q_block_local_idx * BLOCK_Q +
                          (BLOCK_M - 1) // num_queries_per_kv + 1).to(tl.int32)
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = (max_seq_prefix_len + TILE_SIZE - 1) // TILE_SIZE

    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
                             cur_batch_query_len - 1)
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    k_head_base = kv_cache_ptr + kv_head_idx * STRIDE_KV_HEAD
    v_head_base = k_head_base + V_OFFSET
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        physical_page = tl.load(block_tables_ptr + block_table_offset +
                                seq_offset // PAGE_SIZE,
                                mask=tile_mask, other=0).to(tl.int64)
        slot = (seq_offset % PAGE_SIZE).to(tl.int64)
        k_offset = (physical_page[None, :] * STRIDE_PAGE +
                    slot[None, :] * STRIDE_SLOT + offs_d[:, None])
        v_offset = (physical_page[:, None] * STRIDE_PAGE +
                    slot[:, None] * STRIDE_SLOT + offs_d[None, :])
        k = tl.load(k_head_base + k_offset,
                    mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
        v = tl.load(v_head_base + v_offset,
                    mask=dim_mask[None, :] & tile_mask[:, None], other=0.0)

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1
        scores = scale * tl.dot(q, k)
        if USE_SOFTCAP:
            s = scores / softcap
            p1 = tl.exp(s)
            p2 = tl.exp(-s)
            scores = softcap * (p1 - p2) / (p1 + p2)
        scores = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
                          scores, float("-inf"))
        if SLIDING_WINDOW > 0:
            scores = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                scores, float("-inf"))

        m_curr = tl.maximum(m_prev, tl.max(scores, axis=1))
        m_curr = tl.where(m_curr > float("-inf"), m_curr, 0.0)
        p = tl.exp(scores - m_curr[:, None])
        l_curr = tl.sum(p, axis=1)
        alpha = tl.exp(m_prev - m_curr)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_prev = l_prev * alpha + l_curr
        m_prev = m_curr

    l_prev = tl.maximum(l_prev, 1e-6)
    acc = acc / l_prev[:, None]
    output_offset = (query_offset_0[:, None] * (num_query_heads * HEAD_SIZE) +
                     query_offset_1[:, None] * HEAD_SIZE + offs_d[None, :])
    tl.store(out_ptr + output_offset, acc,
             mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None])


def _sanitize(kv_lens, cu_q_lens, distribution):
    """Padded sequences (index >= num_seqs) become empty: cu_q_lens repeats the
    total and kv_lens is 0, so both the scatter and the kernel skip them."""
    num_seqs = distribution[2]
    max_num_seqs = kv_lens.shape[0]
    seq_ids = jnp.arange(max_num_seqs + 1, dtype=jnp.int32)
    total = cu_q_lens[num_seqs]
    cu = jnp.where(seq_ids <= num_seqs, cu_q_lens, total).astype(jnp.int32)
    lens = jnp.where(seq_ids[:-1] < num_seqs, kv_lens, 0).astype(jnp.int32)
    return lens, cu, total


def _write_kv(k, v, kv_cache, kv_lens, page_indices, cu, total):
    """Scatter the new tokens' K/V into their pages (TPU layout: K of kv head h
    at combined head 2h, V at 2h+1, head_dim padded to the cache's)."""
    num_tokens, num_kv_heads, head_dim = k.shape
    num_pages, page_size, heads_per_pack, pack, head_dim_padded = kv_cache.shape
    combined = heads_per_pack * pack
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    merged = jnp.concatenate([k, v], axis=-1).reshape(num_tokens, 2 * num_kv_heads, head_dim)
    merged = jnp.pad(merged, ((0, 0), (0, combined - 2 * num_kv_heads),
                              (0, head_dim_padded - head_dim))).astype(kv_cache.dtype)

    t = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.clip(jnp.searchsorted(cu, t, side="right") - 1, 0, max_num_seqs - 1)
    q_len = cu[seq + 1] - cu[seq]
    pos = kv_lens[seq] - q_len + (t - cu[seq])
    page_slot = jnp.clip(seq * pages_per_seq + pos // page_size, 0,
                         page_indices.shape[0] - 1)
    page = jnp.where(t < total, page_indices[page_slot], num_pages)  # OOB -> dropped
    cache = kv_cache.reshape(num_pages, page_size, combined, head_dim_padded)
    cache = cache.at[page, pos % page_size].set(merged, mode="drop")
    return cache.reshape(kv_cache.shape)


def ragged_paged_attention(
    queries: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
    keys: jax.Array,  # [max_num_tokens, num_kv_heads, head_dim]
    values: jax.Array,  # [max_num_tokens, num_kv_heads, head_dim]
    kv_cache: jax.Array,  # [num_pages, page_size, 2*num_kv_heads // pack, pack, head_dim_padded]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]; [2] = num_seqs
    attention_sink: jax.Array | None = None,
    *,
    use_causal_mask: bool = True,
    update_kv_cache: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype=None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    **_tpu_tuning,  # block sizes / vmem limits of the TPU kernel: not used here
):
    if not use_causal_mask:
        raise NotImplementedError("GPU RPA: bidirectional attention")
    if q_scale is not None:
        raise NotImplementedError("GPU RPA: quantized queries")
    num_tokens, num_q_heads, head_dim = queries.shape
    num_kv_heads = keys.shape[1]
    num_pages, page_size, heads_per_pack, pack, head_dim_padded = kv_cache.shape
    combined = heads_per_pack * pack
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    kv_lens, cu, total = _sanitize(kv_lens, cu_q_lens, distribution)
    if update_kv_cache:
        kv_cache = _write_kv(keys, values, kv_cache, kv_lens, page_indices, cu, total)

    q = queries
    if q.dtype != kv_cache.dtype:
        q = q.astype(kv_cache.dtype)
    per_kv = num_q_heads // num_kv_heads
    block_m = 16 if per_kv <= 16 else triton.next_power_of_2(per_kv)
    block_q = block_m // per_kv
    head_padded = max(16, triton.next_power_of_2(head_dim))
    tile = int(os.environ.get("HETERO_RPA_TILE", "32"))
    num_stages = int(os.environ.get("HETERO_RPA_STAGES", "2" if head_padded >= 512 else "3"))
    sinks = (attention_sink.astype(jnp.float32) if attention_sink is not None
             else jnp.zeros((1,), jnp.float32))
    scale = float(sm_scale) * (float(k_scale) if k_scale is not None else 1.0)

    out = jt.triton_call(
        q,
        kv_cache,
        sinks,
        page_indices,
        kv_lens,
        cu,
        kernel=_rpa_unified_2d,
        out_type=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(num_tokens // block_q + max_num_seqs, num_kv_heads),
        zeroed_outputs=(0,),
        num_warps=4,
        num_stages=num_stages,
        name="hetero_rpa_unified_2d",
        scale=scale,
        softcap=float(soft_cap or 0.0),
        PAGES_PER_SEQ=pages_per_seq,
        NUM_SEQS=max_num_seqs,
        num_query_heads=num_q_heads,
        num_queries_per_kv=per_kv,
        PAGE_SIZE=page_size,
        TILE_SIZE=tile,
        HEAD_SIZE=head_dim,
        HEAD_SIZE_PADDED=head_padded,
        STRIDE_PAGE=page_size * combined * head_dim_padded,
        STRIDE_SLOT=combined * head_dim_padded,
        STRIDE_KV_HEAD=2 * head_dim_padded,
        V_OFFSET=head_dim_padded,
        USE_SOFTCAP=bool(soft_cap),
        USE_SINKS=attention_sink is not None,
        SLIDING_WINDOW=int(sliding_window or 0),
        BLOCK_Q=block_q,
        BLOCK_M=block_m,
    )
    if v_scale is not None:
        out = (out.astype(jnp.float32) * v_scale).astype(q.dtype)
    if out_dtype is not None:
        out = out.astype(out_dtype)
    return out, kv_cache
