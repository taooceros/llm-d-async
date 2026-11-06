"""Trace loading for the autoscale cost model (TPUFlex cost-efficiency case study).

Sources (all local, see README):
  gemma   Chutes "A Year in LLM Serving" (arXiv 2608.13573, CC BY 4.0) extract of unsloth/gemma-3-27b-it,
          days 300.. (~/.cache/pm-render/traces/chutes_gemma3_27b_d300plus.parquet; s = seconds from trace start)
  qwen    Qwen-Bailian to-C trace A (~/.cache/pm-render/traces/qwen_traceA_blksz_16.jsonl), non-bursty control
Rows without token counts (failed/aborted on Chutes) are dropped, exactly as the real trace-A runs did
(scripts/asb_trace.py), and lengths are fit into max_model_len 8192 the same way (output capped at L-1, input
truncated to L - output).
Arrays are cached as .npz under /tmp/cm (large intermediates stay off the repo).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

TRACES = Path(os.path.expanduser("~/.cache/pm-render/traces"))
GEMMA = TRACES / "chutes_gemma3_27b_d300plus.parquet"
QWEN = TRACES / "qwen_traceA_blksz_16.jsonl"
CACHE = Path("/tmp/cm")
MAX_MODEL_LEN = 8192


def fit_lengths(it: np.ndarray, ot: np.ndarray, L: int = MAX_MODEL_LEN):
    out = ot.astype(np.int64)
    out = np.where(out >= L, L - 1, out)
    inp = np.maximum(it.astype(np.int64), 1)
    inp = np.maximum(np.minimum(inp, L - out), 1)
    return inp.astype(np.int32), out.astype(np.int32)


def load_gemma(with_instances: bool = False) -> dict:
    """Valid gemma-3-27b requests sorted by s: {'s', 'in', 'out'} (+ 'inst' codes)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "gemma.npz"
    if not f.exists():
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        tbl = pq.read_table(GEMMA, columns=["s", "it", "ot", "instance_id"])
        s = tbl["s"].to_numpy(zero_copy_only=False).astype(np.float64)
        it = tbl["it"].to_numpy(zero_copy_only=False).astype(np.float64)
        ot = tbl["ot"].to_numpy(zero_copy_only=False).astype(np.float64)
        inst = pc.dictionary_encode(tbl["instance_id"]).combine_chunks()
        codes = inst.indices.to_numpy(zero_copy_only=False).astype(np.int32)
        valid_all = ~np.isnan(it) & ~np.isnan(ot) & (np.nan_to_num(ot) >= 1)
        order = np.argsort(s, kind="stable")
        s, it, ot, codes, valid_all = s[order], it[order], ot[order], codes[order], valid_all[order]
        inp, out = fit_lengths(np.nan_to_num(it[valid_all], nan=1), ot[valid_all])
        np.savez(f, s=s[valid_all], inp=inp, out=out, inst_valid=codes[valid_all],
                 s_all=s, inst_all=codes, valid_all=valid_all)
    z = np.load(f)
    d = {"s": z["s"], "in": z["inp"], "out": z["out"]}
    if with_instances:
        d.update(inst_all=z["inst_all"], s_all=z["s_all"], valid_all=z["valid_all"])
    return d


def load_qwen() -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "qwen.npz"
    if not f.exists():
        ts, it, ot = [], [], []
        with open(QWEN) as fh:
            for line in fh:
                r = json.loads(line)
                ts.append(r["timestamp"]); it.append(r["input_length"]); ot.append(r["output_length"])
        ts, it, ot = np.array(ts, float), np.array(it, float), np.array(ot, float)
        ok = ot >= 1
        order = np.argsort(ts[ok], kind="stable")
        inp, out = fit_lengths(it[ok][order], ot[ok][order])
        np.savez(f, s=ts[ok][order], inp=inp, out=out)
    z = np.load(f)
    return {"s": z["s"], "in": z["inp"], "out": z["out"]}


def thin(d: dict, frac: float, seed: int = 20261025) -> dict:
    """Bernoulli thinning (keeps timestamps: burst shapes and durations stay those of the trace)."""
    if frac >= 1.0:
        return dict(d)
    rng = np.random.Generator(np.random.PCG64(seed))
    keep = rng.random(d["s"].size) < frac
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.shape == d["s"].shape else v) for k, v in d.items()}
