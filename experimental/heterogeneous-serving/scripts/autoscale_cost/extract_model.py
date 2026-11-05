#!/usr/bin/env python3
"""Extract one model's requests for days [d0, d1) from the public Chutes trace over HTTPS (DuckDB httpfs range
reads; the 91 GB file is never downloaded). Output: /tmp/cm/<tag>.npz with s (seconds from trace start), inp/out
(fit into max_model_len 8192 as in data.fit_lengths), inst_all (instance codes), valid mask.

    /tmp/cm-venv/bin/python scripts/autoscale_cost/extract_model.py --chute e3653034-9f58-5cf6-84f8-d5555e55fbd6 \
        --tag mistral_small_24b --d0 300 --d1 367
Source: Chutes "A Year in LLM Serving" (arXiv 2608.13573, CC BY 4.0); model names from
huggingface.co/datasets/harvardMadsys/chutes_metadata (data/chute_models.parquet).
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as dm  # noqa: E402

URL = "https://harvardsys-datasets.s3.us-east-1.amazonaws.com/2026_chutes_anonymized/chutes_trace.parquet"


def one_day(chute: str, d: int):
    import duckdb
    c = duckdb.connect()
    c.sql("load httpfs")
    q = f"""select epoch(started_at) as s, it, ot, instance_id from '{URL}'
            where started_at >= to_timestamp({d * 86400}) and started_at < to_timestamp({(d + 1) * 86400})
              and chute_id = '{chute}'"""
    for attempt in range(4):
        try:
            t0 = time.time()
            df = c.sql(q).fetchnumpy()
            return d, df, time.time() - t0
        except Exception as e:  # noqa: BLE001
            print(f"day {d} attempt {attempt}: {e}", flush=True)
            time.sleep(5)
    raise RuntimeError(f"day {d} failed")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chute", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--d0", type=int, default=300)
    ap.add_argument("--d1", type=int, default=367)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args(argv)
    parts = {}
    with ThreadPoolExecutor(a.threads) as ex:
        for d, df, dt in ex.map(lambda d: one_day(a.chute, d), range(a.d0, a.d1)):
            parts[d] = df
            print(f"day {d}: {len(df['s'])} rows in {dt:.1f}s", flush=True)
    s = np.concatenate([np.asarray(parts[d]["s"], float) for d in sorted(parts)])
    it = np.concatenate([np.asarray(parts[d]["it"], float) for d in sorted(parts)])
    ot = np.concatenate([np.asarray(parts[d]["ot"], float) for d in sorted(parts)])
    inst = np.concatenate([np.asarray(parts[d]["instance_id"]).astype(str) for d in sorted(parts)])
    order = np.argsort(s, kind="stable")
    s, it, ot, inst = s[order], it[order], ot[order], inst[order]
    _, codes = np.unique(inst, return_inverse=True)
    valid = ~np.isnan(it) & ~np.isnan(ot) & (np.nan_to_num(ot) >= 1)
    inp, out = dm.fit_lengths(np.nan_to_num(it[valid], nan=1), ot[valid])
    dm.CACHE.mkdir(parents=True, exist_ok=True)
    f = dm.CACHE / f"{a.tag}.npz"
    np.savez(f, s=s[valid], inp=inp, out=out, s_all=s, inst_all=codes.astype(np.int32), valid_all=valid)
    print(f"wrote {f}: {valid.sum()} valid of {len(s)}")


if __name__ == "__main__":
    main()
