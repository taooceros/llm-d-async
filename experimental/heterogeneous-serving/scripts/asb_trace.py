#!/usr/bin/env python3
"""Trace builder for the AutoscaleBurst experiment (workstation side).

Subcommands:
  trace-a   cut + rescale a Chutes Gemma-3-27B burst window into a replay trace
  poisson   Poisson arrivals, piecewise-constant rate, lengths resampled from a trace
  sample    N requests (t = i) with lengths resampled from a trace (calibration)
  describe  print a trace's meta + quick stats

Trace JSON: {"meta": {...}, "requests": [{"i", "t", "in", "out", ...}]} sorted by t.
Deterministic: same args -> byte-identical output.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

DEFAULT_PARQUET = "~/.cache/pm-render/traces/chutes_gemma3_27b_d300plus.parquet"


def _pct(a, q):
    a = np.asarray(a, dtype=float)
    return float(np.percentile(a, q)) if a.size else None


def _len_stats(a):
    a = np.asarray(a, dtype=float)
    if not a.size:
        return {}
    return {"p50": _pct(a, 50), "p90": _pct(a, 90), "p99": _pct(a, 99),
            "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def _write(path, meta, reqs):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write('{"meta": ')
        json.dump(meta, f, sort_keys=True)
        f.write(', "requests": [\n')
        for j, r in enumerate(reqs):
            f.write(("," if j else "") + json.dumps(r, separators=(",", ":")) + "\n")
        f.write("]}\n")
    os.replace(tmp, path)


def _load(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- trace-a
def cmd_trace_a(a):
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    src = os.path.expanduser(a.parquet)
    m0 = a.burst_min + a.from_min
    m1 = a.burst_min + a.to_min  # exclusive
    b0 = a.burst_min + a.base_from_min
    b1 = a.burst_min + a.base_to_min  # exclusive
    if not (m0 <= b0 < b1 <= m1):
        sys.exit("base window must lie inside [from, to)")
    lo, hi = m0 * 60.0, m1 * 60.0
    dset = ds.dataset(src, format="parquet")
    tbl = dset.to_table(columns=["s", "it", "ot"],
                        filter=(pc.field("s") >= lo) & (pc.field("s") < hi))
    s = tbl["s"].to_numpy(zero_copy_only=False).astype(float)
    it = tbl["it"].to_numpy(zero_copy_only=False).astype(float)  # nulls -> nan
    ot = tbl["ot"].to_numpy(zero_copy_only=False).astype(float)
    order = np.argsort(s, kind="stable")
    s, it, ot = s[order], it[order], ot[order]
    valid = ~np.isnan(it) & ~np.isnan(ot) & (np.nan_to_num(ot, nan=0.0) >= 1)
    nmin = m1 - m0
    mins_all = np.floor(s / 60.0).astype(np.int64) - m0
    dropped_pm = np.bincount(mins_all[~valid], minlength=nmin)[:nmin]
    s, it, ot = s[valid], it[valid], ot[valid]
    mins = np.floor(s / 60.0).astype(np.int64) - m0
    counts_pm = np.bincount(mins, minlength=nmin)[:nmin]

    base_counts = counts_pm[b0 - m0:b1 - m0]
    base_rpm = float(base_counts.mean())
    if base_rpm <= 0:
        sys.exit("zero base rate")
    C = a.capacity_rps
    k = a.base_frac * C * 60.0 / base_rpm
    t = (s - lo) / k

    L = a.max_model_len
    out = ot.astype(np.int64)
    inp = np.maximum(it.astype(np.int64), 1)
    out_capped = out >= L
    out = np.where(out_capped, L - 1, out)
    inp_tr = np.minimum(inp, L - out)
    inp_tr = np.maximum(inp_tr, 1)
    # truncation accounting over total tokens (input and/or capped output), so an
    # output-capped row counts as truncated even when its input was already 1 token
    removed = (inp + ot.astype(np.int64)) - (inp_tr + out)
    truncated = removed > 0

    reqs = [{"i": int(j), "t": round(float(t[j]), 6), "in": int(inp_tr[j]), "out": int(out[j]),
             "src_s": round(float(s[j]), 3), "orig_in": int(inp[j]), "orig_out": int(ot[j])}
            for j in range(len(t))]

    peak_rpm = int(counts_pm.max())
    # burst-mean: mean rpm over trace minutes [burst_min, burst_min + burst_dur_min)
    bs = max(0, a.burst_min - m0)
    burst_counts = counts_pm[bs:bs + a.burst_dur_min]
    burst_mean_rpm = float(burst_counts.mean()) if burst_counts.size else None
    meta = {
        "kind": "trace-a",
        "source": src,
        "window_abs_min": [int(m0), int(m1)],
        "burst_min": a.burst_min,
        "rel_window_min": [a.from_min, a.to_min],
        "base_window_abs_min": [int(b0), int(b1)],
        "n": len(reqs),
        "dropped_invalid": int((~valid).sum()),
        "dropped_invalid_per_min": [int(x) for x in dropped_pm],
        "per_min_counts": [int(x) for x in counts_pm],
        "base_rpm_trace": base_rpm,
        "capacity_rps": C,
        "base_frac": a.base_frac,
        "target_base_rps": a.base_frac * C,
        "factor_k": k,
        "replay_duration_s": (hi - lo) / k,
        "last_t": float(t[-1]) if len(t) else 0.0,
        "peak_rpm_trace": peak_rpm,
        "peak_rps_replay": peak_rpm * k / 60.0,
        "peak_over_capacity": peak_rpm * k / 60.0 / C,
        "burst_dur_min": a.burst_dur_min,
        "burst_mean_rpm_trace": burst_mean_rpm,
        "burst_mean_rps_replay": burst_mean_rpm * k / 60.0 if burst_mean_rpm else None,
        "peak_over_base": peak_rpm / base_rpm,
        "burst_mean_over_base": burst_mean_rpm / base_rpm if burst_mean_rpm else None,
        "max_model_len": L,
        "truncated_n": int(truncated.sum()),
        "truncated_frac": float(truncated.mean()) if len(t) else 0.0,
        "truncated_mean_tokens_removed": float(removed[truncated].mean()) if truncated.any() else 0.0,
        "output_capped_n": int(out_capped.sum()),
        "in_stats": _len_stats(inp_tr),
        "out_stats": _len_stats(out),
        "seed": a.seed,
        "time_rule": "t = (s - window_start_s) / factor_k",
    }
    _write(a.out, meta, reqs)
    _print_meta(meta)


# ----------------------------------------------------------------------------- poisson
def _parse_schedule(spec, unit):
    segs = []
    for part in spec.split(","):
        d, r = part.split(":")
        segs.append((float(d), round(float(r) * unit, 9)))
    return segs


def _lengths(path):
    tr = _load(path)
    rq = tr["requests"]
    return (np.array([r["in"] for r in rq], dtype=np.int64),
            np.array([r["out"] for r in rq], dtype=np.int64))


def cmd_poisson(a):
    unit = a.rate_unit_rps if a.rate_unit_rps is not None else 1.0
    segs = _parse_schedule(a.schedule, unit)
    rng = np.random.Generator(np.random.PCG64(a.seed))
    ins, outs = _lengths(a.lengths_from)
    ts, seg_n = [], []
    t0 = 0.0
    for dur, rate in segs:
        n0 = len(ts)
        if rate > 0:
            t = t0
            while True:
                t += rng.exponential(1.0 / rate)
                if t >= t0 + dur:
                    break
                ts.append(t)
        seg_n.append(len(ts) - n0)
        t0 += dur
    idx = rng.integers(0, len(ins), size=len(ts))
    reqs = [{"i": j, "t": round(float(ts[j]), 6), "in": int(ins[idx[j]]), "out": int(outs[idx[j]])}
            for j in range(len(ts))]
    meta = {
        "kind": "poisson", "label": a.label, "seed": a.seed,
        "schedule_rps": [{"dur_s": d, "rps": r, "n": n} for (d, r), n in zip(segs, seg_n)],
        "rate_unit_rps": a.rate_unit_rps, "schedule_arg": a.schedule,
        "duration_s": t0, "n": len(reqs), "lengths_from": os.path.abspath(a.lengths_from),
        "in_stats": _len_stats([r["in"] for r in reqs]),
        "out_stats": _len_stats([r["out"] for r in reqs]),
    }
    _write(a.out, meta, reqs)
    _print_meta(meta)


def cmd_sample(a):
    rng = np.random.Generator(np.random.PCG64(a.seed))
    ins, outs = _lengths(a.lengths_from)
    idx = rng.integers(0, len(ins), size=a.n)
    reqs = [{"i": j, "t": float(j), "in": int(ins[idx[j]]), "out": int(outs[idx[j]])}
            for j in range(a.n)]
    meta = {"kind": "sample", "seed": a.seed, "n": a.n,
            "lengths_from": os.path.abspath(a.lengths_from),
            "in_stats": _len_stats(ins[idx]), "out_stats": _len_stats(outs[idx])}
    _write(a.out, meta, reqs)
    _print_meta(meta)


def _print_meta(meta):
    m = {k: v for k, v in meta.items() if k not in ("per_min_counts", "dropped_invalid_per_min")}
    print(json.dumps(m, indent=1, sort_keys=True))


def cmd_describe(a):
    tr = _load(a.trace)
    meta, rq = tr["meta"], tr["requests"]
    print(json.dumps(meta, indent=1, sort_keys=True))
    if not rq:
        print("no requests")
        return
    t = np.array([r["t"] for r in rq])
    ins = np.array([r["in"] for r in rq])
    outs = np.array([r["out"] for r in rq])
    dur = float(t[-1]) if len(t) else 0.0
    nb = max(1, int(math.ceil(dur / 60.0)) + (1 if dur % 60 == 0 else 0))
    per_min = np.bincount(np.floor(t / 60.0).astype(int), minlength=nb)
    print(json.dumps({
        "n": len(rq), "first_t": float(t[0]), "last_t": dur,
        "mean_rps": len(rq) / dur if dur > 0 else None,
        "sorted": bool(np.all(np.diff(t) >= 0)),
        "replay_per_min_counts": [int(x) for x in per_min],
        "peak_replay_rps_1min": float(per_min.max() / 60.0),
        "in": _len_stats(ins), "out": _len_stats(outs),
        "in_plus_out_max": int((ins + outs).max()),
        "total_out_tokens": int(outs.sum()), "total_in_tokens": int(ins.sum()),
    }, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    q = sp.add_parser("trace-a")
    q.add_argument("--parquet", default=DEFAULT_PARQUET)
    q.add_argument("--burst-min", type=int, default=464487)
    q.add_argument("--from-min", type=int, default=-8)
    q.add_argument("--to-min", type=int, default=44)
    q.add_argument("--base-from-min", type=int, default=-8)
    q.add_argument("--base-to-min", type=int, default=-1)
    q.add_argument("--capacity-rps", type=float, required=True)
    q.add_argument("--base-frac", type=float, default=0.5)
    q.add_argument("--burst-dur-min", type=int, default=38,
                   help="burst length (minutes) for the burst-mean rate (gemma3_bursts.csv dur_min)")
    q.add_argument("--max-model-len", type=int, default=8192)
    q.add_argument("--seed", type=int, default=20261025)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_trace_a)

    q = sp.add_parser("poisson")
    q.add_argument("--schedule", required=True, help="dur_s:rate,dur_s:rate,...")
    q.add_argument("--rate-unit-rps", type=float, default=None,
                   help="if set, schedule rates are multiples of this rps")
    q.add_argument("--lengths-from", required=True)
    q.add_argument("--seed", type=int, default=20261025)
    q.add_argument("--label", default="")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_poisson)

    q = sp.add_parser("sample")
    q.add_argument("--lengths-from", required=True)
    q.add_argument("--n", type=int, required=True)
    q.add_argument("--seed", type=int, default=20261025)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_sample)

    q = sp.add_parser("describe")
    q.add_argument("trace")
    q.set_defaults(fn=cmd_describe)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
