#!/usr/bin/env python3
"""Azure LLM inference trace -> token-arrival envelope replay (autoscaling theory counterfactual).

Dependencies: Python >= 3.9 and numpy only (scipy/pandas are NOT used; scipy is broken in this env).

This is NOT an end-to-end serving simulation.  Each request's ContextTokens + GeneratedTokens is
attributed entirely to its arrival bin; per-bin average token rate is divided by an illustrative
per-instance capacity (default 24394 tokens/s) to get a work-proxy "instance demand".  We then
compute the exact offline (clairvoyant) rolling-future-max *reserved* footprint for boot time B,
which already contains booting + bridged-idle time; no extra boot term must be added.

Reproduce (see also trace_report.md):
    python3 /tmp/autoscale_theory/replay.py --download      # downloads if missing
    python3 /tmp/autoscale_theory/replay.py                 # full trace, 10 s and 60 s bins
Outputs: /tmp/autoscale_theory/trace_results.json, /tmp/autoscale_theory/trace_report.md
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone
from fractions import Fraction

import numpy as np

OUT_DIR = "/tmp/autoscale_theory"
DATA_DIR = os.path.join(OUT_DIR, "data")
BASE = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/"
DOC_URL = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzureLLMInferenceDataset2024.md"
TRACE_FILES = {
    "conv": "AzureLLMInferenceTrace_conv_1week.csv",
    "code": "AzureLLMInferenceTrace_code_1week.csv",
}
# Recorded when the file was downloaded for this task (conv variant only; code variant not recorded).
RECORDED = {
    "conv": {
        "sha256": "a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941",
        "bytes": 1135195393,
        "lines_incl_header": 27304000,
    }
}
DEFAULT_CAPACITY = 24394  # tokens/s per instance; illustrative work proxy, NOT an SLO claim
DEFAULT_CHIPS = 16  # chips per instance (TP16)
# Required measured boot times, plus a round-number sensitivity sweep.
ASSIGNMENT_BOOTS = [18.8, 27.5, 50.2, 284.9, 350.0, 385.0, 419.0, 779.4]
SWEEP_BOOTS = [1, 5, 10, 30, 60, 120, 300, 600, 900, 1800, 3600]
CHECK_BOOTS = ASSIGNMENT_BOOTS  # brute-force cross-check on real slices
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
EXPECTED_HEADER = b"TIMESTAMP,ContextTokens,GeneratedTokens"


# --------------------------------------------------------------------------------------
# Core envelope math (exact, integer/Fraction arithmetic)
# --------------------------------------------------------------------------------------
def sliding_forward_max_stream(x, k):
    """Streaming (monotonic deque, O(n)) y[i] = max(x[i..i+k]); beyond the end = zero pad.
    Requires x >= 0 so that truncated windows equal zero-padded windows."""
    n = len(x)
    out = [0] * n
    dq = deque()  # indices; values strictly decreasing from front to back
    for j in range(n - 1, -1, -1):
        v = x[j]
        while dq and x[dq[-1]] <= v:
            dq.pop()
        dq.append(j)
        if dq[0] > j + k:
            dq.popleft()
        out[j] = x[dq[0]]
    return out


def brute_forward_max(x, k):
    """Direct O(n*k) reference for sliding_forward_max_stream."""
    n = len(x)
    return [max(x[i:min(n, i + k + 1)]) for i in range(n)]


def split_boot(B, delta):
    """B = (m + phi) * delta with integer m >= 0 and exact rational phi in [0,1)."""
    bf = Fraction(str(B))
    m, rem = divmod(bf, Fraction(delta))
    return int(m), rem / Fraction(delta)


def envelope_sums(x, B, delta):
    """Exact integral of the rolling future-max reserved level for piecewise-constant demand.

    x[i] >= 0 is the (integer) demand level on bin i = [i*delta, (i+1)*delta); zero outside [0, N).
    Reserved level at time t = max of x over bins intersecting [t, t+B) (the bin containing t when
    B = 0).  With B = (m+phi)*delta and t = i*delta + s:
        reserved = M_i^m            for s <= (1-phi)*delta
        reserved = M_i^(m+1)        for s >  (1-phi)*delta       (M_i^k = max x[i..i+k])
    so the integral over bin i is delta*((1-phi)*M_i^m + phi*M_i^(m+1)).  Bins i<0 (pre-observation
    boot allocation, zero-padded demand) are included separately as `pre`.
    Returns Fractions in units of (level * bins): reserved integral = delta * (pre + win)."""
    m, phi = split_boot(B, delta)
    pad = m + 1  # bins i = -pad..-1 can already reserve for demand in bin 0 (only if phi>0 for -pad)
    xp = [0] * pad + [int(v) for v in x]
    sm = sliding_forward_max_stream(xp, m)
    pre_m, win_m = sum(sm[:pad]), sum(sm[pad:])
    if phi:
        sm1 = sliding_forward_max_stream(xp, m + 1)
        pre_1, win_1 = sum(sm1[:pad]), sum(sm1[pad:])
    else:
        pre_1, win_1 = pre_m, win_m
    return {
        "m": m,
        "phi": phi,
        "pre": pre_m + phi * (pre_1 - pre_m),
        "win": win_m + phi * (win_1 - win_m),
    }


def integer_replicas(counts, delta, cap):
    """ceil(counts / (delta * cap)) exactly, for a rational cap (Fraction)."""
    counts = np.asarray(counts, dtype=np.int64)
    num, den = cap.numerator, cap.denominator
    return -(-(counts * den) // (delta * num))


# --------------------------------------------------------------------------------------
# Brute-force validation
# --------------------------------------------------------------------------------------
def brute_reserved_ticks(x, q, p):
    """Independent tick-level reference.  Bin width q ticks, boot p ticks (integers).
    For every unit cell [k, k+1) use the midpoint t = k + 1/2 and take the max of x[j] over all bins
    j (zero outside [0,N)) whose interval overlaps [t, t+p) with positive measure.
    Returns (pre, win) as ints in level*ticks."""
    n = len(x)
    lo = -(p // q + 2) * q
    pre = win = 0
    for k in range(lo, n * q):
        tm2 = 2 * k + 1  # doubled midpoint
        jc = k // q
        best = 0
        for j in range(jc - 1, jc + p // q + 3):
            if 2 * j * q < tm2 + 2 * p and 2 * (j + 1) * q > tm2:
                if 0 <= j < n and x[j] > best:
                    best = x[j]
        if k < 0:
            pre += best
        else:
            win += best
    return pre, win


def layer_dilation_ticks(x, q, p):
    """Layer view: for each integer level l, busy set {x >= l}; reserve [a-p, b) per busy run [a,b),
    union overlapping/touching runs (gap <= p merges).  Returns (pre, win) in level*ticks."""
    pre = win = 0
    n = len(x)
    for lvl in range(1, (max(x) if x else 0) + 1):
        runs, j = [], 0
        while j < n:
            if x[j] >= lvl:
                a = j
                while j < n and x[j] >= lvl:
                    j += 1
                runs.append((a * q - p, j * q))
            else:
                j += 1
        merged = []
        for a, b in runs:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            if a < 0:
                pre += min(b, 0) - a
            win += max(b, 0) - max(a, 0)
    return pre, win


def run_validation(seed=20241020, trials=400):
    """Synthetic validation: streaming vs brute window max; closed-form fractional-B integral vs
    tick-level brute force; layer-dilation (merge gaps <= B) identity."""
    parser_rows = [
        "2024-05-12 00:00:00+00:00,123,45",
        "2024-05-12 00:00:00.1+00:00,0,9",
        "2024-05-12 00:00:00.12+00:00,10,0",
        "2024-05-18 23:59:59.995+00:00,50000,1000",
        "2024-02-29T12:34:56.123456+00:00,99999,987",
    ]
    parser_body = ("\r\n".join(parser_rows) + "\r\n").encode("ascii")
    fast, slow = parse_chunk_fast(parser_body), parse_chunk_slow(parser_body)
    assert all(np.array_equal(a, b) for a, b in zip(fast, slow))
    rng = random.Random(seed)
    n_stream = n_frac = n_layer = 0
    for _ in range(trials):
        n = rng.randint(1, 40)
        x = [0 if rng.random() < 0.35 else rng.randint(0, 9) for _ in range(n)]
        for k in range(0, 13):
            assert sliding_forward_max_stream(x, k) == brute_forward_max(x, k), (x, k)
            n_stream += 1
    for _ in range(trials):
        q = rng.randint(1, 7)
        n = rng.randint(1, 14)
        p = rng.choice([0, q, 2 * q, rng.randint(0, 5 * q + 3), rng.randint(0, 5 * q + 3)])
        x = [0 if rng.random() < 0.4 else rng.randint(0, 6) for _ in range(n)]
        env = envelope_sums(x, Fraction(p), q)  # delta = q "seconds", B = p "seconds" (ticks)
        bp, bw = brute_reserved_ticks(x, q, p)
        assert q * env["pre"] == bp and q * env["win"] == bw, (x, q, p, env, bp, bw)
        n_frac += 1
        lp, lw = layer_dilation_ticks(x, q, p)
        assert (lp, lw) == (bp, bw), ("layer", x, q, p, (lp, lw), (bp, bw))
        n_layer += 1
    return {
        "seed": seed,
        "fast_vs_stdlib_parser_rows": len(parser_rows),
        "streaming_vs_brute_window_max_cases": n_stream,
        "closed_form_vs_tick_bruteforce_cases": n_frac,
        "layer_dilation_vs_bruteforce_cases": n_layer,
        "passed": True,
    }


def validate_on_trace_slice(counts10, counts60, cap):
    """Brute-force cross-check on real slices around the peak bin (0.1 s tick resolution),
    for both the token-count series (continuous path) and ceil'd replica series (integer path)."""
    out = {"slices": [], "passed": True}
    for delta, counts, nbins in ((10, counts10, 40), (60, counts60, 40)):
        c = np.asarray(counts)
        pk = int(np.argmax(c))
        a = max(0, min(pk - nbins // 2, len(c) - nbins))
        sl_counts = [int(v) for v in c[a:a + nbins]]
        sl_int = [int(v) for v in integer_replicas(c[a:a + nbins], delta, cap)]
        q = delta * 10
        for name, series in (("tokens", sl_counts), ("replicas", sl_int)):
            for B in CHECK_BOOTS:
                p = Fraction(str(B)) * 10
                if p.denominator != 1:
                    continue
                p = int(p)
                env = envelope_sums(series, B, delta)
                bp, bw = brute_reserved_ticks(series, q, p)
                assert q * env["pre"] == bp and q * env["win"] == bw, (delta, name, B)
                for k in (env["m"], env["m"] + 1):
                    assert sliding_forward_max_stream(series, k) == brute_forward_max(series, k)
            out["slices"].append({"bin_s": delta, "series": name, "first_bin": a, "bins": nbins,
                                  "boots_checked": CHECK_BOOTS})
    return out


# --------------------------------------------------------------------------------------
# Trace parsing (vectorised fixed-schema parser with stdlib fallback)
# --------------------------------------------------------------------------------------
def _digits_to_int(arr, starts, ends, what):
    length = ends - starts
    if length.size == 0:
        return np.zeros(0, dtype=np.int64)
    if length.min() < 1 or length.max() > 15:
        raise ValueError("bad %s field length" % what)
    val = np.zeros(starts.shape, dtype=np.int64)
    bad = np.zeros(starts.shape, dtype=bool)
    last = arr.size - 1
    for k in range(int(length.max())):
        valid = k < length
        d = arr[np.minimum(starts + k, last)].astype(np.int64) - 48
        bad |= valid & ((d < 0) | (d > 9))
        val = np.where(valid, val * 10 + d, val)
    if bad.any():
        raise ValueError("non-digit in %s" % what)
    return val


def parse_chunk_fast(body):
    """body: whole CSV lines (LF or CRLF).  Returns (t_us int64 since epoch, ctx, gen)."""
    arr = np.frombuffer(body, dtype=np.uint8)
    nl = np.flatnonzero(arr == 10)
    n = nl.size
    starts = np.empty(n, dtype=np.int64)
    starts[0] = 0
    starts[1:] = nl[:-1] + 1
    commas = np.flatnonzero(arr == 44)
    if commas.size != 2 * n:
        raise ValueError("comma count")
    c1, c2 = commas[0::2], commas[1::2]
    if not ((c1 > starts).all() and (c2 < nl).all() and (c1 < c2).all()):
        raise ValueError("line structure")
    ln = c1 - starts
    nd = np.maximum(ln - 26, 0)
    if not ((ln == 25) | ((ln >= 27) & (ln <= 32))).all():
        raise ValueError("timestamp length")
    for off, ch in ((4, 45), (7, 45), (13, 58), (16, 58)):
        if not (arr[starts + off] == ch).all():
            raise ValueError("timestamp separators")
    if not ((arr[starts + 10] == 32) | (arr[starts + 10] == 84)).all():
        raise ValueError("timestamp date/time separator")
    if not (((arr[starts + 19] == 46) == (nd > 0)).all()):
        raise ValueError("fraction marker")
    for i, ch in enumerate(b"+00:00"):  # all timestamps must be UTC
        if not (arr[c1 - 6 + i] == ch).all():
            raise ValueError("non-UTC timestamp")

    def dg(off):
        d = arr[starts + off].astype(np.int64) - 48
        if ((d < 0) | (d > 9)).any():
            raise ValueError("timestamp digit")
        return d

    yr = dg(0) * 1000 + dg(1) * 100 + dg(2) * 10 + dg(3)
    mo = dg(5) * 10 + dg(6)
    dy = dg(8) * 10 + dg(9)
    hh = dg(11) * 10 + dg(12)
    mi = dg(14) * 10 + dg(15)
    ss = dg(17) * 10 + dg(18)
    frac = np.zeros(n, dtype=np.int64)
    for k in range(6):
        valid = k < nd
        d = arr[np.minimum(starts + 20 + k, arr.size - 1)].astype(np.int64) - 48
        if (valid & ((d < 0) | (d > 9))).any():
            raise ValueError("fraction digit")
        frac = frac * 10 + np.where(valid, d, 0)
    # days from civil (Hinnant)
    y = yr - (mo <= 2)
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (mo + np.where(mo > 2, -3, 9)) + 2) // 5 + dy - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    t_us = ((days * 86400 + hh * 3600 + mi * 60 + ss) * 1_000_000) + frac
    end3 = nl - (arr[nl - 1] == 13)
    ctx = _digits_to_int(arr, c1 + 1, c2, "ContextTokens")
    gen = _digits_to_int(arr, c2 + 1, end3, "GeneratedTokens")
    return t_us, ctx, gen


def parse_chunk_slow(body):
    t, c, g = [], [], []
    for line in body.decode("ascii").splitlines():
        line = line.strip()
        if not line:
            continue
        a, b, d = line.split(",")
        dt = datetime.fromisoformat(a)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = dt.astimezone(timezone.utc) - EPOCH
        t.append((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds)
        c.append(int(b))
        g.append(int(d))
    return (np.array(t, dtype=np.int64), np.array(c, dtype=np.int64), np.array(g, dtype=np.int64))


def load_trace(path, subset_hours=None, chunk_bytes=1 << 26):
    ts, cx, gn = [], [], []
    st = {"slow_chunks": 0, "cutoff": None, "stop": False}

    def ingest(body):
        try:
            t, c, g = parse_chunk_fast(body)
        except (ValueError, IndexError):
            st["slow_chunks"] += 1
            t, c, g = parse_chunk_slow(body)
        if t.size == 0:
            return
        if subset_hours is not None and st["cutoff"] is None:
            st["cutoff"] = (int(t.min()) // 60_000_000) * 60_000_000 + int(subset_hours * 3600 * 1_000_000)
        if st["cutoff"] is not None and int(t.min()) >= st["cutoff"]:
            st["stop"] = True
            return
        ts.append(t)
        cx.append(c)
        gn.append(g)

    with open(path, "rb") as f:
        header = f.readline().strip().lstrip(b"\xef\xbb\xbf")
        if header != EXPECTED_HEADER:
            raise SystemExit("unexpected header %r" % header)
        carry = b""
        while not st["stop"]:
            data = f.read(chunk_bytes)
            if not data:
                if carry.strip():
                    ingest(carry + b"\n")
                break
            buf = carry + data
            i = buf.rfind(b"\n")
            if i < 0:
                carry = buf
                continue
            body, carry = buf[: i + 1], buf[i + 1:]
            ingest(body)
    t = np.concatenate(ts)
    c = np.concatenate(cx)
    g = np.concatenate(gn)
    if st["cutoff"] is not None:
        keep = t < st["cutoff"]
        t, c, g = t[keep], c[keep], g[keep]
    return t, c, g, st


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iso_us(us):
    return (EPOCH + timedelta(microseconds=int(us))).isoformat()


def pctiles(a, qs=(50, 90, 99, 99.9)):
    v = np.percentile(a, qs)
    return {("p%g" % q): float(x) for q, x in zip(qs, v)}


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def bin_series(rel_us, tok, n10):
    b10 = rel_us // 10_000_000
    c10 = np.rint(np.bincount(b10, weights=tok.astype(np.float64), minlength=n10)).astype(np.int64)
    if int(c10.sum()) != int(tok.sum()):
        raise SystemExit("binning lost tokens")
    c60 = c10.reshape(-1, 6).sum(axis=1)
    return c10, c60


def demand_stats(counts, delta, cap):
    rate = counts.astype(np.float64) / delta
    pk = int(counts.max())
    return {
        "bin_s": delta,
        "bins": int(counts.size),
        "mean_tokens_per_s": float(rate.mean()),
        "std_tokens_per_s": float(rate.std()),
        "min_tokens_per_s": float(rate.min()),
        "peak_tokens_per_s": float(rate.max()),
        "peak_to_mean": float(rate.max() / rate.mean()) if rate.mean() > 0 else None,
        "percentiles_tokens_per_s": pctiles(rate),
        "empty_bins": int((counts == 0).sum()),
        "peak_instances_continuous": float(Fraction(pk, delta) / cap),
        "peak_instances_integer": int(-(-(pk * cap.denominator) // (delta * cap.numerator))),
        "mean_instances_continuous": float(rate.mean() / float(cap)),
    }


def evaluate(counts, delta, window_s, cap, chips, boots):
    counts = np.asarray(counts, dtype=np.int64)
    n_int = integer_replicas(counts, delta, cap)
    xc = [int(v) for v in counts]
    xi = [int(v) for v in n_int]
    total_tok = int(counts.sum())
    b0 = {"continuous": Fraction(total_tok) / cap, "integer": Fraction(delta * int(n_int.sum()))}  # inst-s
    static = {
        "continuous": Fraction(int(counts.max()), delta) / cap * window_s,
        "integer": Fraction(int(n_int.max()) * window_s),
    }

    def hrs(v):
        return float(v) / 3600.0

    def pack(kind, total, win, pre):
        b0v, stv = b0[kind], static[kind]
        return {
            "instance_hours_incl_pre": hrs(total),
            "chip_hours_incl_pre": hrs(total) * chips,
            "instance_hours_window_only": hrs(win),
            "chip_hours_window_only": hrs(win) * chips,
            "pre_observation_instance_hours": hrs(pre),
            "savings_vs_static_peak_incl_pre": float(1 - total / stv),
            "savings_vs_static_peak_window_only": float(1 - win / stv),
            "ratio_vs_B0_incl_pre": float(total / b0v),
            "ratio_vs_B0_window_only": float(win / b0v),
            "boot_and_bridge_overhead_instance_hours_incl_pre": hrs(total - b0v),
            "boot_and_bridge_overhead_instance_hours_window_only": hrs(win - b0v),
        }

    rows = []
    prev_tot = {"continuous": Fraction(0), "integer": Fraction(0)}
    mono = True
    bounds = True
    for B in boots:
        ec = envelope_sums(xc, B, delta)
        ei = envelope_sums(xi, B, delta)
        cont_pre, cont_win = ec["pre"] / cap, ec["win"] / cap
        int_pre, int_win = delta * ei["pre"], delta * ei["win"]
        for kind, pre, win in (("continuous", cont_pre, cont_win), ("integer", int_pre, int_win)):
            mono &= (pre + win) >= prev_tot[kind]
            prev_tot[kind] = pre + win
            bounds &= (b0[kind] <= win <= static[kind])
        rows.append({
            "boot_s": B,
            "m_full_bins": ec["m"],
            "phi_fraction": float(ec["phi"]),
            "phi_exact": str(ec["phi"]),
            "continuous": pack("continuous", cont_pre + cont_win, cont_win, cont_pre),
            "integer": pack("integer", int_pre + int_win, int_win, int_pre),
        })
    return {
        "bin_s": delta,
        "baselines": {
            "B0": {k: {"instance_hours": hrs(v), "chip_hours": hrs(v) * chips} for k, v in b0.items()},
            "static_peak": {k: {"instance_hours": hrs(v), "chip_hours": hrs(v) * chips} for k, v in static.items()},
        },
        "boots": rows,
        "sanity": {"monotone_nondecreasing_in_B": bool(mono), "B0_le_window_le_static": bool(bounds)},
    }


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
METHODOLOGY = r"""
## What this is (and is not)

* A **token-arrival envelope** counterfactual. Each request contributes `ContextTokens + GeneratedTokens`
  to the bin containing its arrival timestamp. There is no latency, queueing, KV-memory, batching,
  prefill/decode split or SLO model in it. It is **not** an end-to-end serving result.
* `capacity` (default 24394 tokens/s per instance) is an **illustrative work proxy** only: a
  measured 16k prefill-ish throughput figure, not end-to-end capacity and not an SLO claim.
* **Chip-hours are a modeled reserved footprint**: `chips_per_instance (16) x instance-hours`.
* The reserved footprint under boot time `B` **already contains** the boot/start-up time and any
  bridged idle gaps (a replica that would idle for a gap `<= B` is cheaper to keep than to re-boot).
  **No separate boot term is added anywhere**; the "boot and bridge overhead" columns are
  `reserved - B0` (a decomposition *inside* the envelope), not an add-on.

## Binning and demand

* Timestamps are parsed to integer microseconds (UTC). The origin `t0` is the first observed
  timestamp floored to a 60 s boundary. The observation window is `[0, W)` with `W` = whole 60 s bins
  up to and including the last arrival (the same `W` for 10 s and 60 s bins; 60 s bins are exact sums of 10 s bins).
* `counts[i]` = total tokens (context + generated) of requests arriving in bin `i`. Demand is taken as
  constant at `counts[i]/delta` tokens/s on the bin (fluid / bin-average assumption).
* Continuous instances `q_i = counts[i] / (delta * cap)`; integer instances `n_i = ceil(q_i)`
  (computed exactly with integer arithmetic).

## Rolling future-max reserved footprint (exact, fractional B)

For boot time `B`, reserved level `R(t) = max demand over [t, t+B)` (the current bin if `B = 0`).
With `B = (m + phi) * delta`, `m` integer, `phi in [0,1)` (exact rational from the decimal string of `B`)
and `t = i*delta + s`, `0 <= s < delta`:

* `R(t) = M_i^m`      for `s <= (1-phi)*delta`
* `R(t) = M_i^(m+1)`  for `s >  (1-phi)*delta`, where `M_i^k = max(x[i..i+k])`.

Hence the integral over bin `i` is exactly `delta * ((1-phi) * M_i^m + phi * M_i^(m+1))`; no rounding of
`B` to a bin multiple. Both maxima are computed with a streaming monotonic-deque sliding maximum (O(n)).
For continuous replicas the integer token counts are maxed and divided by `cap` at the end, so all sums
are exact `Fraction`s until the final float conversion.

Interpretation: with identical, independently releasable instances, known future demand, zero drain /
minimum billing and immediate useful rate after boot, `R(t)` is the minimum total *allocated*
(booting + ready) level, i.e. the lower bound is attained by "layer interval dilation": for each integer
level, expand every busy interval `[a,b)` to `[a-B, b)` and merge intervals separated by gaps `<= B`.
`R(t) >= l  <=>  some s in [t, t+B) has demand >= l`, which is exactly that union. This identity is
brute-force checked in `run_validation`.

## Zero-padding / end convention

* Demand is zero before `t0` and after `t0 + W` (zero padding on both sides).
* **Pre-observation boot allocation**: reserved capacity in `[-B, 0)` needed to be ready for demand in
  bin 0 is included in the headline `*_incl_pre` figures (the clairvoyant plan must boot before `t0`).
  The `*_window_only` figures integrate `R(t)` only over `[0, W)`.
* At the end, windows simply shrink (zero pad): reserved capacity is released once no future demand exists.
* B = 0 baseline: `R = demand`, i.e. integrated work `sum(counts)/cap` (continuous) or
  `delta * sum(n_i)` (integer). Static peak: `peak(bin) * W` (continuous `peak/cap`, integer `ceil`),
  provisioned over the observation window only (no pre-observation term).
* Savings vs static = `1 - reserved/static`; ratio vs B0 = `reserved/B0`.

## Validation (executed by the script on every run, before the replay)

1. Streaming sliding maximum == brute-force `max(x[i:i+k+1])` on random arrays and window sizes.
2. Closed-form fractional-B integral == tick-level brute force (midpoint sampling of `R(t)` by explicit
   interval overlap, incl. pre-observation cells) on random small cases with `B` not a bin multiple.
3. Layer-dilation (merge gaps `<= B`) union length == the same brute force.
4. The same brute force against real trace slices around the peak bin (10 s and 60 s bins, tokens and
   integer replicas, all eight requested boot times at 0.1 s resolution).
5. Sanity on the full results: reserved is non-decreasing in `B`, and `B0 <= window-only <= static peak`.

## Caveats

* Token-arrival envelope only; arrival-time attribution of generated tokens (no decode spreading over
  service time), bin-average rates hide sub-bin bursts (10 s is finer than 60 s).
* Clairvoyant (perfect foresight) offline optimum, identical instances, zero drain, no minimum billing,
  no warm pools / partial-ready capacity, no failures, no fragmentation. This is a bound, not a policy.
* The trace is a sample of Azure production traffic (DynamoLLM, HPCA 2025); one variant and one week.
  The assignment text names a Gemma-4/TPU context that is not in the trace; no model-specific claim is made.
* `capacity` is a proxy, so continuous absolute chip-hours scale inversely with it; integer-replica
  results are not linear in it. Savings fractions in the continuous case are capacity independent.
"""


def fmt(v, nd=2):
    if v is None:
        return "n/a"
    return ("{:,.%df}" % nd).format(v)


def pct(v):
    return "%.2f%%" % (100 * v)


def build_report(res, results_available=True):
    L = []
    L.append("# Azure LLM inference trace: token-arrival envelope replay\n")
    L.append("Generated by `/tmp/autoscale_theory/replay.py`. %s\n" % (
        "Numbers below are produced by that run." if results_available else
        "**Results not yet generated: run replay.py; this file is overwritten with actual numbers.**"))
    L.append("""
## Provenance and reproduction

* Dataset doc: %s
* Trace URL (conv): %s
* Trace URL (code): %s
* License: CC-BY (see doc); cite Stojkovic et al., DynamoLLM, HPCA 2025.
* Dependencies: Python 3 + numpy only (no scipy/pandas). Install if missing: `python3 -m pip install numpy`.
* Commands:

```
mkdir -p /tmp/autoscale_theory/data
curl -L -o /tmp/autoscale_theory/data/AzureLLMInferenceTrace_conv_1week.csv \\
  %s
sha256sum /tmp/autoscale_theory/data/AzureLLMInferenceTrace_conv_1week.csv
python3 /tmp/autoscale_theory/replay.py                      # full trace, 10 s + 60 s bins
python3 /tmp/autoscale_theory/replay.py --subset-hours 24    # deterministic first-24h subset
python3 /tmp/autoscale_theory/replay.py --variant code --download
python3 /tmp/autoscale_theory/replay.py --validate-only
```
""" % (DOC_URL, BASE + TRACE_FILES["conv"], BASE + TRACE_FILES["code"], BASE + TRACE_FILES["conv"]))
    if results_available:
        meta, inp = res["metadata"], res["input"]
        L.append("## Input metadata and statistics\n")
        L.append("| item | value |\n|---|---|")
        rows = [
            ("file", inp["path"]), ("variant", inp["variant"]), ("bytes", "{:,}".format(inp["bytes"])),
            ("sha256", inp["sha256"]), ("sha256 matches recorded download", str(inp.get("sha256_matches_recorded"))),
            ("subset", inp["subset"]),
            ("rows replayed", "{:,}".format(inp["rows"])),
            ("first / last timestamp", "%s / %s" % (inp["first_timestamp"], inp["last_timestamp"])),
            ("origin t0 (60 s floor)", inp["t0"]),
            ("window W (s)", "{:,}".format(inp["window_s"])),
            ("rows / non-monotonic steps / duplicate timestamps",
             "%s / %s / %s" % ("{:,}".format(inp["rows"]), "{:,}".format(inp["non_monotonic_steps"]),
                               "{:,}".format(inp["duplicate_timestamps"]))),
            ("rows with whole-second timestamps (no fractional part)", "{:,}".format(inp["whole_second_timestamps"])),
            ("context tokens total / mean / max", "%s / %s / %s" % (
                "{:,}".format(inp["context"]["sum"]), fmt(inp["context"]["mean"]), "{:,}".format(inp["context"]["max"]))),
            ("generated tokens total / mean / max", "%s / %s / %s" % (
                "{:,}".format(inp["generated"]["sum"]), fmt(inp["generated"]["mean"]), "{:,}".format(inp["generated"]["max"]))),
            ("context percentiles", ", ".join("%s=%s" % (k, fmt(v, 1)) for k, v in inp["context"]["percentiles"].items())),
            ("generated percentiles", ", ".join("%s=%s" % (k, fmt(v, 1)) for k, v in inp["generated"]["percentiles"].items())),
            ("rows with 0 context / 0 generated", "%s / %s" % (inp["context"]["zero_rows"], inp["generated"]["zero_rows"])),
            ("generated share of total tokens", pct(inp["generated_token_share"])),
            ("capacity (tokens/s per instance, proxy)", str(meta["capacity_tokens_per_s"])),
            ("chips per instance", str(meta["chips_per_instance"])),
            ("python / numpy", "%s / %s" % (meta["python"], meta["numpy"])),
            ("parse seconds / chunks needing slow parser", "%s / %s" % (meta["parse_seconds"], meta["slow_parse_chunks"])),
        ]
        for k, v in rows:
            L.append("| %s | %s |" % (k, v))
        L.append("\n### Per-bin demand statistics\n")
        L.append("| bin | bins | mean tok/s | peak tok/s | peak/mean | p50 | p90 | p99 | p99.9 | empty bins | peak instances (cont / int) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for ds in res["demand_stats"]:
            pc = ds["percentiles_tokens_per_s"]
            L.append("| %ds | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %d |" % (
                ds["bin_s"], "{:,}".format(ds["bins"]), fmt(ds["mean_tokens_per_s"]), fmt(ds["peak_tokens_per_s"]),
                fmt(ds["peak_to_mean"]), fmt(pc["p50"]), fmt(pc["p90"]), fmt(pc["p99"]), fmt(pc["p99.9"]),
                ds["empty_bins"], fmt(ds["peak_instances_continuous"]), ds["peak_instances_integer"]))
        L.append("\n## Results (modeled reserved footprint, chip-hours)\n")
        L.append("Headline = including pre-observation boot allocation `[-B,0)`. 'x B0' = reserved / B=0 baseline. "
                 "Savings = 1 - reserved/static peak.\n")
        for ev in res["evaluations"]:
            bl = ev["baselines"]
            L.append("### %d s bins\n" % ev["bin_s"])
            L.append("Baselines (chip-hours): B=0 continuous %s, integer %s; static peak continuous %s, integer %s.\n" % (
                fmt(bl["B0"]["continuous"]["chip_hours"]), fmt(bl["B0"]["integer"]["chip_hours"]),
                fmt(bl["static_peak"]["continuous"]["chip_hours"]), fmt(bl["static_peak"]["integer"]["chip_hours"])))
            L.append("| B (s) | m | phi | cont chip-h | cont vs static | cont x B0 | int chip-h | int vs static | int x B0 |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for r in ev["boots"]:
                c, i = r["continuous"], r["integer"]
                L.append("| %g | %d | %.4f | %s | %s | %.4f | %s | %s | %.4f |" % (
                    r["boot_s"], r["m_full_bins"], r["phi_fraction"],
                    fmt(c["chip_hours_incl_pre"]), pct(c["savings_vs_static_peak_incl_pre"]), c["ratio_vs_B0_incl_pre"],
                    fmt(i["chip_hours_incl_pre"]), pct(i["savings_vs_static_peak_incl_pre"]), i["ratio_vs_B0_incl_pre"]))
            L.append("\nWindow-only convention (R(t) integrated over [0, W) only) and overhead inside the envelope "
                     "(reserved - B0, instance-hours; not an extra term):\n")
            L.append("| B (s) | pre-obs cont inst-h | cont chip-h window-only | cont vs static | cont overhead inst-h | "
                     "pre-obs int inst-h | int chip-h window-only | int vs static | int overhead inst-h |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for r in ev["boots"]:
                c, i = r["continuous"], r["integer"]
                L.append("| %g | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["boot_s"], fmt(c["pre_observation_instance_hours"], 3), fmt(c["chip_hours_window_only"]),
                    pct(c["savings_vs_static_peak_window_only"]),
                    fmt(c["boot_and_bridge_overhead_instance_hours_incl_pre"]),
                    fmt(i["pre_observation_instance_hours"], 3), fmt(i["chip_hours_window_only"]),
                    pct(i["savings_vs_static_peak_window_only"]),
                    fmt(i["boot_and_bridge_overhead_instance_hours_incl_pre"])))
            L.append("\nSanity: %s\n" % json.dumps(ev["sanity"]))
        L.append("## Validation results\n")
        L.append("```\n%s\n```\n" % json.dumps(res["validation"], indent=1))
    L.append(METHODOLOGY)
    return "\n".join(L)


# --------------------------------------------------------------------------------------
def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print("downloading %s -> %s" % (url, dest), flush=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(TRACE_FILES), default="conv")
    ap.add_argument("--data", default=None, help="CSV path (default: data/<variant file>)")
    ap.add_argument("--download", action="store_true", help="download the CSV (and doc) if missing")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--capacity", default=str(DEFAULT_CAPACITY), help="tokens/s per instance (proxy)")
    ap.add_argument("--chips", type=int, default=DEFAULT_CHIPS)
    ap.add_argument("--bins", default="10,60", help="comma list of integer bin sizes in seconds (multiples of 10)")
    ap.add_argument("--boots", default=None, help="comma list of boot times in seconds (default: assignment+sweep grid)")
    ap.add_argument("--subset-hours", type=float, default=None, help="deterministic subset: first H hours from t0")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    cap = Fraction(args.capacity)
    bins = [int(b) for b in args.bins.split(",")]
    if any(b % 10 for b in bins):
        raise SystemExit("bin sizes must be multiples of 10 s")
    boots = ([float(b) if "." in b else int(b) for b in args.boots.split(",")] if args.boots
             else sorted(set(ASSIGNMENT_BOOTS) | set(SWEEP_BOOTS)))
    boots = sorted(set(b for b in boots if b > 0))

    t_start = time.time()
    print("== synthetic validation ==", flush=True)
    validation = {"synthetic": run_validation()}
    print(json.dumps(validation["synthetic"]), flush=True)
    if args.validate_only:
        return

    path = args.data or os.path.join(DATA_DIR, TRACE_FILES[args.variant])
    if not os.path.exists(path):
        if not args.download:
            raise SystemExit("missing %s ; rerun with --download" % path)
        download(BASE + TRACE_FILES[args.variant], path)
    doc_path = os.path.join(os.path.dirname(path), "AzureLLMInferenceDataset2024.md")
    if args.download and not os.path.exists(doc_path):
        download(DOC_URL, doc_path)

    size = os.path.getsize(path)
    digest = sha256_file(path)
    rec = RECORDED.get(args.variant)
    print("file %s bytes=%d sha256=%s" % (path, size, digest), flush=True)

    t0p = time.time()
    t_us, ctx, gen, st = load_trace(path, args.subset_hours)
    parse_s = time.time() - t0p
    rows = int(t_us.size)
    print("parsed %d rows in %.1fs (slow chunks: %d)" % (rows, parse_s, st["slow_chunks"]), flush=True)

    t0_us = (int(t_us.min()) // 60_000_000) * 60_000_000
    rel = t_us - t0_us
    n60 = int(rel.max()) // 60_000_000 + 1
    window_s = n60 * 60
    tok = ctx + gen
    c10, c60 = bin_series(rel, tok, n60 * 6)
    series = {10: c10, 60: c60}
    for b in bins:
        if b not in series:
            if b % 60 == 0:
                series[b] = c60.reshape(-1, b // 60).sum(axis=1) if c60.size % (b // 60) == 0 else None
            else:
                series[b] = c10.reshape(-1, b // 10).sum(axis=1) if c10.size % (b // 10) == 0 else None
            if series[b] is None:
                raise SystemExit("window %d s is not divisible by bin %d s" % (window_s, b))

    diffs = np.diff(t_us)
    gen_sum, ctx_sum = int(gen.sum()), int(ctx.sum())
    inp = {
        "path": path, "variant": args.variant, "bytes": size, "sha256": digest,
        "sha256_matches_recorded": (digest == rec["sha256"]) if rec else None,
        "recorded_reference": rec,
        "subset": ("first %g h from t0 (deterministic)" % args.subset_hours) if args.subset_hours else "full trace",
        "rows": rows,
        "first_timestamp": iso_us(t_us.min()), "last_timestamp": iso_us(t_us.max()),
        "t0": iso_us(t0_us), "window_s": window_s,
        "non_monotonic_steps": int((diffs < 0).sum()), "duplicate_timestamps": int((diffs == 0).sum()),
        "whole_second_timestamps": int((t_us % 1_000_000 == 0).sum()),
        "context": {"sum": ctx_sum, "mean": float(ctx.mean()), "min": int(ctx.min()), "max": int(ctx.max()),
                    "percentiles": pctiles(ctx), "zero_rows": int((ctx == 0).sum())},
        "generated": {"sum": gen_sum, "mean": float(gen.mean()), "min": int(gen.min()), "max": int(gen.max()),
                      "percentiles": pctiles(gen), "zero_rows": int((gen == 0).sum())},
        "generated_token_share": gen_sum / (gen_sum + ctx_sum),
        "total_tokens": ctx_sum + gen_sum,
    }

    validation["trace_slices"] = validate_on_trace_slice(c10, c60, cap)
    validation["trace_slices"]["bin_totals_equal_row_totals"] = True
    print("trace-slice brute-force validation passed", flush=True)

    demand = [demand_stats(series[b], b, cap) for b in bins]
    evals = []
    for b in bins:
        t1 = time.time()
        evals.append(evaluate(series[b], b, window_s, cap, args.chips, boots))
        print("evaluated %d s bins in %.1fs" % (b, time.time() - t1), flush=True)
    validation["result_sanity"] = {str(e["bin_s"]): e["sanity"] for e in evals}
    assert all(all(e["sanity"].values()) for e in evals), "result sanity check failed"

    res = {
        "metadata": {
            "script": os.path.abspath(__file__), "argv": sys.argv, "python": platform.python_version(),
            "numpy": np.__version__, "capacity_tokens_per_s": str(cap), "chips_per_instance": args.chips,
            "bins_s": bins, "boot_times_s": boots, "parse_seconds": round(parse_s, 1),
            "slow_parse_chunks": st["slow_chunks"], "total_seconds": round(time.time() - t_start, 1),
            "dataset_doc_url": DOC_URL, "trace_url": BASE + TRACE_FILES[args.variant],
            "kind": "token-arrival envelope counterfactual; not end-to-end serving; capacity is an illustrative work proxy; "
                    "chip-hours = modeled reserved footprint; boot already inside reserved envelope (no extra boot term)",
            "conventions": {
                "zero_padding": "demand = 0 outside [0, W); windows shrink at the end",
                "pre_observation": "*_incl_pre includes reserved capacity in [-B, 0); *_window_only integrates [0, W)",
                "static_peak": "peak bin demand x W (no pre-observation term)",
                "B0": "sum(tokens)/cap (continuous) or delta*sum(ceil) (integer)",
            },
        },
        "input": inp,
        "demand_stats": demand,
        "evaluations": evals,
        "validation": validation,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "trace_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(args.out_dir, "trace_report.md"), "w") as f:
        f.write(build_report(res, True))
    print("wrote %s/trace_results.json and trace_report.md" % args.out_dir)
    for ev in evals:
        print("-- %d s bins: B0 chip-h cont=%.1f int=%.1f | static cont=%.1f int=%.1f" % (
            ev["bin_s"], ev["baselines"]["B0"]["continuous"]["chip_hours"], ev["baselines"]["B0"]["integer"]["chip_hours"],
            ev["baselines"]["static_peak"]["continuous"]["chip_hours"], ev["baselines"]["static_peak"]["integer"]["chip_hours"]))
        for r in ev["boots"]:
            print("   B=%7g cont %.1f chip-h (save vs static %s) | int %.1f chip-h (save %s)" % (
                r["boot_s"], r["continuous"]["chip_hours_incl_pre"], pct(r["continuous"]["savings_vs_static_peak_incl_pre"]),
                r["integer"]["chip_hours_incl_pre"], pct(r["integer"]["savings_vs_static_peak_incl_pre"])))


if __name__ == "__main__":
    main()
