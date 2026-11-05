"""Headroom vs boot time across traces, with a decode-aware TP16 capacity model.

Capacity model, Gemma-4 31B on one TP16 instance (16 v5e chips), from measured rates:
  prefill: 24,394 tok/s (16k bucket)
  decode:  peak-batch tok/s per instance as a function of end context, measured in
           results/decode_allocation_20260913 (per chip x 16):
           512 -> 2,701; 768 -> 2,546; 1,024 -> 2,545; 1,280 -> 2,463; 2,304 -> 2,196; 4,352 -> 1,485.
           Beyond 4,352 the batch is KV-bound, so the rate is extrapolated as 1,485 * 4,352 / ctx [INFERENCE].
  A request costs ctx/24,394 + gen/decode_rate(ctx + gen) instance-seconds, charged to its arrival bin.

Headroom policy: provision d + h; when demand eats into h, launch capacity, which is ready B seconds
later. Overload happens iff R_B(t) = max_{0<=u<=B} d(t+u) - d(t) > h. We report quantiles of R_B as a
fraction of mean demand, its scaling exponent in B, and an integer-TP16 simulation (cheapest h and
keep-alive T meeting an overload-time target).

Run: ~/hetero-wt/Autoscale/.venv/bin/python multi_trace.py
"""
import csv
import json
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import replay

PREFILL = 24394.0
DEC_CTX = np.array([512, 768, 1024, 1280, 2304, 4352], float)
DEC_RATE = np.array([168.80, 159.13, 159.06, 153.91, 137.26, 92.81]) * 16
CHIPS = 16


def decode_rate(ctx):
    ctx = np.asarray(ctx, float)
    r = np.interp(np.log(np.clip(ctx, DEC_CTX[0], DEC_CTX[-1])), np.log(DEC_CTX), DEC_RATE)
    return np.where(ctx > DEC_CTX[-1], DEC_RATE[-1] * DEC_CTX[-1] / ctx, r)


def service(ctx, gen):
    ctx = np.asarray(ctx, float); gen = np.asarray(gen, float)
    return ctx / PREFILL + gen / decode_rate(ctx + gen)


def load_azure(name):
    t, c, g, _ = replay.load_trace(f"data/AzureLLMInferenceTrace_{name}_1week.csv")
    return (t - t.min()) / 1e6, c, g


def load_burstgpt(model):
    ts, c, g = [], [], []
    with open("data/burstgpt/BurstGPT_without_fails_1.csv") as fh:
        for r in csv.DictReader(fh):
            if r["Model"] == model:
                ts.append(float(r["Timestamp"])); c.append(int(r["Request tokens"])); g.append(int(r["Response tokens"]))
    ts = np.array(ts)
    return ts - ts.min(), np.array(c), np.array(g)


def load_mooncake(name):
    L = [json.loads(l) for l in open(f"data/mooncake/{name}_trace.jsonl")]
    ts = np.array([x["timestamp"] for x in L], float) / 1000.0
    return ts - ts.min(), np.array([x["input_length"] for x in L]), np.array([x["output_length"] for x in L])


TRACES = {
    "Azure conv (1 wk)": lambda: load_azure("conv"),
    "Azure code (1 wk)": lambda: load_azure("code"),
    "BurstGPT ChatGPT (61 d)": lambda: load_burstgpt("ChatGPT"),
    "BurstGPT GPT-4 (61 d)": lambda: load_burstgpt("GPT-4"),
    "Mooncake conversation (1 h)": lambda: load_mooncake("conversation"),
    "Mooncake toolagent (1 h)": lambda: load_mooncake("toolagent"),
}
BOOTS = {"TPUFlex": 50.2, "TPUFlex+GKE": 292.5, "tuned+GKE": 527.2, "default": 779.4, "default+GKE": 1021.7}
SWEEP = [60, 120, 180, 300, 450, 600, 900, 1200]


def fwd_max(a, k):
    if k <= 0:
        return a
    p = np.concatenate([a, np.full(k, a[-1])])
    return maximum_filter1d(p, size=k + 1, mode="nearest")[(k + 1) // 2: (k + 1) // 2 + len(a)]


def past(f, a, k):
    if k <= 0:
        return a
    p = np.concatenate([np.full(k, a[0]), a])
    return f(p, size=k + 1, mode="nearest")[(k + 1) // 2: (k + 1) // 2 + len(a)]


_r = np.random.default_rng(2).random(501)
assert all(np.allclose(fwd_max(_r, k), [_r[i:i + k + 1].max() for i in range(len(_r))]) for k in (1, 2, 3, 6, 17))


def rise_q(d, B, dt, q):
    k = max(1, int(round(B / dt)))
    R = (fwd_max(d, k) - d)[: len(d) - k]  # drop the tail where the window runs past the trace
    return float(np.quantile(R, q))


def int_sim(d, dt, B, target_time):
    best = None
    for T in (0, 60, 120, 300, 600, 900, 1200):
        for h in np.arange(0, 3.0 * d.mean() + 1, max(0.05, d.mean() / 100)):
            tgt = np.ceil(d + h - 1e-9)
            alloc = past(maximum_filter1d, tgt, int(round(T / dt)))
            ready = past(minimum_filter1d, alloc, max(1, int(round(B / dt))))
            if np.mean(d - ready > 1e-9) <= target_time:
                ch = CHIPS * alloc.sum() * dt / 3600
                if best is None or ch < best[0]:
                    best = (ch, float(h), T)
                break  # larger h only costs more at this T
    return best


out = {}
for name, loader in TRACES.items():
    ts, c, g = loader()
    s = service(c, g)
    dt = 60 if "Mooncake" not in name else 10
    nb = int(ts.max() // dt) + 1
    d = np.bincount((ts // dt).astype(int), weights=s, minlength=nb)[:nb] / dt
    mean = d.mean()
    row = {"requests": int(len(ts)), "span_h": float(ts.max() / 3600), "bin_s": dt,
           "mean_instances": float(mean), "peak_instances": float(d.max()),
           "mean_ctx": float(np.mean(c)), "mean_gen": float(np.mean(g)),
           "cv_bin": float(d.std() / mean), "h": {}}
    for bn, B in BOOTS.items():
        row["h"][bn] = {q: rise_q(d, B, dt, q) / mean for q in (0.99, 0.999, 1.0)}
    q99 = [rise_q(d, b, dt, 0.99) for b in SWEEP if b >= dt]
    row["alpha_q99"] = float(np.polyfit(np.log([b for b in SWEEP if b >= dt]), np.log(np.maximum(q99, 1e-9)), 1)[0])
    if "Mooncake" not in name:
        ideal = CHIPS * np.ceil(d).sum() * dt / 3600
        row["int_sim_0.1pct"] = {"ideal_chip_h": ideal}
        for bn in ("TPUFlex", "TPUFlex+GKE", "tuned+GKE", "default+GKE"):
            row["int_sim_0.1pct"][bn] = int_sim(d, dt, BOOTS[bn], 0.001)
    out[name] = row
    print(name, f"mean {mean:.2f} inst, alpha {row['alpha_q99']:.2f}", flush=True)
json.dump(out, open("multi_trace_results.json", "w"), indent=1)

print("\n## trace scale under the decode-aware TP16 capacity model")
print("| trace | requests | span | mean ctx / gen | mean (peak) TP16 instances | CV per bin | q99 ~ B^alpha |")
for n, r in out.items():
    print(f"| {n} | {r['requests']:,} | {r['span_h']:.1f} h | {r['mean_ctx']:.0f} / {r['mean_gen']:.0f} | "
          f"{r['mean_instances']:.2f} ({r['peak_instances']:.1f}) | {r['cv_bin']:.2f} | {r['alpha_q99']:.2f} |")
for q, lab in ((0.99, "p99"), (1.0, "max (never overload)")):
    print(f"\n## headroom h as % of mean demand, {lab}")
    print("| trace | " + " | ".join(f"{b} ({BOOTS[b]:.0f} s)" for b in BOOTS) + " | default+GKE - PM+GKE | default - PM (engine) |")
    for n, r in out.items():
        h = r["h"]
        print(f"| {n} | " + " | ".join(f"{100*h[b][q]:.0f}%" for b in BOOTS)
              + f" | {100*(h['default+GKE'][q]-h['TPUFlex+GKE'][q]):.0f} pp | {100*(h['default'][q]-h['TPUFlex'][q]):.0f} pp |")
print("\n## integer TP16 sim: cheapest (h, keep-alive T) with overloaded time <= 0.1%")
print("| trace | ideal chip-h | TPUFlex | TPUFlex+GKE | tuned+GKE | default+GKE | PM+GKE saves vs default+GKE / tuned+GKE |")
for n, r in out.items():
    s = r.get("int_sim_0.1pct")
    if not s:
        continue
    f = lambda b: "-" if s[b] is None else f"{s[b][0]:,.0f} (h={s[b][1]:.2f}, T={s[b][2]})"
    sv = lambda o: 100 * (s[o][0] - s["TPUFlex+GKE"][0]) / s[o][0]
    print(f"| {n} | {s['ideal_chip_h']:,.0f} | {f('TPUFlex')} | {f('TPUFlex+GKE')} | {f('tuned+GKE')} | {f('default+GKE')} | "
          f"{sv('default+GKE'):.1f}% / {sv('tuned+GKE'):.1f}% |")
