"""Reactive autoscaling with headroom: how much spare capacity is needed for a given boot time B?

Policy: always provision demand + h spare capacity. When demand eats into the headroom (d(t) + h >
allocated), launch instances; they become ready B seconds later. Over those B seconds the spare h
must absorb the rise, so a burst overloads the fleet iff

    R_B(t) = max_{0 <= u <= B} d(t+u) - d(t) > h.

(1) Headroom from the rise distribution: h_p(B) = p-quantile of R_B(t) over time. Overload probability
    per scale-up window <= 1 - p. Compared with the diffusion theory h = z * sigma * sqrt(B).
(2) Full simulation, integer TP16 instances: target = ceil(d + h), alloc = max target over [t-T, t],
    ready = min alloc over [t-B, t]. For each B, find the smallest h (and the cheapest keep-alive T)
    that meets an overload target, and report chip-hours.

Run: ~/hetero-wt/Autoscale/.venv/bin/python headroom.py
"""
import json
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import replay

CHIPS = 16
t, c, g, _ = replay.load_trace("data/AzureLLMInferenceTrace_conv_1week.csv")
t0 = (int(t.min()) // 60_000_000) * 60_000_000
rel = t - t0
nb10 = 604800 // 10
keep = rel < nb10 * 10_000_000
x10, _ = replay.bin_series(rel[keep], (c + g)[keep], nb10)


def demand(dt):
    x = x10.reshape(-1, dt // 10).sum(axis=1)
    return x / dt / replay.DEFAULT_CAPACITY  # TP16 instances (fluid)


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


# sanity check of window alignment
_r = np.random.default_rng(1).random(997)
assert all(np.allclose(fwd_max(_r, k), [ _r[i:i + k + 1].max() for i in range(len(_r))]) for k in (1, 2, 5, 8))

BOOTS = {"TPUFlex": 50.2, "tuned": 284.9, "default": 779.4,
         "TPUFlex+GKE": 292.5, "tuned+GKE": 527.2, "default+GKE": 1021.7}
QS = [0.99, 0.999, 1.0]
out = {"bins": {}}
for dt in (60, 10):
    d = demand(dt)
    mean = d.mean()
    rows = {}
    # per-bin volatility of increments for the sqrt(B) model (robust: MAD of first differences)
    inc = np.diff(d)
    sigma = 1.4826 * np.median(np.abs(inc - np.median(inc))) / np.sqrt(dt)  # instances / sqrt(s)
    for name, B in BOOTS.items():
        k = max(1, int(round(B / dt)))
        R = fwd_max(d, k) - d
        rows[name] = {"B": B, **{f"q{q}": float(np.quantile(R, q)) for q in QS},
                      "sqrt_model_q0.99": float(2.326 * sigma * np.sqrt(B))}
    # sqrt-B fit of the 99th percentile across a sweep
    sweep = [20, 30, 60, 120, 180, 300, 450, 600, 900, 1200]
    sweep = [b for b in sweep if b >= dt]
    q99 = [float(np.quantile(fwd_max(d, int(round(b / dt))) - d, 0.99)) for b in sweep]
    A = np.log(sweep); slope = np.polyfit(A, np.log(q99), 1)[0]
    out["bins"][dt] = {"mean_demand_instances": float(mean), "peak": float(d.max()), "sigma": float(sigma),
                       "rows": rows, "sweep": dict(zip(sweep, q99)), "loglog_slope_q99": float(slope)}

# (2) integer simulation on 60 s bins: smallest h meeting overload-time target, cheapest keep-alive
d = demand(60)
W = d.sum()
ideal = CHIPS * np.ceil(d).sum() * 60 / 3600
HS = np.round(np.arange(0, 4.01, 0.05), 2)
TS = [0, 60, 120, 300, 600, 900, 1200]
TARGETS = {"overload time <= 1%": ("time", 0.01), "overload time <= 0.1%": ("time", 0.001),
           "unserved work <= 0.01%": ("work", 1e-4)}


def sim(B, T, h):
    target = np.ceil(d + h - 1e-9)
    alloc = past(maximum_filter1d, target, int(round(T / 60)))
    ready = past(minimum_filter1d, alloc, int(round(B / 60)))
    short = np.maximum(0, d - ready)
    return CHIPS * alloc.sum() * 60 / 3600, float(np.mean(short > 1e-9)), short.sum() / W


sim_rows = {}
for name, B in BOOTS.items():
    pts = [(h, T, *sim(B, T, h)) for h in HS for T in TS]
    r = {}
    for label, (kind, lim) in TARGETS.items():
        ok = [p for p in pts if (p[3] if kind == "time" else p[4]) <= lim]
        best = min(ok, key=lambda p: p[2]) if ok else None
        r[label] = None if best is None else {"h": best[0], "T": best[1], "chip_h": best[2]}
    sim_rows[name] = {"B": B, **r}
out["sim60"] = {"ideal_chip_h": ideal, "rows": sim_rows}
json.dump(out, open("headroom_results.json", "w"), indent=1)

for dt, v in out["bins"].items():
    print(f"\n## rise R_B within one boot window, {dt} s bins; mean demand {v['mean_demand_instances']:.2f}, "
          f"peak {v['peak']:.2f} TP16 instances; log-log slope of q99 vs B = {v['loglog_slope_q99']:.2f}")
    print("| approach | B (s) | h p99 | h p99.9 | h max | sqrt model p99 |")
    for name, r in v["rows"].items():
        print(f"| {name} | {r['B']:.0f} | {r['q0.99']:.2f} | {r['q0.999']:.2f} | {r['q1.0']:.2f} | {r['sqrt_model_q0.99']:.2f} |")
print(f"\n## integer TP16 simulation (60 s bins), ideal {ideal:,.0f} chip-h/week")
print("| approach | B | " + " | ".join(TARGETS) + " |")
for name, r in sim_rows.items():
    cells = ["-" if r[k] is None else f"h={r[k]['h']:.2f}, T={r[k]['T']}, {r[k]['chip_h']:,.0f} chip-h" for k in TARGETS]
    print(f"| {name} | {r['B']:.0f} | " + " | ".join(cells) + " |")
