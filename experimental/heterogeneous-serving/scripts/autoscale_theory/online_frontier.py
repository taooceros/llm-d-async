"""Online reactive autoscaler: the cost of not knowing how long a burst will last.

The policy sees only current demand. Its knobs:
- m: multiplicative margin. target(t) = ceil(d(t) * (1 + m)) TP16 instances.
- T: keep-alive timer. A layer is released after target has stayed below it for T seconds.
- B: boot time. A launched layer becomes ready after B seconds.
Allocated (billed) capacity and ready capacity:
    alloc(t) = max_{s in [t-T, t]} target(s)
    ready(t) = min_{s in [t-B, t]} alloc(s)
Shortfall = sum max(0, d - ready) / sum d, i.e., the fraction of offered work that arrives while
capacity is short (it queues or is shed).

For each B we sweep (T, m) and report the cheapest policy that meets a shortfall target. That is
the chip-hour price of boot time when burst length is unknown. It is compared with the clairvoyant
envelope, which has zero shortfall.

Run: ~/hetero-wt/Autoscale/.venv/bin/python online_frontier.py
"""
import json
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import replay

DT, CHIPS = 10, 16
t, c, g, _ = replay.load_trace("data/AzureLLMInferenceTrace_conv_1week.csv")
t0 = (int(t.min()) // 60_000_000) * 60_000_000
nbin = 604800 // DT
rel = t - t0
keep = rel < nbin * DT * 1_000_000
x, _ = replay.bin_series(rel[keep], (c + g)[keep], nbin)
d = x / DT / replay.DEFAULT_CAPACITY
W, N = d.sum(), len(d)


def past(f, a, k):
    if k <= 0:
        return a
    p = np.concatenate([np.full(k, a[0]), a])
    return f(p, size=k + 1, mode="nearest")[(k + 1) // 2: (k + 1) // 2 + N]


def sim(B, T, m):
    target = np.ceil(d * (1 + m) - 1e-9)
    alloc = past(maximum_filter1d, target, int(round(T / DT)))
    ready = past(minimum_filter1d, alloc, int(round(B / DT)))
    short = np.maximum(0.0, d - ready)
    return CHIPS * alloc.sum() * DT / 3600, short.sum() / W, float(np.mean(short > 1e-9))


ideal = CHIPS * np.ceil(d).sum() * DT / 3600
BOOTS = [("TPUFlex", "engine only", 50.2), ("tuned", "engine only", 284.9), ("default", "engine only", 779.4),
         ("TPUFlex", "+GKE", 292.5), ("tuned", "+GKE", 527.2), ("default", "+GKE", 1021.7)]
TS = list(range(0, 2401, 30))
MS = [round(0.025 * i, 3) for i in range(0, 17)]
SLOS = [1e-2, 1e-3, 1e-4]
res = {"ideal_chip_h": ideal, "grid": {"T": TS, "m": MS}, "boots": {}}
for name, variant, B in BOOTS:
    pts = [(T, m, *sim(B, T, m)) for T in TS for m in MS]
    best = {}
    for s in SLOS:
        ok = [p for p in pts if p[3] <= s]
        best[str(s)] = None if not ok else dict(zip(["T", "m", "chip_h", "shortfall", "overload_time"], min(ok, key=lambda p: p[2])))
    fixed = sim(B, 600, 0.0)  # GKE-like fixed policy: 600 s keep-alive, no margin
    res["boots"][f"{name} {variant}"] = {"B": B, "best": best,
                                       "fixed_T600_m0": dict(zip(["chip_h", "shortfall", "overload_time"], fixed))}
json.dump(res, open("online_frontier.json", "w"), indent=1)

print(f"ideal B=0 clairvoyant: {ideal:,.0f} chip-h/week")
print("\n| approach | B (s) | fixed policy T=600 s, m=0: shortfall / overloaded time | "
      + " | ".join(f"cheapest for shortfall <= {s:g}: chip-h (T, m)" for s in SLOS) + " |")
for k, v in res["boots"].items():
    f = v["fixed_T600_m0"]
    cells = []
    for s in SLOS:
        b = v["best"][str(s)]
        cells.append("-" if not b else f"{b['chip_h']:,.0f} (+{100*(b['chip_h']/ideal-1):.1f}%; T={b['T']}, m={b['m']})")
    print(f"| {k} | {v['B']:.0f} | {100*f['shortfall']:.3f}% / {100*f['overload_time']:.2f}% | " + " | ".join(cells) + " |")
print("\nsavings at equal shortfall (TPUFlex vs X):")
for variant in ("engine only", "+GKE"):
    for other in ("tuned", "default"):
        for s in SLOS:
            a = res["boots"][f"TPUFlex {variant}"]["best"][str(s)]
            b = res["boots"][f"{other} {variant}"]["best"][str(s)]
            if a and b:
                print(f"  {variant:11s} vs {other:7s} shortfall<={s:g}: {b['chip_h']-a['chip_h']:,.0f} chip-h/week "
                      f"({100*(b['chip_h']-a['chip_h'])/b['chip_h']:.1f}%)")
