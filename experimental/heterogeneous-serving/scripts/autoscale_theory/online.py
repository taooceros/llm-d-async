"""Online (non-clairvoyant) reactive autoscaler on the Azure LLM conversation trace.

The policy only sees current demand.
- Scale-up: target(t) = ceil(d(t)) + h, where h is spare TP16 instances. Whenever
  allocated < target it launches the difference, and those instances are ready B seconds later.
- Scale-down: an instance is released after the target has stayed below its level for T seconds.
  T is the keep-alive timer.

With nested capacity layers, the closed forms are:
    alloc(t) = max_{s in [t-T, t]} target(s)    (booting + ready, billed)
    ready(t) = min_{s in [t-B, t]} alloc(s)     (a layer is ready once allocated continuously for B)
Unserved work = sum max(0, d(t) - ready(t)). Measured as a fraction of all work, this is a
capacity-shortfall proxy (queueing delay or shed load), not a latency SLO.

Run: python3 online.py   (needs data/AzureLLMInferenceTrace_conv_1week.csv, replay.py)
"""
import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
import replay

DT = 10
CHIPS = 16
t, c, g, _ = replay.load_trace("data/AzureLLMInferenceTrace_conv_1week.csv")
t0 = (int(t.min()) // 60_000_000) * 60_000_000
nbin = 604800 // DT
rel = t - t0
keep = rel < nbin * DT * 1_000_000
x, _ = replay.bin_series(rel[keep], (c + g)[keep], nbin)
d = x / DT / replay.DEFAULT_CAPACITY  # fluid demand in TP16 instances
W = d.sum()
n = np.ceil(d).astype(int)


def past_max(a, k):
    return a if k <= 0 else swv(np.concatenate([np.full(k, a[0]), a]), k + 1).max(axis=1)


def past_min(a, k):
    return a if k <= 0 else swv(np.concatenate([np.full(k, a[0]), a]), k + 1).min(axis=1)


def fwd_max(a, k):
    return a if k <= 0 else swv(np.concatenate([a, np.full(k, a[-1])]), k + 1).max(axis=1)


def online(B, T, h):
    kb, kt = int(round(B / DT)), int(round(T / DT))
    target = n + h
    alloc = past_max(target, kt)
    ready = past_min(alloc, kb)
    short = np.maximum(0.0, d - ready)
    return {"chip_h": CHIPS * alloc.sum() * DT / 3600, "unserved": short.sum() / W,
            "overload_time": float(np.mean(short > 0)),
            "launches": int(np.maximum(np.diff(alloc), 0).sum())}


ideal = CHIPS * n.sum() * DT / 3600
clair = {B: CHIPS * fwd_max(n, int(round(B / DT))).sum() * DT / 3600 for B in (50.2, 284.9, 292.5, 527.2, 779.4, 1021.7)}
static = CHIPS * n.max() * nbin * DT / 3600
BOOTS = {"TPUFlex": (50.2, 292.5), "tuned": (284.9, 527.2), "default": (779.4, 1021.7)}
TIMERS = {"T=B (ski rental)": None, "T=600 s (GKE)": 600.0, "T=0": 0.0}
SLOS = [0.01, 0.001, 0.0001]

out = {"ideal_chip_h": ideal, "static_peak_chip_h": static, "clairvoyant": clair, "rows": []}
for approach, (b_eng, b_gke) in BOOTS.items():
    for label, B in (("engine only", b_eng), ("+GKE", b_gke)):
        for tname, T in TIMERS.items():
            Tv = B if T is None else T
            curve = [(h, online(B, Tv, h)) for h in range(0, 16)]
            row = {"approach": approach, "variant": label, "B": B, "timer": tname,
                   "h0": curve[0][1], "slo": {}}
            for slo in SLOS:
                ok = [(h, r) for h, r in curve if r["unserved"] <= slo]
                row["slo"][str(slo)] = None if not ok else {"h": ok[0][0], **ok[0][1]}
            out["rows"].append(row)
json.dump(out, open("online_results.json", "w"), indent=1)

print(f"ideal (B=0) {ideal:,.0f} chip-h/week; static peak {static:,.0f}")
print("\n## h = 0 (purely reactive): cost and shortfall")
print("| approach | variant | B | timer | chip-h | vs ideal | unserved work | overloaded time | launches |")
for r in out["rows"]:
    z = r["h0"]
    print(f"| {r['approach']} | {r['variant']} | {r['B']:.0f} | {r['timer']} | {z['chip_h']:,.0f} | "
          f"+{100*(z['chip_h']/ideal-1):.1f}% | {100*z['unserved']:.2f}% | {100*z['overload_time']:.1f}% | {z['launches']} |")
print("\n## equal shortfall: minimum headroom h meeting unserved <= SLO, best timer")
print("| approach | variant | B | clairvoyant chip-h | " + " | ".join(f"SLO {s:g}: h, chip-h" for s in SLOS) + " |")
for approach in BOOTS:
    for variant in ("engine only", "+GKE"):
        rows = [r for r in out["rows"] if r["approach"] == approach and r["variant"] == variant]
        cells = []
        for slo in SLOS:
            cands = [(r["slo"][str(slo)]["chip_h"], r["slo"][str(slo)]["h"], r["timer"]) for r in rows if r["slo"][str(slo)]]
            best = min(cands) if cands else None
            cells.append("-" if not best else f"h={best[1]}, {best[0]:,.0f} ({best[2].split(' ')[0]})")
        print(f"| {approach} | {variant} | {rows[0]['B']:.0f} | {clair[rows[0]['B']]:,.0f} | " + " | ".join(cells) + " |")
