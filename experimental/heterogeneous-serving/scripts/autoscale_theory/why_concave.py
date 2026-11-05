"""Why allocated chip-hours vs boot time B is concave: decompose the envelope overhead on the Azure trace.
Integer TP16 fleet, 10 s bins, clairvoyant envelope Q_B(t) = max_{0<=u<=B} n(t+u), n = ceil(demand)."""
import numpy as np, replay
from numpy.lib.stride_tricks import sliding_window_view as swv
t, c, g, _ = replay.load_trace("data/AzureLLMInferenceTrace_conv_1week.csv")
t0 = (int(t.min()) // 60_000_000) * 60_000_000
n10 = 604800 // 10
rel = t - t0; keep = rel < n10 * 10_000_000
x, _ = replay.bin_series(rel[keep], (c + g)[keep], n10)
n = np.ceil(x / 10.0 / replay.DEFAULT_CAPACITY).astype(int)
base = n.sum()
pad = np.concatenate([n, np.full(200, n[-1])])
def env(k):
    return swv(pad, k + 1)[:len(n)].max(axis=1) if k else n
def runs_below(q):  # (#instance-launches, mean idle-gap bridged) for unit capacity layers
    ups = np.maximum(np.diff(q), 0).sum()
    return int(ups)
# idle-gap distribution of capacity layers in raw demand: a gap of length G at level L costs min(G,B) once bridged
gaps = []
for L in range(1, n.max() + 1):
    above = (n >= L).astype(np.int8)
    d = np.diff(np.concatenate([[1], above, [1]]))
    starts, ends = np.where(d == -1)[0], np.where(d == 1)[0]
    gaps += list((ends - starts) * 10)
gaps = np.array(gaps)
print(f"raw demand: {len(gaps)} idle gaps across capacity layers; median {np.median(gaps):.0f} s, "
      f"share <=60 s: {100*np.mean(gaps<=60):.0f}%, <=300 s: {100*np.mean(gaps<=300):.0f}%, <=1000 s: {100*np.mean(gaps<=1000):.0f}%")
print("|  B (s) | overhead % | launches/week | overhead per launch (B-units) | ln-fit | sqrt-fit | sum_gaps min(G,B) % |")
Bs = [10, 30, 60, 120, 300, 600, 1020]
ov = []
for B in Bs:
    k = B // 10
    q = env(k)
    o = (q.sum() - base) / base
    launches = runs_below(q)
    per = (q.sum() - base) * 10 / max(launches, 1) / B
    bridge = np.minimum(gaps, B).sum() / 10 / base  # ski-rental bridging cost of all gaps (lower bound view)
    ov.append(o)
    print(f"| {B:6d} | {100*o:6.2f} | {launches:7d} | {per:5.2f} | | | {100*bridge:6.2f} |")
lb = np.log(Bs); sb = np.sqrt(Bs)
for name, f in [("a + b ln B", lb), ("a + b sqrt B", sb), ("b B (linear)", np.array(Bs, float))]:
    A = np.vstack([np.ones_like(f), f]).T if "linear" not in name else f[:, None]
    coef, res, *_ = np.linalg.lstsq(A, np.array(ov), rcond=None)
    pred = A @ coef
    print(f"{name:14s} fit RMSE {100*np.sqrt(np.mean((pred-np.array(ov))**2)):.2f} pp")
