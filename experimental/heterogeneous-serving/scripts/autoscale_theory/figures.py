#!/usr/bin/env python3
"""Figures for the autoscaling boot-time report (Google Doc tab + paper candidates).

  fig1_gke_scaleup.png    GKE TPU v5e scale-from-zero stage breakdown, 5 topologies x on-demand/spot
  fig2_full_scaleup.png   Full fresh-host TP16 scale-up = GKE provisioning + engine boot, per approach
  fig3_envelope.png       Azure trace slice: demand d(t) vs allocation envelope Q_B for two boot times
  fig4_chiphours_vs_B.png Weekly allocated chip-hours vs boot time B (trace replay), measured B marked

    python3 figures.py --gke <campaign dir> --engine <autoscale_figure.py --json output> --sweep sweep/trace_results.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

C_PROV, C_PULL, C_JAX = "#9aa5b1", "#c9b37e", "#6c8ebf"
ENGINE_COLORS = {"download": "#e6a157", "spawn": "#b9b9b9", "init_device": "#d9d9d9", "load_model": "#c0504d",
                 "kv": "#8064a2", "compile": "#4f81bd"}
ENGINE_LABELS = {"download": "state/checkpoint download", "spawn": "spawn + imports", "init_device": "init device",
                 "load_model": "load_model", "kv": "KV cache init", "compile": "compile + warm-up"}
TOPOS = ["1x1", "2x2", "2x4", "4x4", "4x8"]
CHIPS = {"1x1": 1, "2x2": 4, "2x4": 8, "4x4": 16, "4x8": 32}
HOSTS = {"1x1": 1, "2x2": 1, "2x4": 1, "4x4": 4, "4x8": 8}


def load_gke(campaign: Path) -> dict:
    arms: dict[str, list[dict]] = {}
    for line in (campaign / "samples.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("status") == "complete":
            arms.setdefault(rec["arm"], []).append(rec["derived"])
    return arms


def median_run(runs: list[dict]) -> dict:
    runs = sorted(runs, key=lambda d: d["jax_devices"])
    return runs[len(runs) // 2]


def gke_segments(d: dict) -> tuple[float, float, float]:
    prov = d["pod_scheduled"]
    pull = d["image_pulled"] - d["pod_scheduled"]
    jax = d["jax_devices"] - d["image_pulled"]
    return prov, pull, jax


def fig1(arms: dict, out: Path) -> None:
    rows = [(t, m) for t in TOPOS for m in ("od", "spot")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = []
    for i, (t, m) in enumerate(rows):
        runs = arms[f"{t}-{m}"]
        prov, pull, jax = gke_segments(median_run(runs))
        y = len(rows) - 1 - i
        ax.barh(y, prov, color=C_PROV, height=0.65, label="node provisioning (VM, register, Ready, TPU plugin)" if i == 0 else None)
        ax.barh(y, pull, left=prov, color=C_PULL, height=0.65, label="image pull (2.7 GB)" if i == 0 else None)
        ax.barh(y, jax, left=prov + pull, color=C_JAX, height=0.65, label="container start + JAX init" if i == 0 else None)
        totals = [r["jax_devices"] for r in runs]
        tot = prov + pull + jax
        ax.plot([min(totals), max(totals)], [y, y], color="black", lw=0.8)
        ax.plot([min(totals)] * 2, [y - 0.15, y + 0.15], color="black", lw=0.8)
        ax.plot([max(totals)] * 2, [y - 0.15, y + 0.15], color="black", lw=0.8)
        ax.text(max(totals) + 6, y, f"{tot:.0f} s", va="center", fontsize=8.5)
        labels.append(f"{t} ({CHIPS[t]} chip{"s" if CHIPS[t] > 1 else ""}, {HOSTS[t]} host{"s" if HOSTS[t] > 1 else ""}) {'on-demand' if m == 'od' else 'spot'}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels[::-1], fontsize=8.5)
    ax.set_xlabel("seconds from Job submission to jax.devices() on every host")
    ax.set_xlim(0, 430)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2, fontsize=8, frameon=False)
    ax.set_title("GKE TPU v5e scale-from-zero (median run of 5; whisker = min–max)", fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def fig2(arms: dict, engine: dict, out: Path) -> None:
    gke = median_run(arms["4x4-od"])
    prov, pull, jax = gke_segments(gke)
    approaches = [("vLLM default\n(lazy GCS FUSE,\nempty XLA cache)", "raw"),
                  ("Tuned baseline\n(pre-copied ckpt,\nwarm XLA cache)", "ckpt-gcs-xla"),
                  ("TPUFlex\n(pre-sharded weights,\nExecutable Bank)", "peer")]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    seen = set()
    for i, (label, key) in enumerate(approaches):
        y = len(approaches) - 1 - i
        left = 0.0
        for name, width, color in [("GKE node provisioning", prov, C_PROV), ("GKE image pull", pull, C_PULL),
                                   ("container start + JAX init", jax, C_JAX)]:
            ax.barh(y, width, left=left, color=color, height=0.6, hatch="//", edgecolor="white", lw=0,
                    label=name if name not in seen else None)
            seen.add(name)
            left += width
        gke_end = left
        for seg, width in engine[key]["segments"].items():
            if width <= 0:
                continue
            ax.barh(y, width, left=left, color=ENGINE_COLORS[seg], height=0.6,
                    label=ENGINE_LABELS[seg] if seg not in seen else None)
            seen.add(seg)
            left += width
        total = gke_end + engine[key]["total"]
        chip_h = 16 * total / 3600
        ax.text(total + 10, y, f"{total:,.0f} s  ({chip_h:.2f} chip-h,\nGKE {100 * gke_end / total:.0f}%)",
                va="center", fontsize=8.5)
    ax.axvline(gke_end, color="black", lw=0.6, ls=":")
    ax.text(gke_end + 4, 2.45, f"GKE {gke_end:.0f} s", fontsize=8)
    ax.set_yticks(range(len(approaches)))
    ax.set_yticklabels([a[0] for a in approaches][::-1], fontsize=8.5)
    ax.set_xlabel("seconds until a new TP16 instance (4 fresh hosts) serves")
    ax.set_xlim(0, 1300)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=7.5, frameon=False)
    fig.savefig(out)
    plt.close(fig)


def fig3(out: Path, b_small: float, b_large: float, hours: float = 2.0) -> None:
    import replay  # /tmp/autoscale_theory/replay.py

    path = HERE / "data" / "AzureLLMInferenceTrace_conv_1week.csv"
    t, c, g, _ = replay.load_trace(str(path))
    t0 = (int(t.min()) // 60_000_000) * 60_000_000
    rel = t - t0
    n10 = 604800 // 10
    keep = rel < n10 * 10_000_000
    c10, _ = replay.bin_series(rel[keep], (c + g)[keep], n10)
    d = c10.reshape(-1, 6).sum(axis=1) / 60.0 / replay.DEFAULT_CAPACITY  # 60 s bins, fluid TP16 instance units
    # window (hours long) containing the steepest one-hour rise of demand
    w = int(hours * 60)
    rise = np.array([d[i + 60] - d[i] for i in range(len(d) - 60)])
    peak = int(np.argmax(rise))
    best = max(0, min(peak - w // 3, len(d) - w - 40))
    seg = d[best:best + w + 40]
    x = np.arange(w)

    def env(B):
        k = B / 60.0
        m, phi = int(k), k - int(k)
        return np.array([(1 - phi) * seg[i:i + m + 1].max() + phi * seg[i:i + m + 2].max() for i in range(w)])

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    for B, color in [(b_large, "#c0504d"), (b_small, "#4f81bd")]:
        q = env(B)
        extra = 100 * (q.sum() - seg[:w].sum()) / seg[:w].sum()
        ax.plot(x, q, color=color, lw=1.4, label=f"allocation Q_B(t), B = {B:.0f} s  (+{extra:.1f}% in window)")
        ax.fill_between(x, seg[:w], q, color=color, alpha=0.18, lw=0)
    ax.plot(x, seg[:w], color="black", lw=0.9, label="demand d(t) = allocation with B = 0")
    ax.set_xlabel(f"minutes (Azure LLM conversation trace, {hours:.0f} h window around its steepest ramp, 60 s bins)")
    ax.set_ylabel("TP16 instances (fluid)")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    fig.savefig(out)
    plt.close(fig)


def fig4(sweep: Path, out: Path, marks: dict) -> None:
    res = json.loads(sweep.read_text())["evaluations"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for e, style in zip(sorted(res, key=lambda e: e["bin_s"]), ["-", "--"]):
        pts = sorted((b["boot_s"], b["integer"]["chip_hours_incl_pre"]) for b in e["boots"])
        h0 = e["baselines"]["B0"]["integer"]["chip_hours"]
        xs = [p[0] for p in pts]
        ys = [100 * (p[1] - h0) / h0 for p in pts]
        ax.plot(xs, ys, style, color="#444", lw=1.2, label=f"{e['bin_s']} s demand bins")
        if e["bin_s"] == 10:
            lut = dict(pts)
            for (label, B), color in zip(marks.items(), ["#c0504d", "#e6a157", "#4f81bd"] * 2):
                ys_b = 100 * (lut[B] - h0) / h0
                engine_only = "engine only" in label
                ax.plot(B, ys_b, "o" if not engine_only else "s", color=color, ms=6,
                        mfc=color if not engine_only else "white")
                text = label.replace(" (engine only)", ", engine only") if engine_only else f"{label} + GKE"
                ax.annotate(text, (B, ys_b), textcoords="offset points",
                            xytext=(6, -12 if engine_only else 5), fontsize=7.5, color=color)
    ax.set_xlabel("boot time B (s)")
    ax.set_ylabel("allocated chip-hours over B = 0 ideal (%)")
    ax.set_xlim(0, 1200)
    ax.plot([], [], "o", color="#666", label="GKE included (B_GKE + B_engine)")
    ax.plot([], [], "s", color="#666", mfc="white", label="engine only")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Trace replay, 1 week, integer TP16 fleet, clairvoyant envelope", fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gke", type=Path, required=True)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arms = load_gke(args.gke)
    engine = json.loads(args.engine.read_text())
    fig1(arms, args.out / "fig1_gke_scaleup.png")
    fig2(arms, engine, args.out / "fig2_full_scaleup.png")
    gke = median_run(arms["4x4-od"])["jax_devices"]
    print(f"B_GKE (4x4 on-demand, median run) = {gke:.1f} s")
    marks = {"default": 1021.7, "tuned": 527.2, "TPUFlex": 292.5,
             "default (engine only)": 779.4, "tuned (engine only)": 284.9, "TPUFlex (engine only)": 50.2}
    fig4(args.sweep, args.out / "fig4_chiphours_vs_B.png", marks)
    fig3(args.out / "fig3_envelope.png", 292.5, 1021.7)
    print("wrote", sorted(p.name for p in args.out.glob("*.png")))


if __name__ == "__main__":
    main()
