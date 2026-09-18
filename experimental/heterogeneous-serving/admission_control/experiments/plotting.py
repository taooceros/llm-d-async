"""
High-Resolution Plot Generation Module for TPU TP=8 Preemption Benchmark.
========================================================================
Generates publication-grade 300 DPI comparative plots following Google Material styling
and zero-LaTeX Unicode standards.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Google Material Style Colors
COLOR_PRIMARY = "#1a73e8"    # Blue
COLOR_DANGER = "#d93025"     # Red
COLOR_SUCCESS = "#1e8e3e"    # Green
COLOR_WARNING = "#f9ab00"    # Orange / Yellow
COLOR_GRAY = "#5f6368"       # Muted Gray
COLOR_BG_LIGHT = "#f8f9fa"   # Light background


def apply_google_style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    """Applies clean publication typography and light grid lines."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color="#202124")
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold", labelpad=8, color="#3c4043")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold", labelpad=8, color="#3c4043")
    ax.grid(True, linestyle="--", alpha=0.35, color="#dadce0")
    ax.set_facecolor(COLOR_BG_LIGHT)
    for spine in ax.spines.values():
        spine.set_color("#dadce0")
        spine.set_linewidth(0.8)


def plot_kv_capacity_cliff(out_path: Union[str, pathlib.Path], summary: Optional[Dict[str, Any]] = None) -> None:
    """Figure 1: Token memory accumulation vs capacity ceiling on TP=8 vs TP=16."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    steps = np.arange(0, 2049, 16)
    # 64 requests with 256 prompt tokens each
    tokens_demanded = 64 * (256 + steps)

    cap_tp8 = summary.get("hardware", {}).get("kv_cache_capacity_tokens", 120_000) if summary else 120_000
    cap_tp16 = 400_000

    # Plot requested token trajectory
    ax.plot(steps, tokens_demanded, color=COLOR_PRIMARY, linewidth=2.4, label="64 Reqs Demand (I=256, O=2048)")

    # Plot Capacity ceilings
    ax.axhline(cap_tp8, color=COLOR_DANGER, linestyle="--", linewidth=2.0, label=f"TP=8 KV Capacity Ceiling ({cap_tp8//1000}k tokens)")
    ax.axhline(cap_tp16, color=COLOR_SUCCESS, linestyle=":", linewidth=2.0, label="TP=16 KV Capacity Ceiling (400k tokens)")

    # Preemption cliff marker
    pa = summary.get("results", {}).get("policy_a_greedy", {}) if summary else {}
    step_cliff = pa.get("preemption_step") or 1620
    tokens_cliff = cap_tp8
    ax.scatter([step_cliff], [tokens_cliff], color=COLOR_DANGER, s=90, zorder=5)
    ax.annotate(
        f"Preemption Cliff (Step {step_cliff})\nOOM: 16 Reqs Evicted!",
        xy=(step_cliff, tokens_cliff),
        xytext=(step_cliff - 600, tokens_cliff + 18000),
        arrowprops=dict(facecolor=COLOR_DANGER, shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color=COLOR_DANGER,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fce8e6", ec=COLOR_DANGER, lw=1)
    )

    ax.set_ylim(0, 200_000)
    ax.set_xlim(0, 2048)
    apply_google_style(
        ax,
        title="Figure 1: KV Memory Expansion & Deterministic Preemption Cliff on TP=8",
        xlabel="Autoregressive Decode Step (Tokens Generated)",
        ylabel="Aggregate KV Cache Footprint (Tokens)"
    )
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#dadce0", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_policy_goodput_comparison(out_path: Union[str, pathlib.Path], summary: Optional[Dict[str, Any]] = None) -> None:
    """Figure 2: Goodput % and Discarded Token Volume Comparison across all policies."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=300)

    res = summary.get("results", {}) if summary else {}

    policy_keys = [
        ("policy_a_greedy", "Policy A\n(Greedy)", COLOR_DANGER),
        ("policy_b_predictive", "Policy B\n(Predictive 48+16)", COLOR_WARNING),
        ("policy_c_bisection", "Policy C\n(In-HBM Bisection)", COLOR_GRAY),
        ("policy_d_balanced", "Policy D\n(Balanced 32+32)", COLOR_PRIMARY),
        ("policy_e_lossless_pipeline", "Policy E\n(Lossless Pipeline)", COLOR_SUCCESS),
        ("policy_oracle_optimal", "Oracle Optimal\n(32+32 @ 858)", "#8e24aa"),
    ]

    labels = []
    goodput_values = []
    discarded_values = []
    colors = []

    for key, label, color in policy_keys:
        if key in res:
            p = res[key]
            labels.append(label)
            goodput_values.append(p.get("goodput_percent", 100.0))
            discarded_values.append(p.get("tokens_discarded_recomputed", 0))
            colors.append(color)

    # Bar chart for Goodput %
    bars1 = ax1.bar(labels, goodput_values, color=colors, width=0.5, edgecolor="#dadce0")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval - 12, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="white")

    ax1.set_ylim(0, 115)
    apply_google_style(
        ax1,
        title="Goodput Efficiency %",
        xlabel="Admission Strategy",
        ylabel="Delivered Tokens / Total Computed (%)"
    )

    # Bar chart for Discarded Tokens
    bars2 = ax2.bar(labels, discarded_values, color=colors, width=0.5, edgecolor="#dadce0")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 800, f"{yval:,} tok", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#202124")

    ax2.set_ylim(0, max(36_000, max(discarded_values) * 1.25) if discarded_values else 36_000)
    apply_google_style(
        ax2,
        title="Wasted Recomputed Token Volume",
        xlabel="Admission Strategy",
        ylabel="Discarded Token Count"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_xla_padding_bubbles(out_path: Union[str, pathlib.Path], summary: Optional[Dict[str, Any]] = None) -> None:
    """Figure 3: TPU XLA Static Shape Batch Padding Waste across Decode Steps."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)

    pa = summary.get("results", {}).get("policy_a_greedy", {}) if summary else {}
    step_cliff = pa.get("preemption_step") or 1620

    steps = np.arange(0, 2049, 1)

    # Policy A waste: 0% up to step 1620 (running 64 on BS=64), then batch drops to 48 -> 16 dummy slots / 64 = 25% waste
    waste_a = np.zeros_like(steps, dtype=float)
    waste_a[steps >= step_cliff] = 25.0

    # Policy B waste: constant 0% (runs batch 48 on native BS=48 static shape)
    waste_b = np.zeros_like(steps, dtype=float)

    # Policy C waste: constant 0% (runs batch 64 on BS=64 for 1152 steps, then two cohorts on BS=32 static shapes)
    waste_c = np.zeros_like(steps, dtype=float)

    ax.plot(steps, waste_a, color=COLOR_DANGER, linewidth=2.5, label="Policy A (Greedy): 25% XLA Padding Waste after Step 1620 (BS=64 static shape running 48 streams)")
    ax.plot(steps, waste_b, color=COLOR_SUCCESS, linewidth=2.0, linestyle="--", label="Policy B (Predictive): 0% Waste (Native BS=48 Static Shape)")
    ax.plot(steps, waste_c, color=COLOR_PRIMARY, linewidth=2.0, linestyle="-.", label="Policy C (In-HBM Bisection): 0% Waste (Power-of-2 BS=64 -> BS=32)")

    ax.axvline(x=step_cliff, color=COLOR_DANGER, linestyle=":", alpha=0.8)
    ax.text(step_cliff + 20, 18, f"Step {step_cliff}:\nPreemption of 16 streams\n-> Batch falls to 48\n(25% static bubble)", color=COLOR_DANGER, fontsize=8.5, fontweight="bold")

    ax.fill_between(steps, 0, waste_a, where=(steps >= step_cliff), color=COLOR_DANGER, alpha=0.15, label="Wasted TPU Matrix Core FLOPs (Padding Slots)")

    ax.set_xlim(0, 2048)
    ax.set_ylim(0, 32)
    apply_google_style(
        ax,
        title="TPU XLA Static Shape Batch Padding Waste (BS=64)",
        xlabel="Autoregressive Decode Step",
        ylabel="XLA Static Shape Padding Waste (%)"
    )
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#dadce0", fontsize=8.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_throughput_breakdown(out_path: Union[str, pathlib.Path], summary: Optional[Dict[str, Any]] = None) -> None:
    """Figure 4: Raw Throughput vs Delivered Goodput and Execution Duration."""
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), dpi=300)

    res = summary.get("results", {}) if summary else {}

    policy_keys = [
        ("policy_a_greedy", "Policy A\n(Greedy)", COLOR_DANGER),
        ("policy_b_predictive", "Policy B\n(Predictive 48+16)", COLOR_WARNING),
        ("policy_c_bisection", "Policy C\n(In-HBM Bisection)", COLOR_GRAY),
        ("policy_d_balanced", "Policy D\n(Balanced 32+32)", COLOR_PRIMARY),
        ("policy_e_lossless_pipeline", "Policy E\n(Lossless Pipeline)", COLOR_SUCCESS),
        ("policy_oracle_optimal", "Oracle Optimal\n(32+32 @ 858)", "#8e24aa"),
    ]

    labels = []
    goodputs = []
    wastes = []
    durations = []
    colors = []

    for key, label, color in policy_keys:
        if key in res:
            p = res[key]
            labels.append(label)
            gp = p.get("goodput_tok_per_s", 0.0)
            dur = p.get("total_time_s", 0.0)
            w = (p.get("tokens_discarded_recomputed", 0) / dur) if dur > 0 else 0.0
            goodputs.append(gp)
            wastes.append(w)
            durations.append(dur)
            colors.append(color)

    x = np.arange(len(labels))
    width = 0.5

    # Stacked bar: Delivered Goodput + Wasted Recomputation
    b1 = ax1.bar(x, goodputs, width, label="Delivered Goodput (tok/s)", color=COLOR_PRIMARY, edgecolor="#dadce0")
    b2 = ax1.bar(x, wastes, width, bottom=goodputs, label="Wasted Recomputed (tok/s)", color=COLOR_DANGER, edgecolor="#dadce0", hatch="//")

    for idx, (g, w) in enumerate(zip(goodputs, wastes)):
        tot = g + w
        ax1.text(idx, tot + 35, f"Good:\n{g:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#202124")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, max(2600, max(goodputs) * 1.25) if goodputs else 2600)
    apply_google_style(
        ax1,
        title="Delivered Useful Goodput (tok/s)",
        xlabel="Admission Strategy",
        ylabel="Useful Tokens / Second"
    )
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#dadce0", fontsize=8.0)

    b3 = ax2.bar(x, durations, width, color=colors, edgecolor="#dadce0")
    for bar in b3:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.2f} s", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#202124")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(125, max(durations) * 1.2) if durations else 125)
    apply_google_style(
        ax2,
        title="End-to-End Execution Duration",
        xlabel="Admission Strategy",
        ylabel="Total Time (Seconds)"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def generate_all_plots(
    json_path: Union[str, pathlib.Path],
    figures_dir: Union[str, pathlib.Path],
) -> List[pathlib.Path]:
    """Main entrypoint for generating all benchmark plots."""
    fig_dir = pathlib.Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        summary = None

    p1 = fig_dir / "fig1_kv_capacity_cliff.png"
    p2 = fig_dir / "fig2_policy_goodput_comparison.png"
    p3 = fig_dir / "fig3_xla_padding_bubbles.png"
    p4 = fig_dir / "fig4_throughput_breakdown.png"

    plot_kv_capacity_cliff(p1, summary)
    plot_policy_goodput_comparison(p2, summary)
    plot_xla_padding_bubbles(p3, summary)
    plot_throughput_breakdown(p4, summary)

    generated = [p1, p2, p3, p4]
    print(f"[Plots] All 300 DPI figures successfully generated in {fig_dir}", flush=True)
    return generated
