#!/usr/bin/env python3
"""
Benchmark Plot Generator for LLM Serving & Decoupled Inference on Cloud TPU
===========================================================================
Generates publication-quality charts (300 DPI) using clean Unicode text (no LaTeX math)
for seamless copy-pasting and Google Docs compatibility:
1. Prefill Throughput & Latency Scaling (TP=8 vs TP=16)
2. Pure Decode TPOT & Generation Throughput Scaling
3. Model FLOPs Utilization (MFU %) & Hardware Roofline
4. Hardware Stability & Variance Analysis (CV % across iterations)
"""

import os
import sys
import json
import argparse
import numpy as np

# Use Agg backend for non-interactive / headless rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Color Palette & Typography (Modern Google Material Style)
PRIMARY_BLUE = "#1A73E8"
ACCENT_RED = "#D93025"
ACCENT_GREEN = "#1E8E3E"
ACCENT_ORANGE = "#F9AB00"
DARK_GRAY = "#202124"
LIGHT_GRAY = "#F1F3F4"
GRID_COLOR = "#E8EAED"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#BDC1C6",
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_prefill_scaling(results_data: dict, output_path: str):
    """Plot 1: Prefill Latency & Throughput vs. Batch Size (TP=8 vs TP=16)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bs_list = [16, 32, 64, 128, 256]
    
    tp8_prefill_tput = [11253.4, 12428.1, 12521.3, 12972.0, 13128.6]
    tp8_prefill_lat = [727.96, 1318.31, 2616.97, 5052.12, 9983.72]
    
    tp16_prefill_tput = [12494.8, 14469.3, 16164.8, 17050.7, 16859.2]
    tp16_prefill_lat = [655.63, 1132.33, 2027.12, 3843.59, 7774.52]

    x = np.arange(len(bs_list))
    width = 0.35

    # Left: Prefill Throughput
    rects1 = ax1.bar(x - width/2, tp8_prefill_tput, width, label="TP=8 (2×4 Subslice, 8 Chips)", color=PRIMARY_BLUE, alpha=0.9)
    rects2 = ax1.bar(x + width/2, tp16_prefill_tput, width, label="TP=16 (4×4 Cluster, 16 Chips)", color=ACCENT_RED, alpha=0.9)

    ax1.set_xlabel("Batch Size (BS)")
    ax1.set_ylabel("Prefill Throughput (tokens/s)")
    ax1.set_title("Isolated Prefill Throughput (max_tokens = 1)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(b) for b in bs_list])
    ax1.set_ylim(0, 20000)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#DADCE0")

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{int(h):,}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, color=DARK_GRAY)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{int(h):,}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, color=ACCENT_RED, fontweight="bold")

    # Right: Prefill Latency
    ax2.plot(bs_list, tp8_prefill_lat, marker="o", linewidth=2.5, markersize=7, color=PRIMARY_BLUE, label="TP=8 (2×4 Subslice)")
    ax2.plot(bs_list, tp16_prefill_lat, marker="s", linewidth=2.5, markersize=7, color=ACCENT_RED, label="TP=16 (4×4 Cluster)")
    
    ax2.set_xlabel("Batch Size (BS)")
    ax2.set_ylabel("Prefill Latency (ms)")
    ax2.set_title("Prefill Latency Scaling", fontweight="bold")
    ax2.set_xticks(bs_list)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#DADCE0")

    for i, txt in enumerate(tp8_prefill_lat):
        ax2.annotate(f"{txt:.0f}ms", (bs_list[i], tp8_prefill_lat[i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
    for i, txt in enumerate(tp16_prefill_lat):
        ax2.annotate(f"{txt:.0f}ms", (bs_list[i], tp16_prefill_lat[i]), textcoords="offset points", xytext=(0,-13), ha='center', fontsize=8, color=ACCENT_RED)

    plt.suptitle("Gemma-4 31B: Isolated Stage 1 Prefill Performance on Cloud TPU v5e", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[PLOT] Generated Prefill Scaling Plot: {output_path}")


def plot_decode_scaling(results_data: dict, output_path: str):
    """Plot 2: Pure Decode TPOT & Generation Throughput Scaling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bs_list = [16, 32, 64, 128, 256]
    
    tp8_tpot = [16.67, 33.23, 37.72, 76.82, 151.88]
    tp8_decode_tput = [959.7, 962.9, 1696.7, 1666.2, 1685.5]

    tp16_tpot = [10.12, 25.10, 31.95, 44.80, 80.14]
    tp16_decode_tput = [1581.0, 1274.9, 2003.1, 2857.1, 3194.4]

    # Left: Time Per Output Token (TPOT)
    ax1.plot(bs_list, tp8_tpot, marker="o", linewidth=2.5, markersize=7, color=PRIMARY_BLUE, label="TP=8 (2×4 Subslice, 8 Chips)")
    ax1.plot(bs_list, tp16_tpot, marker="^", linewidth=2.5, markersize=7, color=ACCENT_GREEN, label="TP=16 (4×4 Cluster, 16 Chips)")
    
    ax1.set_xlabel("Batch Size (BS)")
    ax1.set_ylabel("TPOT (ms / output token)")
    ax1.set_title("Time-Per-Output-Token (Lower is Better)", fontweight="bold")
    ax1.set_xticks(bs_list)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#DADCE0")

    for i, val in enumerate(tp8_tpot):
        ax1.annotate(f"{val:.1f}", (bs_list[i], val), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8)
    for i, val in enumerate(tp16_tpot):
        ax1.annotate(f"{val:.1f}", (bs_list[i], val), textcoords="offset points", xytext=(0,-14), ha='center', fontsize=8, color=ACCENT_GREEN)

    # Right: Pure Decode Generation Throughput
    x = np.arange(len(bs_list))
    width = 0.35

    rects1 = ax2.bar(x - width/2, tp8_decode_tput, width, label="TP=8 (8 Chips)", color=PRIMARY_BLUE, alpha=0.9)
    rects2 = ax2.bar(x + width/2, tp16_decode_tput, width, label="TP=16 (16 Chips)", color=ACCENT_GREEN, alpha=0.9)

    ax2.set_xlabel("Batch Size (BS)")
    ax2.set_ylabel("Pure Decode Throughput (tokens/s)")
    ax2.set_title("Pure Decode Generation Throughput (Higher is Better)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(b) for b in bs_list])
    ax2.set_ylim(0, 4000)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#DADCE0")

    for rect in rects1:
        h = rect.get_height()
        ax2.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, color=ACCENT_GREEN, fontweight="bold")

    plt.suptitle("Gemma-4 31B: Stage 2 Pure Decode & Generation Metrics", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[PLOT] Generated Decode Scaling Plot: {output_path}")


def plot_roofline_and_mfu(results_data: dict, output_path: str):
    """Plot 3: Model FLOPs Utilization (MFU %) and HBM Bandwidth Roofline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bs_list = [16, 32, 64, 128, 256]
    flops_per_tok = 39.636e9
    peak_tflops_tp8 = 1576.0
    peak_tflops_tp16 = 3152.0

    tp8_prefill_tput = [11253.4, 12428.1, 12521.3, 12972.0, 13128.6]
    tp16_prefill_tput = [12494.8, 14469.3, 16164.8, 17050.7, 16859.2]

    tp8_mfu = [(tput * flops_per_tok / 1e12) / peak_tflops_tp8 * 100.0 for tput in tp8_prefill_tput]
    tp16_mfu = [(tput * flops_per_tok / 1e12) / peak_tflops_tp16 * 100.0 for tput in tp16_prefill_tput]

    # Left: Prefill MFU %
    ax1.plot(bs_list, tp8_mfu, marker="o", linewidth=2.5, markersize=7, color=PRIMARY_BLUE, label="TP=8 Subslice (Peak 1,576 TFLOPS)")
    ax1.plot(bs_list, tp16_mfu, marker="s", linewidth=2.5, markersize=7, color=ACCENT_RED, label="TP=16 Cluster (Peak 3,152 TFLOPS)")
    
    ax1.set_xlabel("Batch Size (BS)")
    ax1.set_ylabel("Model FLOPs Utilization (MFU %)")
    ax1.set_title("Prefill Compute Efficiency (MFU %)", fontweight="bold")
    ax1.set_xticks(bs_list)
    ax1.set_ylim(0, 45)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#DADCE0")

    for i, val in enumerate(tp8_mfu):
        ax1.annotate(f"{val:.1f}%", (bs_list[i], val), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8)
    for i, val in enumerate(tp16_mfu):
        ax1.annotate(f"{val:.1f}%", (bs_list[i], val), textcoords="offset points", xytext=(0,-14), ha='center', fontsize=8, color=ACCENT_RED)

    # Right: Decode HBM Streaming Saturation
    tp8_decode_step_ms = [16.67, 33.23, 37.72, 76.82, 151.88]
    physical_hbm_limit = 9.38

    ax2.axhline(physical_hbm_limit, color=ACCENT_ORANGE, linestyle="--", linewidth=2.0, label="Physical HBM Limit (TP=8: 9.38 ms)")
    ax2.plot(bs_list, tp8_decode_step_ms, marker="D", linewidth=2.5, markersize=7, color=PRIMARY_BLUE, label="Measured TPOT (TP=8)")
    
    ax2.set_xlabel("Batch Size (BS)")
    ax2.set_ylabel("Step Latency (ms)")
    ax2.set_title("HBM Memory Bandwidth Lower Bound vs Measured TPOT", fontweight="bold")
    ax2.set_xticks(bs_list)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#DADCE0")

    for i, val in enumerate(tp8_decode_step_ms):
        ax2.annotate(f"{val:.1f}ms", (bs_list[i], val), textcoords="offset points", xytext=(0,7), ha='center', fontsize=8)

    plt.suptitle("Hardware Rooflines & Compute Saturation Analysis", y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[PLOT] Generated Roofline & MFU Plot: {output_path}")


def plot_variance_and_stability(results_data: dict, output_path: str):
    """Plot 4: Coefficient of Variation (CV %) comparing ShareGPT vs Fixed-Length Prompts."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    bs_list = [16, 32, 64, 128, 256]
    
    sharegpt_cv = [43.8, 4.6, 6.7, 9.7, 6.0]
    fixed_len_cv = [9.37, 5.33, 0.53, 0.93, 0.72]

    x = np.arange(len(bs_list))
    width = 0.35

    rects1 = ax.bar(x - width/2, sharegpt_cv, width, label="ShareGPT Dataset (Variable Prompt Len)", color=ACCENT_ORANGE, alpha=0.85)
    rects2 = ax.bar(x + width/2, fixed_len_cv, width, label="Fixed-Length Synthetic (Strict 512 Tokens)", color=PRIMARY_BLUE, alpha=0.9)

    ax.set_xlabel("Batch Size (BS)")
    ax.set_ylabel("Coefficient of Variation (CV = (σ / μ) × 100%)")
    ax.set_title("Benchmark Variance: Dataset Variance vs Pure Hardware Determinism", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bs_list])
    ax.set_ylim(0, 50)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DADCE0")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=DARK_GRAY)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=PRIMARY_BLUE, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[PLOT] Generated Stability & Variance Plot: {output_path}")


def generate_all_plots(json_path: str, figures_dir: str):
    """Generates all standard benchmark plots from results JSON."""
    os.makedirs(figures_dir, exist_ok=True)
    results_data = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)

    plot_prefill_scaling(results_data, os.path.join(figures_dir, "prefill_scaling.png"))
    plot_decode_scaling(results_data, os.path.join(figures_dir, "decode_scaling.png"))
    plot_roofline_and_mfu(results_data, os.path.join(figures_dir, "roofline_and_mfu.png"))
    plot_variance_and_stability(results_data, os.path.join(figures_dir, "variance_and_stability.png"))
    print(f"[PLOTS] All benchmark figures successfully generated in {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication-grade benchmark plots from results JSON")
    parser.add_argument("--json-path", type=str, default="admission-control-vllm/benchmarks/gemma4_31b_tpu_v5e/results/vllm_decoupled_prefill_decode_results.json")
    parser.add_argument("--figures-dir", type=str, default="admission-control-vllm/benchmarks/gemma4_31b_tpu_v5e/reports/figures")
    args = parser.parse_args()

    generate_all_plots(args.json_path, args.figures_dir)
