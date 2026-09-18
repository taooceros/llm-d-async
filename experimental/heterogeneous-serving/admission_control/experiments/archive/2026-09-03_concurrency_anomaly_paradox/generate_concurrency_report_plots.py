#!/usr/bin/env python3
"""
Generates publication-quality, 300 DPI Google-styled figures for the
Cloud TPU v5e Scheduling Concurrency Paradox Report (C=103 vs C=128).
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__)) + "/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_PATH = os.path.join(FIGURES_DIR, "plot_data.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Google Material Colors
C_BLUE = "#1a73e8"
C_RED = "#ea4335"
C_GREEN = "#34a853"
C_YELLOW = "#fbbc04"
C_GREY = "#5f6368"
C_LIGHT_GREY = "#dadce0"

# -----------------------------------------------------------------------------
# Plot 1: Concurrency Cap vs Goodput
# -----------------------------------------------------------------------------
caps = [64, 88, 96, 103, 112, 120, 128]
goodput = [1960.5, 2420.4, 2512.9, 2557.0, 2550.5, 2551.9, 2552.8]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
ax.plot(caps, goodput, marker='o', color=C_BLUE, linewidth=2.5, markersize=8, label='Quantized Knapsack Goodput (tok/s)')
ax.axvline(x=103, color=C_RED, linestyle='--', linewidth=1.8, label='Optimal Cap C=103 (2,557.0 tok/s)')
ax.scatter([103], [2557.0], color=C_RED, s=150, zorder=5)

ax.annotate(
    "Peak: C=103\n2,557.0 tok/s (+30.43%)",
    xy=(103, 2557.0),
    xytext=(108, 2585),
    arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5),
    fontsize=10,
    fontweight="bold",
    color=C_RED,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fce8e6", edgecolor=C_RED, alpha=0.9),
)

ax.annotate(
    "C=128 Plateau\n2,552.8 tok/s (-4.2 tok/s)",
    xy=(128, 2552.8),
    xytext=(114, 2510),
    arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.2),
    fontsize=9,
    color=C_GREY,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1f3f4", edgecolor=C_LIGHT_GREY),
)

ax.set_title("Cloud TPU v5e Goodput vs. Active Concurrency Cap C", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Active Concurrency Cap C (Streams in BS=128 Bucket)", fontsize=11, labelpad=8)
ax.set_ylabel("Goodput (Delivered Tokens / Second)", fontsize=11, labelpad=8)
ax.set_xlim(60, 132)
ax.set_ylim(1900, 2630)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor=C_LIGHT_GREY, fontsize=10)

plt.tight_layout()
p1 = os.path.join(FIGURES_DIR, "concurrency_goodput_curve.png")
plt.savefig(p1)
plt.close()
print("Saved:", p1)

# -----------------------------------------------------------------------------
# Plot 2: KV Cache Memory Trajectory Comparison (C=103 vs C=128)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

steps_103 = data["steps_103"]
mem_103 = data["mem_103"]
steps_128 = data["steps_128"]
mem_128 = data["mem_128"]

ax.plot(steps_103, mem_103, color=C_BLUE, linewidth=1.8, label="C=103 (Smooth Pacing, finishes Step 2,394)")
ax.plot(steps_128, mem_128, color="#f2994a", linewidth=1.6, linestyle="-.", label="C=128 (Premature Saturation, finishes Step 2,400)")
ax.axhline(y=120000, color=C_RED, linestyle="--", linewidth=1.5, label="HBM Capacity Ceiling (120,000 tokens)")

# Annotate Step 900 Memory Spike
ax.annotate(
    "Step 900: C=128 reaches 119,168 tokens (99.3%)\nTriggers Preemption Guard Throttle",
    xy=(900, 119168),
    xytext=(950, 95000),
    arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5),
    fontsize=9.5,
    fontweight="bold",
    color=C_RED,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fce8e6", edgecolor=C_RED, alpha=0.9),
)

# Annotate Tail Finish
ax.annotate(
    "C=103 Finishes Step 2,394\n(6 steps earlier)",
    xy=(2394, 10000),
    xytext=(2050, 30000),
    arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.5),
    fontsize=9.5,
    fontweight="bold",
    color=C_BLUE,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f0fe", edgecolor=C_BLUE, alpha=0.9),
)

ax.set_title("KV Cache Memory Trajectory: C=103 vs. C=128 Over Time", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Execution Step Number", fontsize=11, labelpad=8)
ax.set_ylabel("KV Cache Memory (Tokens Allocated in HBM)", fontsize=11, labelpad=8)
ax.set_xlim(0, 2450)
ax.set_ylim(0, 130000)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor=C_LIGHT_GREY, fontsize=9.5)

plt.tight_layout()
p2 = os.path.join(FIGURES_DIR, "memory_trajectory_comparison.png")
plt.savefig(p2)
plt.close()
print("Saved:", p2)

# -----------------------------------------------------------------------------
# Plot 3: Step Count by XLA Bucket Breakdown
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

buckets = ["BS=16 (24.1ms)", "BS=32 (25.8ms)", "BS=64 (28.9ms)", "BS=128 (34.5ms)"]
x = np.arange(len(buckets))
width = 0.35

steps_103_b = [17, 21, 276, 2079]
steps_128_b = [34, 30, 244, 2091]

rects1 = ax.bar(x - width/2, steps_103_b, width, label="C=103 (Total: 2,394 steps)", color=C_BLUE, edgecolor=C_LIGHT_GREY)
rects2 = ax.bar(x + width/2, steps_128_b, width, label="C=128 (Total: 2,400 steps)", color="#f2994a", edgecolor=C_LIGHT_GREY)

# Add value labels on bars
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_title("Cloud TPU v5e Step Counts by Static XLA Compilation Bucket", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("XLA Static Compilation Shape (Padded Bucket)", fontsize=11, labelpad=8)
ax.set_ylabel("Number of Execution Steps", fontsize=11, labelpad=8)
ax.set_xticks(x)
ax.set_xticklabels(buckets, fontsize=10)
ax.set_ylim(0, 2300)
ax.grid(True, linestyle=":", alpha=0.6, axis="y")
ax.legend(frameon=True, facecolor="#ffffff", edgecolor=C_LIGHT_GREY, fontsize=10)

plt.tight_layout()
p3 = os.path.join(FIGURES_DIR, "bucket_step_breakdown.png")
plt.savefig(p3)
plt.close()
print("Saved:", p3)
