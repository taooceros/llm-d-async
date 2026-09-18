#!/usr/bin/env python3
"""
Generate high-fidelity vector visual plots (SVG) from physical Cloud TPU v5e benchmark results:
1. docs/visuals/padding_cost_topology_analysis.svg:
   - Prefill Step Latency vs Sequence Bucket (2x4 vs 4x4)
   - Cluster Throughput vs Bucket Size
   - Boundary Jump Latency Penalties (Delta T_jump)
   - Padding Waste Density Sweep (S=1024)
2. docs/visuals/paged_attention_matrix_heatmap.svg:
   - 60-cell 2D Empirical Grid (B in [1..1024], L in [256..8192])
   - TPOT Latency Heatmap & Speedup Annotations
   - Cluster Throughput Heatmap & SMEM 1.0 MB Boundary Annotations
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
VISUALS_DIR = os.path.join(PROJECT_ROOT, "docs", "visuals")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(VISUALS_DIR, exist_ok=True)

plt.style.use("default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 1.2


def generate_padding_cost_plot():
    json_path = os.path.join(RESULTS_DIR, "padding_cost_topologies.json")
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} does not exist. Skipping padding cost plot.")
        return

    with open(json_path) as f:
        data = json.load(f)

    topo_2x4 = data["topologies"]["2x4"]
    topo_4x4 = data["topologies"]["4x4"]

    buckets = sorted([int(k) for k in topo_2x4["standard_buckets"].keys()])
    lat_2x4 = [topo_2x4["standard_buckets"][str(b)]["step_ms_p50"] for b in buckets]
    lat_4x4 = [topo_4x4["standard_buckets"][str(b)]["step_ms_p50"] for b in buckets]
    tput_2x4 = [topo_2x4["standard_buckets"][str(b)]["throughput_tok_s"] for b in buckets]
    tput_4x4 = [topo_4x4["standard_buckets"][str(b)]["throughput_tok_s"] for b in buckets]

    jumps = ["128_to_256", "256_to_512", "512_to_1024", "1024_to_2048", "2048_to_4096", "4096_to_8192"]
    jump_labels = ["128→256", "256→512", "512→1K", "1K→2K", "2K→4K", "4K→8K"]
    jump_delta_2x4 = [topo_2x4["jump_penalties"][j]["delta_ms"] for j in jumps]
    jump_delta_4x4 = [topo_4x4["jump_penalties"][j]["delta_ms"] for j in jumps]
    jump_pct_2x4 = [topo_2x4["jump_penalties"][j]["overhead_pct"] for j in jumps]
    jump_pct_4x4 = [topo_4x4["jump_penalties"][j]["overhead_pct"] for j in jumps]

    dens_2x4 = topo_2x4["density_sweep_1024"]
    dens_4x4 = topo_4x4["density_sweep_1024"]
    dens_ratios = [x["density_ratio"] * 100 for x in dens_2x4]
    dens_eff_2x4 = [x["effective_tput_tok_s"] for x in dens_2x4]
    dens_eff_4x4 = [x["effective_tput_tok_s"] for x in dens_4x4]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    c_2x4 = "#0284c7"  # Blue
    c_4x4 = "#059669"  # Emerald Green

    # ----------------- Panel A: Latency vs Bucket -----------------
    ax1.plot(buckets, lat_2x4, marker="o", markersize=7, linewidth=2.5, color=c_2x4, label="2x4 Topology (TP=8, 8 chips)")
    ax1.plot(buckets, lat_4x4, marker="s", markersize=7, linewidth=2.5, color=c_4x4, label="4x4 Topology (TP=16, 16 chips)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(buckets)
    ax1.set_xticklabels([str(b) for b in buckets], fontsize=10, fontweight="bold", color="#334155")
    ax1.set_title("A. Single-Step Prefill Latency vs. Sequence Length Bucket", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax1.set_xlabel("Sequence Length Bucket (Tokens)", fontsize=11, fontweight="bold", color="#334155")
    ax1.set_ylabel("Execution Latency (P50 ms)", fontsize=11, fontweight="bold", color="#334155")
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax1.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=10.5, loc="upper left")

    ax1.annotate(
        "Comm-Bound Floor\n(1.01x - 1.09x speedup)",
        xy=(256, 33),
        xytext=(140, 100),
        arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#475569",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1f5f9", edgecolor="#cbd5e1"),
    )

    ax1.annotate(
        "4096: Saves 55.1ms\n(1.29x Speedup)",
        xy=(4096, 188.2),
        xytext=(2200, 260),
        arrowprops=dict(arrowstyle="->", color=c_4x4, lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#047857",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecfdf5", edgecolor="#a7f3d0"),
    )

    ax1.annotate(
        "8192: Saves 97.0ms\n(362.2ms vs 459.3ms)",
        xy=(8192, 362.2),
        xytext=(4500, 430),
        arrowprops=dict(arrowstyle="->", color=c_4x4, lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#047857",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecfdf5", edgecolor="#a7f3d0"),
    )

    # ----------------- Panel B: Throughput vs Bucket -----------------
    ax2.plot(buckets, tput_2x4, marker="o", markersize=7, linewidth=2.5, color=c_2x4, label="2x4 Topology (TP=8)")
    ax2.plot(buckets, tput_4x4, marker="s", markersize=7, linewidth=2.5, color=c_4x4, label="4x4 Topology (TP=16)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(buckets)
    ax2.set_xticklabels([str(b) for b in buckets], fontsize=10, fontweight="bold", color="#334155")
    ax2.set_title("B. Physical Prefill Throughput vs. Bucket Size", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax2.set_xlabel("Sequence Length Bucket (Tokens)", fontsize=11, fontweight="bold", color="#334155")
    ax2.set_ylabel("Hardware Throughput (Tokens / s)", fontsize=11, fontweight="bold", color="#334155")
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax2.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=10.5, loc="upper left")

    ax2.annotate(
        "2x4 Saturates at 17.3k tok/s\n(Memory/Activation bound)",
        xy=(2048, 17305),
        xytext=(1200, 13000),
        arrowprops=dict(arrowstyle="->", color=c_2x4, lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#0369a1",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f9ff", edgecolor="#bae6fd"),
    )

    ax2.annotate(
        "4x4 Scales to 22.6k tok/s\n(+26.8% Throughput)",
        xy=(8192, 22616),
        xytext=(3500, 23500),
        arrowprops=dict(arrowstyle="->", color=c_4x4, lw=1.5),
        fontsize=9.5,
        fontweight="bold",
        color="#047857",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecfdf5", edgecolor="#a7f3d0"),
    )

    # ----------------- Panel C: Boundary Jump Penalty -----------------
    x = np.arange(len(jump_labels))
    width = 0.36

    ax3.bar(x - width / 2, jump_delta_2x4, width, label="2x4 Step Jump (ms)", color=c_2x4, edgecolor="#0369a1", alpha=0.9)
    ax3.bar(x + width / 2, jump_delta_4x4, width, label="4x4 Step Jump (ms)", color=c_4x4, edgecolor="#047857", alpha=0.9)

    ax3.set_title("C. Boundary Jump Penalty (Extra Latency per Boundary Cross)", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax3.set_xticks(x)
    ax3.set_xticklabels(jump_labels, fontsize=10, fontweight="bold", color="#334155")
    ax3.set_xlabel("Compilation Bucket Jump (Tokens: S_i → S_i+1)", fontsize=11, fontweight="bold", color="#334155")
    ax3.set_ylabel("Added Step Latency Penalty ΔT_jump (ms)", fontsize=11, fontweight="bold", color="#334155")
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y", color="#cbd5e1")
    ax3.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=10.5, loc="upper left")

    for i in range(len(jump_labels)):
        d2 = jump_delta_2x4[i]
        d4 = jump_delta_4x4[i]
        ax3.text(i - width / 2, d2 + 3, f"+{d2:.1f}ms\n({jump_pct_2x4[i]:.0f}%)", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#0369a1")
        ax3.text(i + width / 2, d4 + 3, f"+{d4:.1f}ms\n({jump_pct_4x4[i]:.0f}%)", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#047857")

    ax3.set_ylim(0, 270)

    # ----------------- Panel D: Padding Density Sweep -----------------
    x_d = np.arange(len(dens_ratios))
    w_d = 0.35

    ax4.bar(x_d - w_d / 2, dens_eff_2x4, w_d, label="2x4 Effective Throughput", color=c_2x4, edgecolor="#0369a1")
    ax4.bar(x_d + w_d / 2, dens_eff_4x4, w_d, label="4x4 Effective Throughput", color=c_4x4, edgecolor="#047857")

    ax4.plot(x_d, [15093.0] * len(x_d), linestyle=":", color="#94a3b8", linewidth=1.5, label="2x4 Physical Step Capacity")
    ax4.plot(x_d, [16283.2] * len(x_d), linestyle="--", color="#10b981", linewidth=1.5, label="4x4 Physical Step Capacity")

    ax4.set_title("D. Effective Throughput Degradation vs. Padding Waste (S=1024)", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax4.set_xticks(x_d)
    ax4.set_xticklabels([f"{int(r)}% Active\n({int(1024*r/100)} tok)" for r in dens_ratios], fontsize=10, fontweight="bold", color="#334155")
    ax4.set_xlabel("Active Prompt Token Density Ratio in 1024 Bucket", fontsize=11, fontweight="bold", color="#334155")
    ax4.set_ylabel("Effective Throughput (Active Tokens / s)", fontsize=11, fontweight="bold", color="#334155")
    ax4.grid(True, linestyle="--", alpha=0.5, axis="y", color="#cbd5e1")
    ax4.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9.5, loc="upper left")

    for i, (v2, v4) in enumerate(zip(dens_eff_2x4, dens_eff_4x4)):
        ax4.text(i - w_d / 2, v2 + 250, f"{v2/1000:.1f}k", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0369a1")
        ax4.text(i + w_d / 2, v4 + 250, f"{v4/1000:.1f}k", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#047857")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.suptitle(
        "TPU v5e Physical Benchmarks: Static Bucket Padding Overhead & Topology Scaling (Gemma-4 31B)",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
    )

    out_path = os.path.join(VISUALS_DIR, "padding_cost_topology_analysis.svg")
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Successfully generated: {out_path}")


def generate_paged_attention_heatmap():
    json_path = os.path.join(RESULTS_DIR, "paged_attention_matrix_empirical.json")
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} does not exist. Skipping heatmap plot.")
        return

    with open(json_path) as f:
        m_data = json.load(f)

    batches = m_data["metadata"]["grid_batch_sizes"]
    contexts = m_data["metadata"]["grid_context_lengths"]
    contexts_rev = sorted(contexts, reverse=True)

    lat_4x4_grid = np.zeros((len(contexts_rev), len(batches)))
    lat_2x4_grid = np.zeros((len(contexts_rev), len(batches)))
    tput_4x4_grid = np.zeros((len(contexts_rev), len(batches)))
    tput_2x4_grid = np.zeros((len(contexts_rev), len(batches)))
    mask_oom = np.zeros((len(contexts_rev), len(batches)), dtype=bool)

    for r_idx, c in enumerate(contexts_rev):
        for c_idx, b in enumerate(batches):
            cell_lat = m_data["comparison_matrix"]["latency_tpot_ms"][str(c)][str(b)]
            cell_tput = m_data["comparison_matrix"]["throughput_tok_s"][str(c)][str(b)]
            if cell_lat["4x4"] is None:
                mask_oom[r_idx, c_idx] = True
                lat_4x4_grid[r_idx, c_idx] = np.nan
                lat_2x4_grid[r_idx, c_idx] = np.nan
                tput_4x4_grid[r_idx, c_idx] = np.nan
                tput_2x4_grid[r_idx, c_idx] = np.nan
            else:
                lat_4x4_grid[r_idx, c_idx] = cell_lat["4x4"]
                lat_2x4_grid[r_idx, c_idx] = cell_lat["2x4"]
                tput_4x4_grid[r_idx, c_idx] = cell_tput["4x4"]
                tput_2x4_grid[r_idx, c_idx] = cell_tput["2x4"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    im1 = ax1.imshow(lat_4x4_grid, cmap="viridis_r", aspect="auto")
    ax1.set_title("A. Decode Latency TPOT on 4x4 Pod (ms / token)", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax1.set_xticks(range(len(batches)))
    ax1.set_xticklabels([f"B={b}" for b in batches], fontsize=9.5, fontweight="bold", color="#334155")
    ax1.set_yticks(range(len(contexts_rev)))
    ax1.set_yticklabels([f"L={c}" for c in contexts_rev], fontsize=9.5, fontweight="bold", color="#334155")
    ax1.set_xlabel("Concurrent Decode Batch Size (B)", fontsize=11, fontweight="bold", color="#334155")
    ax1.set_ylabel("Historical Context Length (L)", fontsize=11, fontweight="bold", color="#334155")

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Full-Model 48-Layer Step Latency (ms)", fontsize=10, fontweight="bold", color="#334155")

    for r in range(len(contexts_rev)):
        for c in range(len(batches)):
            if mask_oom[r, c]:
                ax1.text(c, r, "SMEM\nOOM", ha="center", va="center", fontsize=8, fontweight="bold", color="#dc2626")
            else:
                val_4x4 = lat_4x4_grid[r, c]
                val_2x4 = lat_2x4_grid[r, c]
                speedup = val_2x4 / val_4x4
                color = "white" if val_4x4 > 80 else "black"
                ax1.text(c, r, f"{val_4x4:.1f}ms\n({speedup:.2f}x)", ha="center", va="center", fontsize=7.5, color=color, fontweight="bold")

    im2 = ax2.imshow(tput_4x4_grid, cmap="plasma", aspect="auto")
    ax2.set_title("B. Cluster Decode Throughput on 4x4 Pod (Tokens / s)", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax2.set_xticks(range(len(batches)))
    ax2.set_xticklabels([f"B={b}" for b in batches], fontsize=9.5, fontweight="bold", color="#334155")
    ax2.set_yticks(range(len(contexts_rev)))
    ax2.set_yticklabels([f"L={c}" for c in contexts_rev], fontsize=9.5, fontweight="bold", color="#334155")
    ax2.set_xlabel("Concurrent Decode Batch Size (B)", fontsize=11, fontweight="bold", color="#334155")
    ax2.set_ylabel("Historical Context Length (L)", fontsize=11, fontweight="bold", color="#334155")

    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Aggregate Tokens / s", fontsize=10, fontweight="bold", color="#334155")

    for r in range(len(contexts_rev)):
        for c in range(len(batches)):
            if mask_oom[r, c]:
                ax2.text(c, r, "SMEM\nOOM", ha="center", va="center", fontsize=8, fontweight="bold", color="#dc2626")
            else:
                t_val = tput_4x4_grid[r, c]
                color = "white" if t_val < 7000 or t_val > 15000 else "black"
                text_val = f"{t_val/1000:.1f}k" if t_val >= 1000 else f"{int(t_val)}"
                ax2.text(c, r, text_val, ha="center", va="center", fontsize=8.5, color=color, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(
        "Gemma-4 31B on Cloud TPU v5e: 60-Cell Empirical PagedAttention Scaling Matrix & SMEM 1MB Cliff",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
    )

    out_matrix_path = os.path.join(VISUALS_DIR, "paged_attention_matrix_heatmap.svg")
    plt.savefig(out_matrix_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Successfully generated: {out_matrix_path}")


if __name__ == "__main__":
    generate_padding_cost_plot()
    generate_paged_attention_heatmap()
