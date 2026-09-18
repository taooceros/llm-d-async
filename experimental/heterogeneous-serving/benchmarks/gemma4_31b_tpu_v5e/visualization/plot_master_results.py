import os
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
VISUALS_DIR = os.path.join(BASE_DIR, "reports", "visuals")
os.makedirs(VISUALS_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
tp4_mfu = [7.08, 13.20, 21.20, 34.56, 46.65, 56.35, 61.41, 65.40, 68.16]
tp8_mfu = [3.94, 7.62, 13.35, 21.34, 30.53, 39.13, 43.72, 48.98, 50.95]
tp16_mfu = [1.68, 3.41, 6.15, 11.36, 16.43, 22.61, 26.30, 28.82, 30.67]

tp4_tput = [1407.8, 2623.7, 4214.0, 6870.6, 9274.2, 11203.6, 12209.4, 13001.4, 13551.3]
tp8_tput = [1565.7, 3029.1, 5306.3, 8484.5, 12139.8, 15559.7, 17383.1, 19476.2, 20260.6]
tp16_tput = [1339.4, 2711.5, 4888.8, 9036.0, 13068.4, 17984.4, 20915.8, 22918.3, 24393.6]

cluster_tput_tp4 = [x * 4 for x in tp4_tput]
cluster_tput_tp8 = [x * 2 for x in tp8_tput]
cluster_tput_tp16 = [x * 1 for x in tp16_tput]

# 2. E2E Serving Data (Short vs Long Decode)
e2e_bs = [1, 4, 8, 16, 32, 64, 128]
short_out_tput = [107.77, 429.21, 761.49, 1141.08, 1650.52, 2110.33, 2472.33]
long_out_tput = [118.68, 434.51, 769.94, 1235.62, 1764.74, 2219.80, 2536.57]
short_tpot = [9.28, 9.32, 10.51, 14.02, 19.39, 30.33, 51.77]
long_tpot = [8.43, 9.21, 10.39, 12.95, 18.13, 28.83, 50.46]

# ----------------- Figure 1: Kernel & Cluster Architecture Scaling -----------------
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
fig1.patch.set_facecolor('#ffffff')

colors = {'TP=4': '#0284c7', 'TP=8': '#10b981', 'TP=16': '#f43f5e'}

ax1.plot(batch_sizes, tp4_mfu, marker='o', linewidth=2.2, color=colors['TP=4'], label='Single-Host (TP=4)')
ax1.plot(batch_sizes, tp8_mfu, marker='s', linewidth=2.2, color=colors['TP=8'], label='Two-Host (TP=8)')
ax1.plot(batch_sizes, tp16_mfu, marker='^', linewidth=2.2, color=colors['TP=16'], label='Four-Host (TP=16)')
ax1.set_xscale('log', base=2)
ax1.set_title('A. Model FLOPs Utilization (MFU %)', fontsize=13, fontweight='bold', pad=10, color='#0f172a')
ax1.set_xlabel('Token Bucket / Batch Size', fontsize=11, color='#334155')
ax1.set_ylabel('MFU (%)', fontsize=11, color='#334155')
ax1.grid(True)
ax1.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
ax1.annotate(f"{tp4_mfu[-1]:.1f}%", (16384, tp4_mfu[-1]), textcoords="offset points", xytext=(-20, 8), fontweight='bold', color=colors['TP=4'])
ax1.annotate(f"{tp8_mfu[-1]:.1f}%", (16384, tp8_mfu[-1]), textcoords="offset points", xytext=(-20, 8), fontweight='bold', color=colors['TP=8'])
ax1.annotate(f"{tp16_mfu[-1]:.1f}%", (16384, tp16_mfu[-1]), textcoords="offset points", xytext=(-20, 8), fontweight='bold', color=colors['TP=16'])

ax2.plot(batch_sizes, tp4_tput, marker='o', linewidth=2.2, color=colors['TP=4'], label='TP=4 (4 chips)')
ax2.plot(batch_sizes, tp8_tput, marker='s', linewidth=2.2, color=colors['TP=8'], label='TP=8 (8 chips)')
ax2.plot(batch_sizes, tp16_tput, marker='^', linewidth=2.2, color=colors['TP=16'], label='TP=16 (16 chips)')
ax2.set_xscale('log', base=2)
ax2.set_title('B. Per-Instance Token Throughput (tokens/s)', fontsize=13, fontweight='bold', pad=10, color='#0f172a')
ax2.set_xlabel('Token Bucket / Batch Size', fontsize=11, color='#334155')
ax2.set_ylabel('Tokens / s', fontsize=11, color='#334155')
ax2.grid(True)
ax2.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

ax3.plot(batch_sizes, cluster_tput_tp4, marker='o', linewidth=2.4, color=colors['TP=4'], label='4x Replicas (DP=4, TP=4)')
ax3.plot(batch_sizes, cluster_tput_tp8, marker='s', linewidth=2.4, color=colors['TP=8'], label='2x Replicas (DP=2, TP=8)')
ax3.plot(batch_sizes, cluster_tput_tp16, marker='^', linewidth=2.4, color=colors['TP=16'], label='1x Replica (DP=1, TP=16)')
ax3.set_xscale('log', base=2)
ax3.set_title('C. Full Cluster (16x TPU v5e) Total Serving Capacity', fontsize=13, fontweight='bold', pad=10, color='#0f172a')
ax3.set_xlabel('Token Bucket / Batch Size', fontsize=11, color='#334155')
ax3.set_ylabel('Total Cluster Tokens / s', fontsize=11, color='#334155')
ax3.grid(True)
ax3.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
ax3.annotate(f"54.2k tok/s (2.22x)", (16384, cluster_tput_tp4[-1]), textcoords="offset points", xytext=(-65, -20), fontweight='bold', color=colors['TP=4'])
ax3.annotate(f"40.5k tok/s (1.66x)", (16384, cluster_tput_tp8[-1]), textcoords="offset points", xytext=(-65, -20), fontweight='bold', color=colors['TP=8'])
ax3.annotate(f"24.4k tok/s (1.00x)", (16384, cluster_tput_tp16[-1]), textcoords="offset points", xytext=(-65, 10), fontweight='bold', color=colors['TP=16'])

# Subplot 4: Memory Capacity vs Context Length
ctx_lens = [512, 1024, 2048, 4096, 8192]
max_bs = [896, 448, 224, 112, 56]
ax4.bar([str(c) for c in ctx_lens], max_bs, color='#6366f1', width=0.55, edgecolor='#4338ca')
ax4.set_title('D. TPU v5e-16 Maximum Concurrency Limit (HBM Capacity)', fontsize=13, fontweight='bold', pad=10, color='#0f172a')
ax4.set_xlabel('Context Sequence Length (tokens)', fontsize=11, color='#334155')
ax4.set_ylabel('Max Schedulable Requests (Batch Size)', fontsize=11, color='#334155')
ax4.grid(True, axis='y')
for i, v in enumerate(max_bs):
    ax4.text(i, v + 20, f"BS={v}", ha='center', fontweight='bold', color='#312e81', fontsize=10)

plt.savefig(os.path.join(VISUALS_DIR, 'tp_scaling_and_memory_analysis.svg'))
print("Successfully generated TP Scaling & Memory Analysis SVG.")

# ----------------- Figure 2: E2E Generation & Latency -----------------
fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
fig2.patch.set_facecolor('#ffffff')

ax21.plot(e2e_bs, short_out_tput, marker='o', linewidth=2.4, color='#0284c7', label='Short Decode (Out=128 tokens)')
ax21.plot(e2e_bs, long_out_tput, marker='s', linewidth=2.4, color='#10b981', label='Long Decode (Out=1024 tokens)')
ax21.set_xscale('log', base=2)
ax21.set_title('A. End-to-End Pure Generation Throughput', fontsize=13, fontweight='bold', pad=10, color='#0f172a')
ax21.set_xlabel('Concurrency / Batch Size (Prompts)', fontsize=11, color='#334155')
ax21.set_ylabel('Output Generation Tokens / s', fontsize=11, color='#334155')
ax21.grid(True)
ax21.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
ax21.annotate(f"{long_out_tput[-1]:.0f} tok/s", (128, long_out_tput[-1]), textcoords="offset points", xytext=(-40, 8), fontweight='bold', color='#10b981')

ax22.plot(e2e_bs, short_tpot, marker='o', linewidth=2.4, color='#0284c7', label='Short Decode (Out=128 tokens)')
ax22.plot(e2e_bs, long_tpot, marker='s', linewidth=2.4, color='#10b981', label='Long Decode (Out=1024 tokens)')
ax22.set_xscale('log', base=2)
ax22.set_title('B. Time Per Output Token (TPOT / Step Latency)', fontsize=13, fontweight='bold', pad=10, color='#0f172a')
ax22.set_xlabel('Concurrency / Batch Size (Prompts)', fontsize=11, color='#334155')
ax22.set_ylabel('Latency per Token (ms / token)', fontsize=11, color='#334155')
ax22.grid(True)
ax22.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
ax22.annotate(f"{long_tpot[-1]:.1f} ms", (128, long_tpot[-1]), textcoords="offset points", xytext=(-35, 8), fontweight='bold', color='#10b981')

plt.savefig(os.path.join(VISUALS_DIR, 'e2e_generation_and_tpot_analysis.svg'))
print("Successfully generated E2E Generation & Latency SVG.")
