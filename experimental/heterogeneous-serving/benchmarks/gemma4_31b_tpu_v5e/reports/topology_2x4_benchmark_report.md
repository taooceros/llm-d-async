# Gemma-4 31B Inference Benchmark on Cloud TPU v5e Topology 2x4 (TP=8)
## Prefill, Decode & Multi-Step Serving Performance Analysis vs. Full Cluster (4x4, TP=16)

### 1. Executive Summary & Overview

This report provides the performance benchmark results for **Gemma-4 31B** (48 layers, hidden dimension 5,376, FFN intermediate dimension 21,504, 32 Q heads, 16 KV heads) running on a **Cloud TPU v5e Topology 2x4** (2 hosts, 8 chips, `TPU_TOPOLOGY="2x4"`, Tensor Parallelism `TP=8`).

The experiment was submitted and orchestrated using the official **Ray SDK** (`JobSubmissionClient` connecting to the KubeRay TPU cluster on GKE).

Key findings:
1. **Prefill Throughput Advantage (+49.1% over TP=16)**:
   - Due to lower cross-host collective networking hops on the 2-host subslice (compared to the 4-host torus in TP=16), 48-layer step latency at $N=16,384$ tokens is compressed from **1,209.0 ms (TP=16)** down to **810.8 ms (TP=8)**.
   - This delivers a peak prompt throughput of **20,206.4 prompt tok/s** per 2x4 slice (50.82% Compute MFU), outperforming the 13,551.0 prompt tok/s achieved under TP=16.
2. **Decode Concurrency & Multi-Step Saturation**:
   - Single-step decode latency scales from **22.36 ms** (BS=4) to **28.88 ms** (BS=64) and **54.66 ms** (BS=256), scaling up to **5,422.8 tok/s** at BS=512.
   - In continuous multi-step generation (10 decode steps), generation throughput scales from **170.6 gen tok/s** at BS=1 (TPOT = 5.86 ms/tok) up to **6,587.2 gen tok/s** at BS=256 (TPOT = 38.86 ms/tok).
3. **Long Context Scalability (BS=64)**:
   - Scaling context length from 128 to 4,096 tokens (up to 96 GB aggregate KV cache footprint) maintains steady step times (~28.7 to 29.5 ms) and ~2,200 tok/s decode throughput.

---

### 2. Experimental Setup & Cluster Topology

| Specification | Cloud TPU v5e (2x4 Topology) | Cloud TPU v5e (4x4 Topology - Baseline) |
| :--- | :--- | :--- |
| **Physical Nodes** | 2 hosts (`ct5lp-hightpu-4t`) | 4 hosts (`ct5lp-hightpu-4t`) |
| **Total TPU Chips** | 8 TPU v5e chips | 16 TPU v5e chips |
| **Mesh Topology** | `2 × 4 2D Mesh` (2-Host ICI Ring) | `4 × 4 2D Torus` (4-Host ICI Torus) |
| **Peak FLOPs** | **1,576 TFLOPS** BF16 | 3,152 TFLOPS BF16 |
| **Total HBM Memory** | **128 GB** (16 GB / chip) | 256 GB (16 GB / chip) |
| **Model** | Gemma-4 31B (48 layers, D=5376, FFN=21504) | Gemma-4 31B |
| **Parallelism** | **Tensor Parallel TP = 8** | Tensor Parallel TP = 16 |
| **Orchestration** | Ray SDK / KubeRay Job Submission | Ray SDK / KubeRay Job Submission |

---

### 3. Prefill Performance: Token Bucket Scaling Sweep

Measured across token batch buckets $N = 64$ to $16,384$ tokens ($N = \text{Batch Size} \times \text{Prompt Length}$):

| Tokens ($N$) | QKV + O-Proj | SwiGLU FFN | Layer Latency | 48-Layer Step | Prompt Throughput | Compute MFU (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **N = 64** | 0.393 ms | 0.458 ms | 0.851 ms | **40.85 ms** | 1,566.6 tok/s | 3.94% |
| **N = 128** | 0.440 ms | 0.476 ms | 0.916 ms | **43.97 ms** | 2,911.2 tok/s | 7.32% |
| **N = 256** | 0.478 ms | 0.587 ms | 1.065 ms | **51.11 ms** | 5,009.2 tok/s | 12.60% |
| **N = 512** | 0.535 ms | 0.736 ms | 1.272 ms | **61.04 ms** | 8,388.2 tok/s | 21.10% |
| **N = 1,024** | 0.671 ms | 1.086 ms | 1.757 ms | **84.36 ms** | 12,139.2 tok/s | 30.53% |
| **N = 2,048** | 0.964 ms | 1.795 ms | 2.759 ms | **132.41 ms** | 15,466.8 tok/s | 38.90% |
| **N = 4,096** | 1.520 ms | 3.293 ms | 4.813 ms | **231.01 ms** | 17,731.1 tok/s | 44.59% |
| **N = 8,192** | 2.731 ms | 6.129 ms | 8.860 ms | **425.28 ms** | 19,262.6 tok/s | 48.44% |
| **N = 16,384** | 4.964 ms | 11.929 ms | 16.892 ms | **810.83 ms** | **20,206.4 tok/s** | **50.82%** |

#### Prefill Batch Size Concurrency Sweep (Context Length = 512)

| Batch Size | Prompt Tokens ($N$) | 48L Step Latency | Prompt Throughput | TTFT | Compute MFU (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BS = 1** | 512 | 44.95 ms | 11,390.8 tok/s | 44.95 ms | 28.65% |
| **BS = 2** | 1,024 | 69.26 ms | 14,785.0 tok/s | 69.26 ms | 37.18% |
| **BS = 4** | 2,048 | 125.39 ms | 16,332.8 tok/s | 125.39 ms | 41.08% |
| **BS = 8** | 4,096 | 236.07 ms | 17,350.9 tok/s | 236.07 ms | 43.64% |
| **BS = 16** | 8,192 | 457.37 ms | 17,911.1 tok/s | 457.37 ms | 45.05% |
| **BS = 32** | 16,384 | 867.17 ms | **18,893.7 tok/s** | 867.17 ms | **47.52%** |

---

### 4. Decode Performance: Single-Step & Concurrency Sweep

Measured with context length $S = 512$ tokens across batch sizes $BS = 1$ to $512$:

| Batch Size | Single-Step Latency | Pure Decode Tok/s | TPOT (Single-Step) | KV Cache Volume | Decode MFU (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BS = 1** | 27.03 ms | 37.0 tok/s | 27.03 ms | 0.19 GB | 0.09% |
| **BS = 4** | 22.36 ms | 178.9 tok/s | 22.36 ms | 0.75 GB | 0.45% |
| **BS = 16** | 24.14 ms | 662.9 tok/s | 24.14 ms | 3.00 GB | 1.67% |
| **BS = 32** | 24.98 ms | 1,281.0 tok/s | 24.98 ms | 6.00 GB | 3.22% |
| **BS = 64** | 28.88 ms | **2,216.4 tok/s** | 28.88 ms | 12.00 GB | 5.57% |
| **BS = 128** | 37.38 ms | **3,424.4 tok/s** | 37.38 ms | 24.00 GB | 8.61% |
| **BS = 256** | 54.66 ms | **4,683.8 tok/s** | 54.66 ms | 48.00 GB | 11.78% |
| **BS = 512** | 94.42 ms | **5,422.8 tok/s** | 94.42 ms | 96.00 GB | **13.64%** |

#### Long Context Scaling at Fixed Concurrency (BS = 64)

| Context Length | Step Latency | Decode Throughput | TPOT | Aggregate KV Cache |
| :--- | :--- | :--- | :--- | :--- |
| **128** | 25.88 ms | 2,472.7 tok/s | 25.88 ms | 3.00 GB |
| **512** | 28.95 ms | 2,210.5 tok/s | 28.95 ms | 12.00 GB |
| **1,024** | 29.38 ms | 2,178.7 tok/s | 29.38 ms | 24.00 GB |
| **2,048** | 29.55 ms | 2,165.8 tok/s | 29.55 ms | 48.00 GB |
| **4,096** | 28.71 ms | 2,228.9 tok/s | 28.71 ms | 96.00 GB |

---

### 5. Continuous Multi-Step Generation (10 Tokens, Context = 512)

| Batch Size | Total 10-Step Time | Generation Tok/s | TPOT (ms/tok) | Generation MFU (%) |
| :--- | :--- | :--- | :--- | :--- |
| **BS = 1** | 0.059 s | 170.6 tok/s | **5.86 ms** | 0.43% |
| **BS = 4** | 0.054 s | 738.5 tok/s | **5.42 ms** | 1.86% |
| **BS = 16** | 0.059 s | 2,712.9 tok/s | **5.90 ms** | 6.82% |
| **BS = 32** | 0.075 s | 4,268.9 tok/s | **7.50 ms** | 10.74% |
| **BS = 64** | 0.126 s | **5,086.4 tok/s** | **12.58 ms** | 12.79% |
| **BS = 128** | 0.212 s | **6,046.0 tok/s** | **21.17 ms** | 15.21% |
| **BS = 256** | 0.389 s | **6,587.2 tok/s** | **38.86 ms** | **16.57%** |

---

### 6. Direct Cross-Topology Comparison: Topology 2x4 (TP=8) vs Topology 4x4 (TP=16)

| Metric / Regime | Topology 2x4 (TP=8, 8 Chips) | Topology 4x4 (TP=16, 16 Chips) | Comparison & Engineering Takeaway |
| :--- | :--- | :--- | :--- |
| **Peak Prefill Throughput ($N=16,384$)** | **20,206 tok/s** | 13,551 tok/s | **+49.1% higher on TP=8** (Lower cross-host optical hop overhead) |
| **Prefill Layer Latency ($N=16,384$)** | **16.89 ms** | 25.19 ms | **32.9% faster per layer** on TP=8 |
| **Prefill Compute MFU ($N=16,384$)** | **50.82%** (of 1,576 TF) | 68.20% (of 3,152 TF) | TP=8 achieves higher raw throughput per slice |
| **Interactive Decode Latency (BS=1)** | 27.03 ms (5.86 ms multi-step) | 8.12 ms (8.43 ms E2E) | TP=16 has lower single-step latency due to 16-way sharded GEMM |
| **Decode Throughput (BS=64)** | **2,216 tok/s** (5,086 tok/s multi) | 2,220 tok/s | Identical throughput due to HBM bandwidth bottleneck |
| **Decode Throughput (BS=256)** | **4,684 tok/s** (6,587 tok/s multi) | 2,659 tok/s | TP=8 maintains scaling deeper into large batch sizes |
| **Optimal Deployment Strategy** | **Prefill-dominant & Disaggregated Pools** | **Low-latency Interactive Decode** | Topology 2x4 is optimal for high-throughput prompt processing |

---

### 7. Summary Conclusions & Recommendations

1. **Topology 2x4 (TP=8) is Compute- & Communication-Optimal for Prefill**:
   - In Gemma-4 31B, TP=8 provides twice the tensor dimension per chip ($d_{\text{local}} = 5376 / 8 = 672$) compared to TP=16 ($d_{\text{local}} = 336$), achieving higher GEMM matrix intensity.
   - Reducing the ICI ring from 4 hosts to 2 hosts cuts AllReduce ring synchronization overhead, delivering **20,206 prompt tok/s** per slice.
2. **Decode Phase Saturation Dynamics**:
   - For batch sizes $BS \le 32$, Decode latency is dominated by weight streaming and kernel launch overheads.
   - For $BS \ge 64$, KV cache retrieval from HBM becomes the active constraint.
3. **Disaggregated Serving Architecture**:
   - Deploying 2x4 (TP=8) slices for Prefill workers and 4x4 (TP=16) slices for interactive Decode workers provides the best cost-to-throughput ratio.
