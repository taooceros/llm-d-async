# Archive: August 2026 Gemma-4-31B TPU v5e Batching & Decoupled Prefill-Decode

**Date Range**: 2026-08-20 – 2026-08-26  
**Hardware Target**: Google Cloud TPU v5e (TP=8, 2×4 Megacore ICI Mesh, 128 GB HBM)  
**Model**: Gemma-4 31B BF16 (~61.4 GB model weights)  
**Primary Investigators**: OMP Autonomous Research & Performance Engineering  

---

## 1. Overview & Objectives

In late August 2026, initial benchmarking on Cloud TPU v5e evaluated:
1. **Dynamic Batching & XLA Compilation Overhead**: The relationship between precompiled XLA bucket sizes ($16, 32, 64, 128, 256$), compilation latency, and HBM memory headroom.
2. **Prefill-Decode Decoupling**: Separating compute-heavy prefill bursts ($N \ge 8192$) from memory-bandwidth-bound autoregressive decoding streams across contiguous TPU slices.
3. **2×4 Topology Interconnect**: Verification of inter-host chip-to-chip latency and all-reduce performance on a two-host TPU submesh.

---

## 2. Archived Artifacts in this Directory

- `vllm_tpu_batching_analysis.ipynb`: Interactive Jupyter exploration notebook examining batch bucket selection, step time variances, and padding token bubble overheads on TPU v5e.

---

## 3. Related Canonical Documentation & Data

- **Architecture Plans**:
  - `docs/architecture/vllm_tpu_batching_analysis.md`: Detailed report on batching behavior.
  - `docs/architecture/vllm_prefill_decode_decoupling_plan.md`: Design document for disaggregated serving.
  - `docs/architecture/2x4_topology_experiment_handoff.md`: Handoff documentation for 2×4 physical topology.
  - `docs/architecture/gemma4_31b_tpu_v5e_benchmark_report.md`: Benchmark performance report.
- **Visualizations**:
  - `docs/visuals/gemma4_31b_e2e_throughput_analysis.svg`: End-to-end throughput curves.
  - `docs/visuals/gemma4_31b_long_decode_analysis.svg`: Long-decode step-time stability.
  - `docs/visuals/gemma4_31b_tpu_v5e_scaling.svg`: Tensor-parallel scaling across TPU topologies.
  - `docs/visuals/tpu_v5e_gemma4_benchmark_analysis.svg`: Comprehensive memory and roofline charts.
- **Benchmark Suites & Standalone Scripts**:
  - `benchmarks/gemma4_31b_tpu_v5e/results/`: Raw JSON benchmark outputs (`topology_2x4_benchmark_results.json`, `vllm_decoupled_prefill_decode_results.json`).
  - `benchmarks/gemma4_31b_tpu_v5e/reports/`: Rendered HTML and Markdown reports.
