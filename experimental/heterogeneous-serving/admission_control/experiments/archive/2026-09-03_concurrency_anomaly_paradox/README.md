# Archive: September 3, 2026 Scheduling Concurrency Anomaly & Belady/Graham Paradox

**Date**: 2026-09-03  
**Hardware Target**: Google Cloud TPU v5e (TP=8, KV Capacity: 120,000 tokens, Power-of-2 XLA Buckets: 16, 32, 64, 128, 256)  
**Workload**: conversational workload Scaled 256-Request Workload (`sharegpt_sampled_500.json`)  
**Topic**: The Belady/Graham Scheduling Paradox (Why Concurrency Cap $C=103$ Outperformed $C=128$)  

---

## 1. Executive Summary

During automated policy optimization on September 3, 2026, an apparent anomaly was observed:
- A tighter concurrency cap ($C=103$) achieved **2,557.0 tok/s** (a **1.3043× speedup** over greedy baseline), while allocating a larger concurrency cap ($C=128$, matching the maximum hardware bucket size) degraded performance.
- Through rigorous analytical modeling, this was identified as an instance of the classical **Belady/Graham Scheduling Anomaly** interacting with discrete TPU XLA compilation buckets:
  1. Setting $C=128$ allows requests with long decodes to straggle, locking the executor in the $BS=128$ bucket while actual active streams drop to 65–70, causing up to **48% XLA padding bubble waste**.
  2. Setting $C=103$ dynamically bounds step-level memory consumption, forcing stream completions to cluster tightly and enabling the executor to drop down to the $BS=64$ bucket faster, drastically reducing average step time and eliminating idle tensor pad overhead.

---

## 2. Archived Artifacts

- `scheduling_concurrency_anomaly_report.md`: Complete theoretical and empirical whitepaper with mathematical proofs, step latency equations, and memory trajectory formulas.
- `scheduling_concurrency_anomaly_report.html`: Standalone, publication-quality HTML report with embedded 300 DPI figures ready for Google Docs paste.
- `generate_concurrency_report_plots.py`: Historical Matplotlib script used to generate the 300 DPI figures.
- `AGGRESSIVE_OVERSUBSCRIPTION_PROMPT.md`: Experimental design notes exploring physical hardware oversubscription limits.
- `figures/`:
  - `concurrency_goodput_curve.png`: Goodput vs. concurrency ceiling curve showing the peak at $C=103$.
  - `memory_trajectory_comparison.png`: KV cache token memory trajectory across steps comparing $C=103$ and $C=128$.
  - `bucket_step_breakdown.png`: Proportion of time spent in $BS=64$ vs $BS=128$ buckets.
  - `plot_data.json`: Raw step telemetry and simulation trace data.
