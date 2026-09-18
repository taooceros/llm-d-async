# Historical Experiment Results & Research Archive

This directory serves as the unified, centralized archive for historical experiment results, research reports, raw traces, and diagnostic artifacts. Experiments are organized chronologically by the time of execution and categorized by research topic.

---

## Chronological Directory Structure

```
admission_control/experiments/archive/
├── README.md                                          # Master index of archived research (this file)
│
├── 2026-08_gemma4_31b_tpu_v5e_batching/               # August 2026: TPU Batching & Prefill-Decode Decoupling
│   ├── README.md                                      # Topic overview and documentation links
│   └── vllm_tpu_batching_analysis.ipynb               # Exploratory Jupyter analysis notebook
│
├── 2026-09-01_tp8_preemption_baseline/                # September 1, 2026: TP=8 Preemption Baseline Runs
│   ├── README.md                                      # Initial empirical benchmark summary
│   ├── results_run1_long_output.json                  # Long output stress run (64 reqs, 129k demand tokens)
│   └── results_run2_long_prefill.json                 # Long prefill run (64 reqs, 120k demand tokens)
│
└── 2026-09-03_concurrency_anomaly_paradox/            # September 3, 2026: Belady/Graham Concurrency Paradox
    ├── README.md                                      # Paradox summary ($C=103$ vs $C=128$)
    ├── scheduling_concurrency_anomaly_report.md       # Full technical report with proofs
    ├── scheduling_concurrency_anomaly_report.html     # Publication-grade HTML report with 300 DPI figures
    ├── generate_concurrency_report_plots.py           # Historical plot generator
    ├── AGGRESSIVE_OVERSUBSCRIPTION_PROMPT.md          # Hardware oversubscription notes
    └── figures/                                       # High-resolution figures and data
        ├── concurrency_goodput_curve.png              # Goodput vs concurrency limit curve
        ├── memory_trajectory_comparison.png           # KV cache token memory trajectory
        ├── bucket_step_breakdown.png                  # XLA batch bucket residency
        └── plot_data.json                             # Underlying numerical telemetry
```

---

## Topic Index & Milestones

| Timeline | Research Topic | Hardware | Key Discoveries & Deliverables |
| :--- | :--- | :--- | :--- |
| **2026-08-20 – 2026-08-26** | **vLLM Dynamic Batching & Prefill-Decode Decoupling** | Cloud TPU v5e (TP=8, 2×4 Mesh) | Established XLA bucket overheads ($16, 32, 64, 128, 256$), proved that disaggregating long prefills ($N \ge 8192$) from decode streams minimizes step jitter. Canonical reports located in `docs/architecture/` and benchmark suites in `benchmarks/gemma4_31b_tpu_v5e/`. |
| **2026-09-01** | **TP8 Preemption Baseline on Real Traces** | Cloud TPU v5e (TP=8, 120k tok KV Cap) | Demonstrated that greedy unconstrained admission causes catastrophic preemption cascades (16 evictions, 30,016 discarded recompute tokens). Proved predictive admission gating eliminates all preemptions, achieving 1.095× speedup. |
| **2026-09-03** | **Belady/Graham Scheduling Concurrency Paradox** | Cloud TPU v5e (TP=8, XLA Buckets) | Discovered and proved that concurrency cap $C=103$ achieves 2,557.0 tok/s (1.3043× speedup) while $C=128$ degrades throughput due to stragglers locking the engine in $BS=128$ with up to 48% idle padding waste. |

---

## Canonical Active Locations

- **Production Package**: `admission_control/`
  - Core models & hardware specifications: `admission_control/core/`
  - Scheduling policies & solvers: `admission_control/algorithms/`
  - Simulator & BlockManager: `admission_control/simulation/`
  - Production experiment runner, plotting & reporting: `admission_control/experiments/`
  - Distributed TPU cluster execution & placement: `admission_control/tpu/` and `admission_control/serving/`
- **Workload Traces**: `data/workloads/`
- **Architecture Documentation**: `docs/architecture/`
- **Visuals & Diagrams**: `docs/visuals/`
- **CLI Utilities & Launchers**: `scripts/` and `python3 -m admission_control.cli`
