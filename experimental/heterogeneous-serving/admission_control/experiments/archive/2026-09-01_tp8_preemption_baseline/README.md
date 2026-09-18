# Archive: September 1, 2026 TP=8 Preemption Benchmark Baseline Runs

**Date**: 2026-09-01  
**Hardware Target**: Google Cloud TPU v5e (TP=8, 2×4 Mesh, KV Cache Capacity: 120,000 tokens, Block Size: 16)  
**Workload Traces**: conversational workload Empirical Traces (`sharegpt_sampled_500.json`, `sharegpt_sampled_500.json`)  
**Topic**: Establishing Baseline vs Predictive Gating & Clairvoyant Oracle Scheduling  

---

## 1. Overview & Context

On September 1, 2026, the first comprehensive physical cluster experiments were executed to quantify the impact of vLLM reactive preemption on TPU TP=8:
- **KV Cache Capacity**: At TP=8, 128 GB total HBM leaves only ~46.8 GB (~120,000 tokens) for KV cache headroom, compared to ~155 GB (~400,000 tokens) at TP=16.
- **Preemption Problem**: Under oversaturated 64-request batches, unconstrained greedy admission causes KV exhaustion mid-stream, triggering expensive recompute preemptions and severe goodput degradation.
- **Experiment Scope**: Comparing baseline reactive preemption (Policy A) against predictive admission gating (Policy B) and clairvoyant lookahead scheduling.

---

## 2. Archived Results

- `results_run1_long_output.json`:
  - **Workload**: `sharegpt_sampled_500.json` (64 requests, total demand 129,361 tokens > 120,000 capacity).
  - **Findings**: Greedy baseline suffered 16 preemptions and discarded 30,016 recomputed tokens, achieving 1,398.9 tok/s. Predictive gating eliminated all preemptions, delivering 1,531.9 tok/s (1.095x speedup).
- `results_run2_long_prefill.json`:
  - **Workload**: `sharegpt_sampled_500.json` (64 requests, total demand 120,981 tokens).
  - **Findings**: Evaluated prefill head-of-line blocking; confirmed that binned LPT priority eliminated prefill clumping and improved hardware step stability.
