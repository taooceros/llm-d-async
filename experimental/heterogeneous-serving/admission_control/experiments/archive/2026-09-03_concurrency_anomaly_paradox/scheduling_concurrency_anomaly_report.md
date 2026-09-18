# Cloud TPU v5e Scheduling Policy Analysis: The Concurrency Paradox (C=103 vs C=128)
**Accelerating vLLM Offline Inference under Static XLA Bucketing & HBM Capacity Constraints**

---

## 1. Executive Summary

In high-throughput LLM serving on Cloud TPU v5e (TP=8), conventional wisdom dictates that maximizing active concurrency up to the upper bound of the static XLA compilation bucket (BS=128) should yield the highest token goodput. 

However, empirical evaluation on the canonical conversational workload trace (256 requests across 4 cohorts of M=64 requests) reveals a **Belady / Graham Scheduling Paradox**:

> **A concurrency cap of C=103 strictly outperforms C=128, delivering higher goodput (2,557.0 tok/s vs 2,552.8 tok/s) and finishing the workload 6 steps faster (80.65 s vs 80.78 s) with zero preemptions and 99.97% KV cache memory utilization.**

This report documents the architectural root causes, memory trajectory dynamics, and straggler tail penalties that cause this paradox, accompanied by formal policy DSL specifications and visual architectural diagrams.

---

## 2. Hardware Architecture & System Constraints

| Parameter | Specification | Architectural Impact |
| :--- | :--- | :--- |
| **Accelerator** | Cloud TPU v5e (8 chips, 2×4 mesh, TP=8) | 1,576 TFLOPS BF16, 128 GB total HBM |
| **Model** | Gemma-4 31B BF16 | 61.4 GB static weights loaded across TPU HBM |
| **KV Cache Capacity** | 120,000 tokens | Shared dynamically across all active request streams |
| **Block Size** | 16 tokens/block | Paged KV allocation unit in physical HBM |
| **Batch Cohort Window** | M=64 requests | Batch boundaries strictly enforce starvation freedom |
| **Precedence Invariant** | Inter-batch precedence (B_k prior to B_k+1) | Requests in B_k+1 admitted only if B_k cannot preempt |
| **Static XLA Buckets** | BS in [16, 32, 64, 128] | Padded execution shape with discrete step latencies |

### Empirical TPU v5e Step Latency Profile
Because XLA compiles fixed computational graph shapes on TPU, step execution latency is non-linear and step-function quantized:

- **BS in [1..16]**: 24.1 ms/step (padded execution overhead)
- **BS in [17..32]**: 25.8 ms/step
- **BS in [33..64]**: 28.9 ms/step
- **BS in [65..128]**: 34.5 ms/step (primary high-concurrency decode bucket)

---

## 3. Concurrency Sweep Empirical Results

Evaluating candidate admission policies across concurrency limits C in [64..128] on the 256-request conversational workload workload yields the following empirical performance frontier:

### Concurrency & Goodput Comparison

| Policy Configuration | Cap C | Goodput (tok/s) | Speedup vs Greedy |
| :--- | :---: | :---: | :---: |
| **Greedy FCFS Baseline** | 64 | 1,960.5 | 1.0000× |
| **Knapsack-88 (LPT)** | 88 | 2,420.4 | 1.2346× |
| **Knapsack-96 (LPT)** | 96 | 2,512.9 | 1.2818× |
| **QuantizedKnapsack-103** | **103** | **2,557.0** | **1.3043×** |
| **Knapsack-112** | 112 | 2,550.5 | 1.3009× |
| **Knapsack-120** | 120 | 2,551.9 | 1.3017× |
| **Knapsack-128** | 128 | 2,552.8 | 1.3021× |

### Hardware Latency & Memory Breakdown

| Policy Configuration | Steps | Runtime (s) | Peak KV Mem % |
| :--- | :---: | :---: | :---: |
| **Greedy FCFS Baseline** | 3,718 | 105.19 s | 69.32% |
| **Knapsack-88 (LPT)** | 2,530 | 85.20 s | 96.45% |
| **Knapsack-96 (LPT)** | 2,435 | 82.06 s | 99.97% |
| **QuantizedKnapsack-103** | **2,394** | **80.65 s** | **99.97%** |
| **Knapsack-112** | 2,400 | 80.85 s | 99.97% |
| **Knapsack-120** | 2,400 | 80.81 s | 99.97% |
| **Knapsack-128** | 2,400 | 80.78 s | 99.97% |

![Cloud TPU v5e Goodput vs. Active Concurrency Cap](figures/concurrency_goodput_curve.png)

---

## 4. Root Cause Analysis: Why C=103 Beats C=128

Both C=103 and C=128 execute in the same hardware XLA bucket (BS=128, latency 34.5 ms/step). Yet C=128 is slower. A detailed breakdown of steps by bucket reveals the discrepancy:

### Step Count Comparison by XLA Bucket

| XLA Bucket | Latency/Step | C=103 Steps | C=128 Steps |
| :--- | :---: | :---: | :---: |
| **BS = 16** | 24.1 ms | 17 steps | 34 steps (+17) |
| **BS = 32** | 25.8 ms | 21 steps | 30 steps (+9) |
| **BS = 64** | 28.9 ms | 276 steps | 244 steps (-32) |
| **BS = 128** | 34.5 ms | 2,079 steps | 2,091 steps (+12) |
| **Total Workload** | — | **2,394 steps** | **2,400 steps (+6)** |

### Step Time Comparison by XLA Bucket

| XLA Bucket | C=103 Time (s) | C=128 Time (s) | Delta Time (s) |
| :--- | :---: | :---: | :---: |
| **BS = 16** | 0.41 s | 0.82 s | +0.41 s |
| **BS = 32** | 0.54 s | 0.78 s | +0.24 s |
| **BS = 64** | 7.97 s | 7.05 s | -0.92 s |
| **BS = 128** | 71.73 s | 72.14 s | +0.41 s |
| **Total Workload** | **80.65 s** | **80.78 s** | **+0.13 s** |

![Step Count by XLA Bucket Breakdown](figures/bucket_step_breakdown.png)

---

## 5. Memory Trajectory Dynamics & Straggler Delays

### Three Cascading Failure Modes in C=128

1. **Greedy Admission Burst (Step 900)**: At step 900, cohort B_1 was completing while cohort B_2 began admission. Under C=103, active streams remained at 103 with 105,376 tokens (87.8% memory). Under C=128, the engine admitted 122 concurrent streams, driving memory to 119,168 tokens (99.3%), within 832 tokens of the 120,000 capacity ceiling.
2. **Admission Throttling**: The piecewise trajectory guard halted admissions to protect preemption immunity, stalling subsequent critical-path requests.
3. **Critical Path Delays**:
   - **Request 170 (Batch 2)**: Prompt=888, Decode=655. Under C=103, admitted at step 1422 and finished at step 2076. Under C=128, admission was blocked until step 1437 (15 steps late), stretching Batch 2 completion to step 2091.
   - **Request 242 (Batch 3)**: Prompt=1068, Decode=945 (longest decode in workload). Under C=103, admitted at step 1446 and finished at step 2390. Under C=128, admitted at step 1455 (9 steps late), finishing at step 2399.
4. **Straggler Tail Penalty**: Under C=103, the final requests (Req 209, 242, 248) completed concurrently at step 2390, followed by Req 225 at step 2393. Under C=128, Request 242 was left running completely alone between steps 2394 and 2399, incurring 6 padded single-request steps (6 × 24.1 ms = 144.6 ms of unamortized TPU compute).

![KV Cache Memory Trajectory Comparison](figures/memory_trajectory_comparison.png)

---

## 6. Policy Architecture & DSL Specification

The optimal scheduling policy (`QuantizedKnapsack-103`) is expressed declaratively using the Composable Scheduling Policy DSL:

### Python DSL Specification

```python
from admission_control.algorithms.policies import (
    PolicyBuilder,
    Priority,
    Guard,
    BatchTransition,
    Victim,
)

# Declarative construction of the optimal QuantizedKnapsack-103 policy
policy = (
    PolicyBuilder("QuantizedKnapsack-103")
    .with_priority(Priority.slack_packed_lpt(weight=0.0010))
    .admit_if(Guard.concurrency_cap(103))
    .admit_if(Guard.preemption_immune(capacity=120_000, block_size=16))
    .draw_next_batch_if(
        BatchTransition.preemption_immune(
            capacity=120_000,
            target_concurrency=103,
            block_size=16,
        )
    )
    .with_victim_selector(Victim.least_progress())
    .build()
)
```

### Declarative YAML Specification (`policy_103.yaml`)

```yaml
policy_class: ComposablePolicy
name: QuantizedKnapsack-103
priority:
  type: slack_packed_lpt
  weight: 0.0010
guards:
  - type: concurrency_cap
    cap: 103
  - type: preemption_immune
    capacity: 120000
    block_size: 16
batch_transition:
  type: preemption_immune
  capacity: 120000
  target_concurrency: 103
  block_size: 16
victim:
  type: least_progress
```

### Verification via CLI

```bash
# 1. Inspect policy ASCII tree and components
python3 -m admission_control.cli policy --spec policy_103.yaml --describe

# 2. Analytically verify preemption immunity and starvation freedom
python3 -m admission_control.cli policy --spec policy_103.yaml --verify \
    --workload data/workloads/sharegpt_sampled_500.json

# 3. Export execution trace for Perfetto timeline profiling
python3 -m admission_control.cli policy --spec policy_103.yaml \
    --workload data/workloads/sharegpt_sampled_500.json \
    --trace trace_103.json
```

---

## 7. Key Takeaways & Production Recommendations

1. **Bucket Boundaries Differ From Optimal Concurrency**: When targeting an XLA compilation bucket (BS=128), setting concurrency directly to the upper bucket edge (C=128) maximizes KV memory pressure and induces admission throttling. Setting C=103 leaves sufficient memory headroom to admit critical-path requests without delay.
2. **Smooth Pacing Eliminates Straggler Tails**: A slightly tighter concurrency cap (C=103) acts as an implicit traffic smoother, ensuring long-decode requests enter early and complete concurrently with shorter requests, avoiding costly single-stream padded steps.
3. **Continuous Slack Packing (w=0.0010)**: Pairing LPT sorting with prompt slack weight (w=0.0010) allows short prompts to pair with long decodes within the colocated prefill token budget without disrupting overall makespan.
4. **Hardware Verification**: The policy achieves **2,557.0 tok/s (+30.43% speedup over Greedy FCFS)** with **zero preemptions** across all 256 requests and 99.97% memory utilization.
