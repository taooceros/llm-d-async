# Aggressive Hardware Oversubscription & Continuous Admission Protocol
## Systematic Review & Iteration Framework for Cloud TPU v5e (TP=8)

### 1. Hardware Architecture & System Constraints
- **Accelerator Architecture**: Cloud TPU v5e (8 chips, 2x4 ICI torus/mesh, TP=8) across physical hosts `192.168.102.18` and `192.168.102.19`.
- **Target LLM Weights**: Gemma-4 31B BF16 (61.4 GB model weights hot in TPU HBM).
- **Physical KV Cache Capacity**: $C_{\text{HBM}} = 120,000$ tokens pool ($\approx 46.8\text{ GB}$ physical headroom).
- **Static XLA Compilation Buckets**:
  $$\mathcal{B} = \{16, 32, 64, 128, 256\}$$
  - Bucket BS=16: Step latency $L(16) \approx 24.14\text{ ms}$
  - Bucket BS=32: Step latency $L(32) \approx 25.84\text{ ms}$
  - Bucket BS=64: Step latency $L(64) \approx 28.88\text{ ms}$
  - Bucket BS=128: Step latency $L(128) \approx 35.00\text{ ms}$
  - Bucket BS=256: Step latency $L(256) \approx 52.10\text{ ms}$

### 2. Inviolable Problem Rules & Constraints
1. **Batch Size Contract**: Ingestion occurs strictly in cohorts of $N=64$ requests ($B_0, B_1, \dots$).
2. **Preemption-Immunity Contract**:
   Requests may be drawn from a new batch $B_{k+1}$ **only if no request in the current batch $B_k$ can be preempted**.
   $$\forall t: \quad M(t) = \sum_{r \in \text{Active}(t)} \left( \text{prompt\_len}_r + \text{tokens\_decoded}_r(t) \right) \le C_{\text{HBM}} = 120,000$$
   Reorder requests only within an individual batch; earlier batches strictly take priority across batch boundaries to prevent starvation.
3. **No Weight Reloading**: Hot persistent worker actor must retain weights across runs with policy switch overhead $\le 0.40\text{ s}$.
4. **Zero Cache Pollution**: Prefix caching disabled (`enable_prefix_caching=False`); unique conversational sequences per cohort.

### 3. Aggressive Oversubscription Objectives
1. **Systolic MXU Density Maximization**:
   Rather than idling compute at $C=64$ (BS=64 bucket, $28.88\text{ ms}$ per step), aggressively oversubscribe active concurrency into the BS=128 XLA bucket:
   $$C \in \{88, 96, 104, 112, 128\}$$
   Amortize the $\sim 22\text{ ms}$ model weight memory bandwidth over up to 128 concurrent decoding streams.
2. **Continuous Inter-Batch Slack-Packed Admission**:
   - Sort each 64-request cohort using Continuous Slack-Packed LPT priority:
     $$\text{key}(r) = \left( \text{start\_cohort}, -d_r + w \cdot p_r, -d_r, p_r \right), \quad w \approx 0.006$$
   - Overlap the arrival of cohort $B_{k+1}$ into the draining tail of cohort $B_k$, holding concurrency pegged at target capacity $C$ without creating tail draining bubbles.
3. **Dynamic Knapsack Memory Guard**:
   At step $t$, admit request $r$ from $B_{k+1}$ into open slot if and only if:
   $$M_{\text{peak-future}}(t) = \max_{\tau \ge t} \sum_{i \in \text{Active}(\tau) \cup \{r\}} \text{Demand}(i, \tau) \le 120,000$$

### 4. Review Protocol for Evaluating Models (e.g. conversational workload Pro)
After each iteration, evaluate and answer:
1. **Mathematical Soundness**:
   Does the proposed oversubscription level $C$ guarantee zero preemptions for all worst-case prompt/decode length distributions?
2. **XLA Hardware Alignment**:
   Does the concurrency match a hardware-efficient static bucket, and what is the padding waste:
   $$\text{Waste}(C) = \frac{\lceil C \rceil_{\mathcal{B}} - C}{\lceil C \rceil_{\mathcal{B}}}$$
3. **Goodput vs. Latency Tradeoff**:
   Compare token delivery throughput ($\text{tok/s}$) against per-token decode latency ($L(C) / C$).
4. **Next Concrete Revision**:
   Provide the exact code modification in `cluster_runner.py`, `admission_control/algorithms/policies.py`, or `admission_control/algorithms/solver.py` to achieve higher goodput safely.
