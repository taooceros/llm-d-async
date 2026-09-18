# Measured state for policy design (all TPU v5e, TP16, gemma-4-31b, real ShareGPT)

Artifacts: `/workspace/results/scheduling_tracks_20260914/*.json` (84 files).
Harness: `/workspace/benchmarks/run_measurement_audit.py` (~2022 lines).
Goal: `/workspace/SCHEDULING_HEADROOM_GOAL.md`.

## Metric

**Makespan** over a fixed request manifest (backlog >> HBM, all requests present at t=0).
Latency/TTFT are explicitly NOT optimized. Also report output tok/s.
Total decode tokens is FIXED by the manifest, identical across arms
(e.g. 721,595 tokens for manifest 5f30f9fa9f). So makespan differences come
**only** from throughput differences, never from doing less work.

## Machine model (measured, R^2=0.9998)

decode step time = `6.75 ms + 0.292 * b + 0.0514 * padded(b)` where b = scheduled requests,
`padded(b)` rounds up to `[8,16,32,64,128,256,512]`.

Implication: per-request marginal cost (0.292 ms) dominates at b>=90
(0.292*90 = 26.3 ms of a ~39.6 ms step). Fixed overhead 6.75 ms is ~17%.
Throughput saturates in b: measured 1525 tok/s @ b=26.6, 1833 @ 44.9,
1975 @ 62.6, 2042 @ 76.9. Strongly diminishing.

## KV geometry

566 usable blocks x 256 tok = **144,896 tokens** total KV.
`max_num_seqs=512` NEVER binds. Residency is **KV-bound**.
A request's KV grows as it decodes: blocks(i) = ceil((p_i + generated_i)/256).
So co-residency depends on the *integral* of KV occupancy over time.

## Stage shares (HLO tf_op, measured)

attention 81.46% (sliding-1024 53.57%, full 18.27%, out-proj 7.74%, QKV 1.29%),
FFN 14.49%, LM head 0.61%, sampling 0.09%.
Sliding window = 1024: past it the sliding component is constant per request;
full attention keeps scaling with context.

## Measured arms (makespan seconds; lower better). Same manifest within a block.

### manifest 43d11866f9 (B1, sigma=1.0, 10x, n=1763)
| arm | order | track | wall_s | preempt | mean batch |
|---|---|---|---:|---:|---:|
| PB_lpt_true_d | long_decode_first | C | **340.4** | 644 | **166.0** |
| NA_watermark | arrival + gate | O | 345.5 | 0 | 105.3 |
| PF_spt_true_d | short_decode_first | C | 346.1 | 174 | 106.0 |
| PA_trajectory | arrival | C | 349.6 | 0 | 100.6 |
| ND_spt_estimate | short_decode_first | O | 351.5 | 0 | 94.8 |
| NC_prompt_proxy | short_decode_first | O | 357.2 | 508 | 95.7 |
| default_greedy | arrival | C | 358.3 | 888 | 108.6 |

### manifest 1b348af184
PB_lpt_true_d **286.9** (481 pre, batch 227.2) | ND_spt_estimate 288.3 (0, 174.1) |
NA_watermark 289.1 (0, 137.2) | default 302.1 (616, 140.5) | PF_spt_true_d 303.6 (263, 130.1)

### manifest 56c83582dd
NA_watermark **292.5** (0, 170.8) | PB_lpt_true_d 294.4 (426, 226.1) |
ND_spt_estimate 298.8 (40, 166.7) | default 304.6 (626, 177.3) | PF_spt_true_d 305.2 (278, 173.3)

### manifest 8cced8211b
NA_watermark **376.1** (0, 92.0) | ND_spt_estimate 377.8 (91, 89.7) |
PF_spt_true_d 378.6 (192, 102.9) | PB_lpt_true_d 384.1 (748, 130.6) | default 397.1 (1120, 95.1)

### manifest 5f30f9fa9f (bucket sweep + cap128 twins, n=1763, 721,595 decode tok)
| arm | wall_s | tok/s | iters | batch | preempt |
|---|---:|---:|---:|---:|---:|
| cap128 (arrival) | **340.8** | 2117 | 8068 | 89.4 | 0 |
| cap128 + true_d SPT | 344.0 | 2097 | 8519 | 84.7 | 0 |
| NA_watermark | 345.6 | 2088 | 6790 | 106.3 | 0 |
| cap128 + estimate SPT | 348.8 | 2069 | 9177 | 78.6 | 0 |
| cap256 | 357.8 | 2017 | 6627 | 108.9 | 873 |
| uncapped | 361.1 | 1999 | 6571 | 109.8 | 865 |
| cap64 | 371.5 | 1942 | 13068 | 55.2 | 0 |

### realised padded(b) distribution, % of steps (authoritative runner padded_requests)
| cap | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 13.6 | 6.4 | **80.1** | 0 | 0 | 0 |
| 48 | 15.0 | 10.0 | 4.2 | **70.9** | 0 | 0 |
| 64 | 13.6 | 12.3 | 8.7 | **65.4** | 0 | 0 |
| 96 | 13.9 | 13.4 | 15.1 | 9.5 | **48.1** | 0 |
| 128 | 16.2 | 9.2 | 20.8 | 11.5 | **42.4** | 0 |
| 192 | 13.7 | 8.5 | 25.2 | 16.3 | 10.8 | **25.5** |
| 256 | 9.0 | 15.4 | 21.7 | 18.4 | 13.1 | **22.4** |

## THE OPEN LEAD

`PB_lpt_true_d` (long-decode-first) achieves **mean batch 175.3** averaged over manifests versus
`NA_watermark`'s 119.2 — 47% higher residency — and wins outright in 3 of 6 manifests.
But it takes **426-748 preemptions**, which eats the gain (mean -0.28% vs best Track O).

**LPT ordering has NEVER been combined with the zero-preemption admission gate
(watermark or cap128).** That cell is empty. Both mechanisms exist and are measured
separately; their combination is untested.

## Existing mechanisms in the harness

- `order_workload(workload, order)` :696 — `arrival`, `short_first`, `long_first`,
  `short_decode_first`, `long_decode_first`, `short_decode_then_short_prompt`, interleaved.
  Decode-keyed orders read `r['d_sort']`, never `r['d']`.
- `policy_view(request, track)` :734 — strips `d` for Track O. Track O policies
  MUST assert no oracle field at construction.
- Admission gates: `greedy` (no gate), `watermark(K)` (free-KV watermark),
  `cap(N)` (residency cap, checked at :1259 BEFORE admission_allows).
- `OnlineLengthEstimator` :749 — kNN on (prompt_tokens -> observed output_tokens)
  from COMPLETED requests only. At rho=0 it has no signal (Spearman -0.030);
  at rho=0.5 it works (Spearman 0.459, MAE 266).
- FCFS preemption victim = `self.running.pop()` (tail of list).
  Reordering `self.running` before `schedule()` selects the victim without subclassing.
  NEVER filter SchedulerOutput post-hoc: `schedule()` calls `_update_after_schedule`.
- Installed scheduler has `schedule(self, throttle_prefills: bool = False)` gated on
  `self.prefill_capacity_bound` — a NATIVE prefill brake, never used as a comparator.

## Workloads

`W-SMALL` (current): ShareGPT natural prompts; decode ~ capped lognormal median=256,
sigma=1.0, d_max=4096. Realised decode: median 253, mean 409, p90 909, p99 2621, CV 1.23.
Mean context per resident ~1080 tok. n=1763 (10x) or 3526 (20x).

`W-LARGE` (to be added): prompts filtered to >=512 tok; decode median=1024, sigma=1.0,
d_max=8192. Raises KV per request, cuts residency, widens spread of remaining work.

## Hard constraints

- Hardware only. No simulator, no analytical estimate. NEVER multiply a fitted coefficient
  by a count and report it as a measured time.
- Identical manifests across compared arms (`workload_sha256` equality asserted).
- Information-only twins: arms compared for oracle value differ ONLY in whether the sort key
  reads `d`. Same gate, same machinery.
- The TPU actor is SINGLE-THREADED and owned by the orchestrator. Design agents do NOT
  submit TPU jobs and do NOT edit `run_measurement_audit.py`.


## THE MAKESPAN IDENTITY — read this before designing anything

Verified against measured data to within 0.04 %: summing the per-step batch over a run
reproduces the manifest's total decode tokens exactly.

| arm | T (steps) | mean b | T x mean_b | D |
|---|---:|---:|---:|---:|
| cap128 | 8068 | 89.4 | 721,279 | 721,595 |
| NA_watermark | 6790 | 106.3 | 721,777 | 721,595 |
| cap64 | 13068 | 55.2 | 721,354 | 721,595 |
| uncapped | 6571 | 109.8 | 721,496 | 721,595 |

So **sum_t b_t = D is an invariant of the manifest**, and substituting the measured step model:

    makespan = 0.292*D  +  6.75*T  +  0.0514 * sum_t padded(b_t)
               ^^^^^^^^     ^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^
               IRREDUCIBLE  step count   padding exposure
               210.7 s      T = D / mean_b

**This is why the measured gains are only 4-6 %.** The per-request term is
0.292 ms x 721,595 tok = **210.7 s, which is 62 % of the best measured makespan (340.8 s)**
and no scheduler can touch it: every request must be attended to once per token it emits,
and D is fixed by the manifest. Only ~38 % of makespan is schedulable at all, and the two
schedulable terms **fight each other**: raising b cuts T but risks crossing a padding bucket.

A scheduler therefore has exactly two levers:
1. **Minimise T** by maximising mean batch. T = D / mean_b.
2. **Minimise sum_t padded(b_t)** by holding b just under a power-of-two boundary.

Planning targets from the model (NOT measurements - hardware validation required):

| pinned b | T | 6.75*T | padding term | projected total |
|---:|---:|---:|---:|---:|
| 64 | 11275 | 76.1 s | 37.1 s | 323.9 s |
| 89.4 (cap128 realised) | 8072 | 54.5 s | 53.1 s | 318.3 s |
| 106.3 (NA realised) | 6788 | 45.8 s | 44.7 s | 301.2 s |
| **128** | 5637 | 38.1 s | 37.1 s | **285.8 s** |
| 256 | 2819 | 19.0 s | 37.1 s | 266.8 s |

Best measured is 340.8 s at realised mean b = 89.4. **cap128 already pays for the 128 bucket
42.4 % of the time but only fills it to 89.4 on average.** Closing that gap is the opportunity,
and it is worth far more than any sort order.

## WHY THIS MAKES DECODE LENGTH VALUABLE — the offline planning frame

Makespan is **order-invariant given a fixed batch profile** (sum_t b_t = D regardless of order).
That is why SPT and LPT as pure sort keys were a wash: reordering does not change D, and only
perturbs mean_b as a side effect. Sorting is the wrong use of d_i.

The right use: **the manifest is fully known at t=0**, so Track C can compute an
**offline admission timetable** by forward-simulating KV occupancy. Knowing d_i means knowing
*exactly when each resident will finish and free its blocks*, so the planner can admit right up
to the KV ceiling without ever overflowing - i.e. **run at high b with zero preemptions**, which
is precisely the combination no measured arm has achieved (high-b arms preempt; zero-preemption
arms run at low b).

Track O cannot do this: without d_i it cannot predict release times, so it must leave a safety
margin (that margin IS the watermark) and therefore runs at lower b.

**That gap is the true value of decode-length knowledge, and it has never been measured.**

KV feasibility constraint the planner must respect at every step t:

    sum over residents i of ceil((p_i + generated_i(t)) / 256) <= 566 blocks

Note KV demand per request GROWS as it decodes, so feasibility is a trajectory property,
not a point property. A request admitted safely now can overflow later.

### Complexity budget (the user explicitly warned about this)

Do NOT propose exhaustive combinatorial search; this is a resource-constrained scheduling
problem and the exact version is NP-hard. Acceptable:
- Forward-simulated greedy with lookahead: O(T * b) per evaluation, T ~ 6k-13k, b ~ 100.
- Local search / annealing over the admission timetable with the forward simulator as the
  evaluator, with a hard wall-clock budget for planning.
- Water-filling / threshold rules derived analytically from the identity above.
Planning happens offline, once per manifest, before the run. Planning time is NOT part of the
measured makespan but MUST be reported.

**Using the cost model INSIDE a planner is legitimate.** Reporting the planner's projected
makespan as a result is NOT. Every headline number must be a measured hardware makespan.
