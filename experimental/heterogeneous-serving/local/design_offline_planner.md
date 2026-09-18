# Planner–engine co-design: executable discrete transitions

## Status and decision

The native-transition design below remains broader than the implemented experiment. A bounded fresh-admission controller now exists in `benchmarks/kv_admission.py`; native arithmetic/shadow checks passed, but its first matched bulk comparison regressed and exceeded its online CPU budget. It is not a throughput improvement or a release-credit certificate. Existing strict-calendar and native order-only modes remain separate reference implementations.

The current investigation is specified in `SCHEDULING_HEADROOM_GOAL.md`: reuse the measured bulk cap results and evaluate adaptive admission/concurrency beyond the best static policy. That goal supersedes the earlier W-LARGE cap64-only experiment prescription; this document supplies the shared native transition contract.

**Decision:** keep native asynchronous execution and co-design admission planning with its actual scheduling transitions. The optimization variable is an executable, state-conditioned admission action—not a start-time vector subsequently reduced to a permutation. Native allocation and bookkeeping remain authoritative. Start with admission only; do not simultaneously change victim selection, resident traversal, model execution or native compilation/padding rules.

**Scope: bulk inference with deep backlogs under sustained memory pressure, not interactive latency.** Optimize useful-output throughput and fixed-workload completion time. Admission deferral is justified only when its reduction in recomputation or improvement in subsequent execution outweighs the useful throughput it sacrifices; latency, TTFT and fairness are not objectives. Prefer mechanisms that help while backlog is plentiful, rather than optimizing a shallow workload’s final drain. Report backlog-present behavior separately as a diagnostic and retain full-workload makespan as the outcome. Historical n600 results do not establish deep-backlog gains.

The engine is discrete, but engine calls, scheduling passes, useful-token observations and block-release acknowledgments are different events. A step is not uniformly one useful token per resident, and discrete steps do not have equal wall-clock cost.

### Bounded admission experiment outcome

`results/scheduling_headroom_20260915/REPORT.md` and `results.json` preserve the frozen-source W-SMALL comparison: arrival cap128 **340.884 s**, cap256 **358.009 s**, and target-aware adaptive admission with maximum submitted population 256 **348.495 s**, all emitting 721,595 tokens. The treatment was 2.23% slower than cap128 and used 4.320 s of measured hook CPU against a 3.408 s limit. It fell back after 1,115 of 1,763 first native admissions; no repeat or parameter sweep followed. These are one-run observations, not a general impossibility result or a C/O oracle-gap measurement.

The runner's `policy.native_admission` accepts `baseline`, `shadow`, or `adaptive`, with native arrival/greedy policy and no synchronous discipline or forced throttle. `baseline` supports submitted cap128/256; `shadow` and `adaptive` require cap256. `--action admission-audit` runs isolated installed-native CPU conformance fixtures, not a benchmark. Use the saved recovery-first RCM plans and retained actor; do not cold-initialize a substitute.

The seam executes immediately before the native waiting-path allocation, after resident allocations and native chunk arithmetic. Its preview compiles the audited allocator's read-only prefix; commit remains native. Above 128 ever-started unfinished requests, the gate requires actual free capacity for candidate endpoint/immediate allocation plus resident endpoint growth, with no predicted releases. Submitted population, ever-started population and scheduled batch are distinct. Recomputed requests bypass the fresh gate; deferred fresh tails retain FCFS order through native queues. Native step functions, batch queue and free discipline remain unchanged and identity-checked.

The endpoint comes from the existing C/O `FutureModel` and is a conservative allocation proxy, not a mixed-workload time model. The controller records issued/computed/observed work separately, invalidates deterministic mismatches, and uses native cap256 progress on exhaustion. Its budget does **not** cap continued observation/fallback costs: the measured total exceeded 1% of the matched cap128 makespan. Keep that failure visible; do not enable this experiment as a default optimization. Generic harness/native traversal overhead remains inside makespan but is not separately priced by hook timers.

## Measured motivation

Physical TPU v5e / TP16 / Gemma-4-31B, W-LARGE, 600 requests, exactly 483,554 useful output tokens per run:

| Completed comparison | Makespan (s) | KV preemptions |
|---|---:|---:|
| Earlier strict seed planner, cap128 | 359.638 | 0 |
| Native LPT, cap64, focused pair | 299.619 | 87 |
| Native seed-planner ordering C, cap64, focused pair | 307.864 | 159 |

The focused pair shares serving source, dependency hashes, manifest and execution mode. It is one repeat per arm. Comparing the new mode with the earlier strict seed also changes cap128 to cap64 and source; it is not an isolated execution-mode experiment.

The native planner order executed 8,794 iterations versus LPT's 8,813, but scheduled 1,527,830 total tokens versus 1,474,251. Mixed prefill/recompute steps numbered 781 versus 606. Terminal drain was 1.281 s versus 1.043 s. Extra preemption/mixed work is the leading explanation for the remaining gap, not a measured kernel-time decomposition.

Evidence: `results/20260915_planner/P3_native_pair_results.json`, `P3_native_pair_analysis.json`, and their two referenced raw artifacts. Earlier strict-mode evidence is in `P3_quick_results.json` and `P3_quick_analysis.json`. Preserve all negative results.

## 1. The authoritative transition contract

Conceptually, for an event ordinal e:

    S[e+1] = F_native(S[e], action[e], observed_event[e])

The ordinal labels events; it is never a substitute for per-request service. Given full state, configuration and the relevant completion-event ordering, transitions are discrete. This does not assert advance knowledge of accelerator completion times or unknown O termination events.

`P3_allocator_probe.json` records installed source for `EngineCore.step_with_batch_queue`, `Scheduler.schedule`, both scheduler `_update_after_schedule` implementations, `_preempt_request`, `update_from_output`, and KV allocation/free methods. It shows:

- Native batching can issue a new batch and return before consuming an earlier output; it later retires the oldest queued result and calls `update_from_output`.
- `_update_after_schedule` advances `num_computed_tokens` and `num_in_flight_tokens` when work is scheduled. The computed-token cursor is therefore not proof that output has been observed.
- `AsyncScheduler` maintains output placeholders separately.
- Preemption resets the computed-token cursor and returns the request to waiting; already emitted useful output is not thereby erased.
- The recorded batch queue capacity is two. Discover and verify the active configuration rather than baking that value into a model.

### State and ownership

| State | Meaning and owner |
|---|---|
| Stable external/internal ID mapping | Reuse the runner's mapping; no unmapped-ID-to-zero fallback |
| Observed generated count g | Actual engine output observations; never inferred from iteration count |
| Native computed cursor c | Scheduling/recompute position, potentially including issued work; distinct from g |
| Issued work and output placeholders | Per-request work attached to identified outstanding batches |
| Request phase and queue membership | Prefill, decode, recompute, waiting/running, and observed completion state |
| Allocated block ownership | Native allocator's actual blocks, excluding its null block |
| Accepted but unallocated demand | Commitments not already represented by allocated blocks |
| In-flight FIFO and retirement state | Issue/retirement identities and actual output events, not assumed zero latency |
| Allocator-return state | Blocks become available only when the allocator actually returns them |
| Current scheduling program point | Remaining token budget, actual `throttle_prefills`/eligibility flags, already scheduled requests and provisional allocations/refunds |
| Configuration and state version | Source identity, block groups/size, caps, queue capacity and relevant feature flags |

Use three explicit boundaries: **before the controlled admission decision**, **after native scheduling/bookkeeping**, and **after output retirement/allocation return**. Compare like boundaries. A request finishing and its blocks being reusable must not be conflated.

All rows containing engine handles or privileged targets stay in the adapter. The policy receives only its legal projection; see section 5.

## 2. One implementation of planning-relevant mechanics

The engine and planner must not maintain independently handwritten versions of token-budget, block-growth, chunking or refund arithmetic.

Reuse the installed native non-mutating allocation query where verified suitable. Where necessary, factor the smallest planning-relevant arithmetic into shared helpers used by both preview and commit. A source-validated hook alone does not establish preview equivalence: exact transition checks are also required. Unsupported native source/configuration must be rejected before enabling the intervention.

The initial supported configuration is the measured text-only, nonsharing single-allocation-group setup: prefix caching disabled, no speculative/lookahead decoding, KV offload, connectors, encoder work or pipeline parallelism. Verify these conditions and the mounted model/KV geometry before installation; do not infer them from a model name. Extend support only with corresponding native transition checks.

A preview operates on a versioned, read-only view or bounded delta overlay. It must not allocate/free live blocks, mutate a request/queue, call model execution, consume a real output, or copy the engine/weights. Pure decision arithmetic may be shared; allocator mutation and asynchronous execution remain native operations.

### First controlled seam

Intercept the native waiting-admission decision after the running-request pass has established its scheduled work, remaining token budget and allocations. The precise source location must be audited before implementation. This is not a post-filter on `SchedulerOutput`.

The adapter must include the pass's provisional work: native allocation may already have changed while `_update_after_schedule` has not yet advanced all request cursors. Previewing from an earlier phase manufactures a false mismatch.

First actions:

- **ADMIT(candidate ID):** select an eligible fresh candidate from a bounded frontier for a native admission attempt. Native code determines its token chunk and commits allocation/bookkeeping only if accepted; a rejected attempt leaves the request correctly queued.
- **DEFER_NEW:** admit no further fresh request in this pass; continue resident service and output retirement normally.
- The ordinary baseline admission choice remains the explicit budget-exhaustion action. A baseline fallback is logged and never counted as a co-designed decision.

No arbitrary resident permutation, custom victim, token-budget override, post-schedule output edit or forced batch drain is included in this first change. Held candidates remain in a single owned queue and are neither lost nor duplicated. Record actual admitted IDs and native token assignments, not merely a preferred order.

Fresh admission control must not hold up already admitted requests returning for recomputation. Preserve their native eligibility and resume path, and retain issue/retirement identity across preemption so an outstanding result is not mistaken for new work. Do not classify a request as fresh merely because its observed g is zero.

### Preview → validate → commit → observe

1. Capture the decision boundary, state version and legal inputs.
2. Evaluate a bounded set of admission/defer alternatives using the shared mechanics and legal future provider.
3. Return an action with its state version, prerequisites, expected immediate structural delta, cost-proxy components and declared uncertainty.
4. Revalidate before commit. If allocations, output retirement, queue membership or configuration changed, discard the stale action; never wait for the engine to recreate its old state.
5. Apply the action through the audited pre-allocation seam. Native code performs allocation, request-state changes, refunds, issue and later output retirement.
6. Compare the predicted immediate delta with actual native bookkeeping at the matching boundary. On deterministic mismatch, invalidate the model/arm, preserve the evidence and stop further optimized interventions; do not patch counters or reinterpret the mismatch as timing noise.

The plan is now an executable action contract. A future dependency such as “after the oldest batch retires and these blocks are returned” is a state predicate, not an engine-step number or a wall-clock deadline.

## 3. Memory safety and progress without forced synchronization

Keep the native batch queue and overlap. Do not switch to `core.step`, impose ready-first traversal, change free behavior or add a synchronization barrier merely to simplify the model.

For an immediate commit, new native allocation demand must be covered by **currently available** native capacity. Existing native preemption paths remain authoritative; do not add eviction to a waiting-admission branch that normally rejects an allocation. Predicted future completions never increase current free capacity. Track commitments across successive candidate decisions; do not double-count already allocated in-flight blocks as unallocated debt. Measure unattributed retained allocations separately.

Future forecasts rank actions and estimate pressure/recomputation risk. A finite-horizon forecast is **not** a whole-horizon, zero-preemption certificate. The first co-designed controller allows native preemption, makes no deterministic zero-preemption claim and must price its extra work rather than minimize the count blindly.

Any later predictive-release certificate requires a separate proof covering issued work, independent progress, all supported output-retirement orderings and actual free acknowledgments. Exact d alone is insufficient. Never silently transfer the old synchronous proof to native asynchronous execution.

Progress requirements:

- Deferring fresh admissions must not stop resident service or retirement of pending outputs.
- If only a pending output can release capacity, continue the native retirement path instead of busy-spinning on the gate.
- If there is no resident/in-flight work and backlog remains, admit an individually feasible request through the baseline path, or report a real infeasible/unsupported state. Do not return empty schedules forever.
- On controller invalidation, stop its interventions and allow native cleanup/progress; do not convert an already admitted set into a claimed endpoint-safe set.
- Interruption handling must preserve requests, restore hooks and verify quiescence without killing/reloading the retained engine. Stopping a submitting Ray job alone is not proof that a detached actor call ceased.

## 4. Optimize actions the engine actually executes

Initialize with the complete executable native baseline, not a serial ideal timetable. The current bulk investigation uses the best previously measured arrival/cap128 configuration and a static cap256 control, with an adaptive controller permitted an admitted population up to 256 under identical native engine limits. Preserve arrival priorities initially; change admission timing/effective concurrency only. The old sparse calendar cannot authorize asynchronous actions or substitute its objective for measured goodput.

Keep the first controller small:

- For the first comparison, preview admission of the eligible baseline head or defer fresh admission. Broader candidate selection belongs to a separately qualified experiment, not a simultaneous ordering change.
- Preview the next native admission decision exactly for known state. Any extension through future issue/retirement events must state its assumptions and uncertainty; do not advance every resident by one per event.
- Future C residual work is d minus observed g. O uses the same provider interface with a conditional law; scheduled/in-flight work is tracked separately to avoid duplicate issue, not subtracted as confirmed useful output.
- Include residual work beyond the lookahead in the score. Moving recomputation, growth or a long survivor past the horizon must not make it appear free.
- Re-evaluate on decision-relevant events: admission opportunity, block growth/allocation return, phase change, output retirement or preemption. Update state incrementally; do not rebuild every resident's full calendar on every engine call.

The score is a **cost proxy**, not a measurement. It must account for useful decode, initial prefill versus recomputation, mixed-step composition, request/token padding when observed, context-dependent work, pipeline occupancy and controller CPU work. Report unmeasured components as uncalibrated; do not invent coefficients to fill them in.

Overlapping batches require an issue/retirement or critical-path model. Summing schedule-to-output durations across overlapping batches is not a makespan estimate. Discrete iteration count is not sufficient either. Do not derive pure decode performance by subtracting a separate prefill run.

The old `6.75 + 0.292*b + 0.0514*padded(b)` curve remains labeled `surrogate_J` evidence only. It cannot be presented as a validated mixed-step/native-pipeline cost model. Select any replacement using held-out **whole-run** residuals within compatible source/mode/workload cohorts. Insufficient independent data means unvalidated, not a license for a broad calibration sweep.

Proposed experimental budgets, not measured achievements: startup at most 5 s; at most 1 ms per admission decision, including its snapshot/preview work; aggregate controller CPU at most 1% of the already measured matched baseline makespan. Charge incremental maintenance and unsuccessful previews too. On exhaustion choose the explicit baseline action and record coverage. Do not claim controller performance from a fallback-dominated run.

## 5. C/O boundary under co-design

The engine necessarily owns real stopping targets; this does not make them legal policy inputs.

- Retain `FutureModel` as the information boundary. C receives exact targets; O receives only the public workload law, prompt features, observed generated tokens, allocated blocks and permitted observations of issued/retired work and completions.
- Assert O construction rejects d, seeds, nested payloads and subclass carriers. Copy immutable public inputs rather than retaining a view into privileged engine/manifest storage.
- The native adapter may use full engine state for actual execution. An O counterfactual preview must not query that private target state and then leak its consequences as an exact future finish, feasibility score or remaining-work estimate. Forecast unknown termination through the O provider.
- The computed cursor and in-flight counts describe already issued work, not an oracle remaining length. Always derive remaining useful work from observed g plus the legal future model.
- Exact structural checks cover transitions determined by legal known state and actual observed events. O's uncertain future completion forecast is not required to equal a hidden target in advance; score its calibration separately.
- Use the same action space, native hooks, cost proxy, budgets and progress rules for C/O. Preserve the fixed, prompt-stratified, length-independent hash cohort; never refit from the completed subset alone.

The first performance check is Track C admission/concurrency versus the best measured static policy and a larger-cap control, not an oracle-gap claim. O gets a targeted information-boundary check before its path is enabled; a full C/O pair follows only if useful and requested.

## 6. Transition conformance before optimization claims

Reuse native source and already recorded data first. Add only the missing observations necessary to join a decision, issued batch, output retirement and allocation return. Existing aggregate traces do not contain every per-request/in-flight state; do not reconstruct missing histories by assuming one token per engine step.

A compact audit record identifies the decision/program point, state version, selected action, issued batch IDs, predicted/actual request-token assignments, allocation deltas, preemptions/refunds, observed outputs and actual frees. Keep native identifiers inside the adapter where possible. Bind evidence to source/configuration hashes and explicitly record execution mode.

Required targeted cases:

1. A prefill split across passes while an earlier batch is outstanding.
2. A KV block-growth boundary with insufficient current free blocks.
3. A computed cursor advanced by scheduling before the corresponding output is observed.
4. A completed output whose allocation is not yet reusable under the active native free rules; do not assert a fixed lag.
5. Preemption after earlier requests were scheduled, including token refund and current-victim break behavior.
6. An admission/defer action invalidated by a changed version before commit.
7. No-progress prevention with only pending outputs or only an individually feasible backlog request.

Preview-versus-native assignments and allocation deltas must agree exactly where the state/event is known. Record the first divergent program point and relevant operands. Allocation IDs may differ in a nonmutating preview; compare the native-equivalent demand and ownership effects, not invented block IDs. Policy-law uncertainty and wall-clock prediction error are distinct from bookkeeping mismatch.

No post-filtering `SchedulerOutput` or adjusting engine counters to force agreement. Do not assume reordering is victim-only: installed tests already disproved that general claim. Keep native traversal and victim selection unchanged in this admission slice.

Correctness fixtures and planner-internal structural projections are not policy performance evaluations. No offline scheduler simulation, synthetic execution loop, or predicted duration is acceptable benchmark evidence. All throughput comparisons use real serving on physical TPU hardware.

## 7. Minimal implementation and measurement sequence

Responsibility map for implementation, not a statement that these changes exist:

| Existing location | Required change |
|---|---|
| `benchmarks/kv_execution.py` | Versioned native-boundary adapter, explicit issue/retirement/allocation accounting, action validation and audited commit seam; no forced synchronous discipline |
| `benchmarks/kv_planner.py` | Bounded native-action proposal/scoring instead of optimizing starts whose timing is discarded; retain old calendar as a reference diagnostic |
| `benchmarks/kv_future_profile.py` | Preserve legal C/O futures; no engine references or hidden-target-dependent O previews |
| `benchmarks/run_measurement_audit.py` | Explicit co-design arm, same native baseline configuration, controller accounting, conformance records and scoped cleanup |
| `benchmarks/analyze_planner.py` | Event/batch-aligned transition residuals and measured wall/output comparison; no epoch=engine-call replay claim |

First establish shared arithmetic and one admission seam in non-intervening shadow mode: preview only, with live engine behavior unchanged. This checks fidelity; it does not demonstrate speedup. Then enable admission/defer actions with baseline arrival priorities unchanged. The current goal permits adaptive concurrency while retaining native resident traversal and eviction; broader priority changes are not part of the first comparison.

After conformance passes, follow the bounded A/B/C comparison in `SCHEDULING_HEADROOM_GOAL.md`: saved B1/W-SMALL manifest, static arrival cap128, static arrival cap256, and adaptive admission with maximum admitted population 256. Reuse the hot actor, model, trace settings and asynchronous configuration. Assert 721,595 useful outputs, matching provenance and restored allocator state. Include all online controller overhead in dispatch-to-drained makespan; report startup separately and in startup-plus-serving time.

Repeat only the relevant matched pair if a promising or close result needs repeatability evidence, with reversed order and no source/configuration edits. Historical repeat spreads are not a universal significance threshold. No automatic additional workload, backlog-depth, padding, F3, throttle or long-search matrix. A failed prediction is a result to report, not a reason to launch another sweep.

Use `scripts/submit_measurement_audit.py` through RCM, one arm per plan step, starting with recovery, and the existing `20260914_tracks` actor. Observe status/events through RCM, never bespoke Ray job polling. No model reload or placement-group recreation. If any new distributed initialization is eventually necessary, explicit topology discovery and node-pinned placement remain mandatory.

## 8. Retained mathematics, with narrower claims

The previous sparse calendar is exact for its idealized nonpreemptive, uniformly served model. Its event sweep, integer comparisons and feasible-incumbent properties remain useful reference tools. They are not a proof of native asynchronous execution fidelity.

For block size L, define H(x) with x=qL+r as:

    H(x) = L*q*(q+1)/2 + r*(q+1)
    A = sum_i [H(p_i+d_i) - H(p_i)]

This is after-token uninterrupted useful-service KV area. For that model, T >= ceil(A/C) and mean useful-service batch <= C*D/A are necessary bounds, not attainable targets or physical makespan bounds. Before-sampling native allocation uses a different phase; compare native allocation records at their actual boundary. Stalls, recomputation and retained in-flight allocations add resource residence not represented by this invariant area.

The exact residual order-statistic tail formula likewise requires uniform uninterrupted service. In native execution use observed drain, with the last step at least half the median positive executed batch through run completion. Do not manufacture residual service from elapsed engine-call counts.

The generalized ideal scheduling problem's hardness is not evidence that more search is the right next step. The immediate defect is transition/cost-model mismatch. Fix and verify that contract before expanding optimization.
