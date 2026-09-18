# F3: eviction borrows KV from the future; it does not eliminate KV work

## Recommendation and scope

Implement F3 as **pressure-horizon victim selection with a terminal-work penalty**, not “largest context,” “smallest context,” “most remaining,” or “fewest generated” globally. Keep the distinction between (i) an exact conditional optimization, (ii) an implementable proxy for that optimization, and (iii) hardware makespan. None of the available measurements identifies a globally optimal scalar eviction index.

The most important integration finding is negative: **moving the desired victim to the tail is not generally a victim-only intervention.** It also moves that request in the token-allocation traversal. A successful sign test proves the sign, not causal isolation. Restrict list-only F3 to an audited homogeneous-decode regime, or label its result a combined traversal/victim policy. A general victim-only experiment requires an independent victim-selection seam.

Evidence already available:

- `results/scheduling_tracks_20260914/scheduler_probe3.json`: installed `AsyncScheduler`, parent `Scheduler` FCFS victim line 535 is `self.running.pop()`; line 539 tests whether the victim is the current request. The priority branch includes scheduled-token refunds at lines 516–523. Line 600 conditions waiting admission on no preemptions. These are substantive control-flow interactions, not just container details.
- The same artifact records FCFS, `running_is_list=true`, sequence limit 512 and token budget 2048. `REPORT.md`'s older statement that the victim probe is missing is superseded by probe3.
- `REPORT.md`, recompute section: cap256 has 204,515 waste tokens, 15.1% of scheduled tokens; uncapped has 221,140, 16.1%. The isolated-prefill coefficient does **not** identify mixed-step recompute time.
- The full installed scheduler source is not mounted in this design workspace at the path recorded by the probe. Source-level sufficient conditions below must be checked against the orchestrator's actual build; probe snippets are not the whole method.

No TPU/Ray job or benchmark was run for this design. The numerical checks below check formulas, not projected or measured performance.

## 1. Exact objective, information and feasibility

For resident request \(i\), let \(p_i\) be prompt length, \(g_i\) its engine-observed emitted decode tokens, \(c_i=p_i+g_i\), and \(k_i\) its **immediately releasable** KV blocks. In the stated nonshared, nondeferred geometry,
\[
k_i=\left\lceil c_i/256\right\rceil.
\]
Use the allocator's released-block accounting if allocated blocks include in-flight/deferred/shared blocks. Do not obtain \(g_i\) by subtracting the prompt from `num_computed_tokens`: preemption resets that counter but does not erase emitted output.

Track C knows \(r_i=d_i-g_i\). Track O sees no \(d_i\), `max_tokens` populated from \(d_i\), oracle sort field, or target-derived residual. It receives a whitelist of request ID, \(p_i,g_i,k_i\), public configuration, and completion observations; assert prohibited fields absent at construction. A public workload-wide cap is legal, an individual engine target is not.

The true objective is \(\min_\pi E[M_\pi\mid\mathcal I]\), where \(M_\pi\) is final-completion wall time and \(\mathcal I\) is the permitted information. For a deterministic known manifest/environment, remove the expectation. Every request must complete; no decode work is discarded, and every actual allocation must obey
\[
\sum_{i\text{ resident at }t} k_i(t)\le566.
\]
The engine's actual allocation requirements, token budget, deferred frees and asynchronous execution are additional constraints. A request that does not fit alone at its legal maximum context is infeasible for the termination guarantee below.

At a decision with block deficit \(\Delta>0\), a single-victim choice must have \(k_i\ge\Delta\). If no candidate does, this is a multiple-victim problem; selecting a singleton and calling it optimal is invalid. For an exact dynamic program, the optimal victim is
\[
i^*\in\arg\min_i\{R_i(s)+V(s_i^{\rm evict})\},
\]
where \(s\) is the full scheduling state, \(R_i\) the incremental cost of the transition and eventual reconstruction allocated to it, and \(V\) the optimal continuation value under a consistent non-double-counting convention. This is a characterization, **not** a tractable planner.

A decode-only median-step regression does not supply \(R_i\) or \(V\). In particular, \(\sum b_t=D\) fixes useful output work, not recompute, mixed-step padding, context-dependent attention or critical-path overlap. The measured 0.292 ms slope is not a hardware lower bound on each token's cost. F3 can change makespan at the same useful-decode batch counts through these other channels.

## 2. What an eviction buys: early relief followed by repayment

Consider a diagnostic model in which an uninterrupted request receives one decode token per step. Let \(q_i(s)\) be its remaining service-indexed KV trajectory, zero after its completion. If eviction delays that entire trajectory by \(h_i\) steps, and reconstruction is temporarily omitted,
\[
\Delta q_i(s)=q_i(s)-\mathbf1_{s\ge h_i}q_i(s-h_i).
\]
Changing variables proves
\[
\sum_{s\ge0}\Delta q_i(s)=0.
\]
This holds for growing, block-rounded KV, not only constant contexts: eviction moves the same service-indexed area later. Reconstruction and idle residency can add further occupancy. Thus “blocks freed × time away” is **not** net lifetime KV savings. It is useful only when early KV is more valuable than later KV. \(\blacksquare\)

Let \(\lambda_s\ge0\) be the marginal value of an available block at future step \(s\), in ms/(block-step), and \(P_i\) any terminal-path cost not represented by that KV value. A local, first-order cost is
\[
J_i=R_i+P_i-\sum_s\lambda_s\Delta q_i(s).
\tag{1}
\]
It is exact for a specified additive rent model with fixed counterfactual trajectories; it is only an approximation to the engine's nonlinear makespan. Constant rent over the whole trajectory makes the KV term zero. Uniformly pricing only the early positive area and ignoring repayment is an explicit **finite-pressure-horizon assumption**, not an identity.

For constant occupancy \(k_i\), common absence horizon \(h\), rent \(\lambda\) during \([0,h)\), and zero rent thereafter, (1) becomes
\[
J_i=R_i+P_i-\lambda k_i\min(r_i,h).
\tag{2}
\]
For block growth, replace \(k_i\min(r_i,h)\) with
\[
B_i(h)=\sum_{s=0}^{h-1}\left\lceil\frac{c_i+s}{256}\right\rceil\mathbf1_{r_i>s}.
\tag{3}
\]
This is early counterfactual occupied block-time avoided. It assumes the victim stays out for the horizon and would otherwise receive one token each step; its interpretation must be weakened if either assumption fails. Freed capacity with no productive alternative use has zero rent.

**The conditional optimal singleton rule is to minimize (1), or (2) under its hypotheses, among feasible candidates.** Proof: all candidate-independent terms cancel in pairwise comparison. No ratio is needed for this additive objective. If instead the chosen local objective is recompute per unit early relief, the exact rule for *that different objective* is \(\arg\min_i R_i/B_i\). A ratio is not generally optimal for an indivisible deficit-constrained eviction set. \(\blacksquare\)

For the token proxy \(R_i\propto c_i\),
\[
256(1-1/k_i)<c_i/k_i\le256.
\]
Recompute cost and blocks freed almost cancel except for small or partially filled blocks. Duration, actual reconstruction efficiency and terminal criticality matter more than raw size. Two failure examples for (2), using \(h=128,\lambda=2,R=c,P=0\), deficit one block and token-equivalent units:

| candidates \((c,k,r)\) | local costs \(J\) | failed size rule |
|---|---:|---|
| \((2048,8,1)\), \((512,2,128)\) | 2032, 0 | largest evicts the expensive imminent completion |
| \((256,1,1)\), \((512,2,128)\) | 254, 0 | smallest gets only one block-step of useful relief |

The examples use the frozen-block submodel explicitly. Including positive block growth for the long candidate only strengthens these comparisons.

## 3. Track C: the sign and its flip are conditional

Compare a long-remaining candidate \(L\) and short-remaining candidate \(S\). From (2), choose \(L\) exactly when
\[
(R_L-R_S)+(P_L-P_S)
<\lambda\{k_L\min(r_L,h)-k_S\min(r_S,h)\}.
\tag{4}
\]
Equality is a tie. With identical costs/blocks and no terminal penalty, **most remaining is weakly preferred**, strictly while the two clipped residuals differ. Once both residuals exceed \(h\), additional remaining work buys no more frozen-block early relief. “Evict the longest” has no justification beyond the pressure horizon from this argument.

A nearly completed request with \(r_i=1\) buys only \(k_i\) early block-steps even if quarantined for 128 steps: it would have freed those blocks naturally after one step. At \(c_i=1024,k_i=4\), its reconstruction/relief ratio is 256 tokens/(block-step), versus 2 for an otherwise equal request with \(r_i\ge128\). It is **128 times worse under this local efficiency metric**, not a 128-fold measured time loss. It also defers a completion, rather than advancing one.

The answer flips when protecting a long request's terminal path outweighs that early relief. For a specified fixed background completion horizon \(L_0\), delay \(h\), and residual \(r\), define
\[
\delta(r;h,L_0)=[h+r-L_0]_+-[r-L_0]_+,
\quad [x]_+=\max(x,0).
\tag{5}
\]
This is exactly the increment in \(\max(L_0,r)\) caused by delaying this request, assuming background completion is unaffected and reconstruction is handled separately. It is nondecreasing in \(r\). Setting \(P_i=\nu\delta_i\), with \(\nu\) the cost per terminal step, gives a precise flip test in (4).

Example: equal \(c=1024,k=4\), residuals 8 and 256, \(h=128,L_0=256\). Early relief difference is \(4(128-8)=480\) block-steps. Terminal increments are 0 and 128. The long victim is preferred iff \(128\nu<480\lambda\); the short victim is preferred iff \(\nu>3.75\lambda\). Thus a pressure-dominated middle phase and a terminal-path-dominated tail can require opposite victims. More generally, if both residuals exceed \(h\), early relief ties and any larger terminal penalty favors evicting the shorter request. These claims follow directly by subtracting the two costs; they do not assume that LPT or SPT globally minimizes makespan. \(\blacksquare\)

In a real high-concurrency run, \(h,L_0,\lambda_s\), survivor progress and reconstruction overlap all depend on the action. Exact \(d_i\) alone does not identify them. Freeze them only as an explicitly logged local approximation, or obtain them from an independently validated planner.

## 4. Track O: residual life is not monotone in attained service

Let unbounded \(X\) have \(\log X\sim N(\mu,\sigma^2)\), let \(S(x)=P(X>x)=\bar\Phi((\log x-\mu)/\sigma)\), and condition on an unfinished request with attained service \(g\). For \(g>0\), completing the square in the truncated first-moment integral gives
\[
m(g)=E[X-g\mid X>g]
=e^{\mu+\sigma^2/2}
\frac{\bar\Phi((\log g-\mu-\sigma^2)/\sigma)}
{\bar\Phi((\log g-\mu)/\sigma)}-g.
\tag{6}
\]
At zero, \(m(0)=e^{\mu+\sigma^2/2}\). Equivalently, the tail-integral identity gives
\[
m(g)=\frac{\int_g^\infty S(u)\,du}{S(g)},\qquad m'(g)=a(g)m(g)-1,
\tag{7}
\]
where \(a(g)=-S'(g)/S(g)\) is the hazard. These integrals are finite for lognormal \(X\).

As \(g\downarrow0\), \(a(g)\to0\), so \(m'(g)\to-1\): expected remaining work initially **decreases**. With \(z=(\log g-\mu)/\sigma\) and inverse Mills ratio \(M(z)=\phi(z)/\bar\Phi(z)\),
\[
\frac{a'(g)}{a(g)}=\frac{M(z)-z-\sigma}{\sigma g}.
\]
The normal-tail Mills expansion \(M(z)=z+1/z+O(z^{-3})\) implies eventually decreasing hazard. The same differentiable tail expansion yields
\[
m(g)\sim\frac{\sigma^2g}{\log g-\mu},\qquad
m'(g)\sim\frac{\sigma^2}{\log g-\mu}>0.
\]
Thus attained service is not a monotone proxy for remaining work. The initial downward and eventual upward behavior are proved; no unproved assertion of a unique turning point is needed.

**Actual workload correction: rounding and winsorization.** Let
\[
D=\min(C,\max(1,\operatorname{round}X)),
\]
with public integer cap \(C\), possibly further bounded by public maximum model context minus prompt. Ties in rounding have probability zero. Define \(S_D(j)=P(D>j)\). Then
\[
S_D(0)=1,\quad
S_D(j)=\bar\Phi((\log(j+1/2)-\mu)/\sigma)\ (1\le j<C),\quad
S_D(j)=0\ (j\ge C).
\]
For integer \(0\le g<C\), the exact conditional mean is
\[
E[D-g\mid D>g]=\frac{\sum_{j=g}^{C-1}S_D(j)}{S_D(g)}.
\tag{8}
\]
It is not a truncated-lognormal formula: the cap has a point mass. In particular the residual is exactly 1 at \(g=C-1\), and cannot keep increasing toward the cap.

For eviction, the relevant statistic is generally **not (8)** but
\[
A_h(g)=E[\min(D-g,h)\mid D>g]
=\sum_{s=0}^{h-1}\frac{S_D(g+s)}{S_D(g)},
\tag{9}
\]
with survival zero beyond the cap. The block-aware benefit is
\[
\bar B_i(h)=\sum_{s=0}^{h-1}\left\lceil\frac{c_i+s}{256}\right\rceil
\frac{S_{D_i}(g_i+s)}{S_{D_i}(g_i)}.
\tag{10}
\]
Use the full conditional distribution to evaluate expected terminal penalty, rather than substituting the mean inside the positive-part function. For fixed integer \(L_0,h\),
\[
E[\delta(R;h,L_0)]=\sum_{a=L_0-h}^{L_0-1}P(R>a),\qquad R=D-g\mid D>g,
\tag{11}
\]
where \(P(R>a)=1\) for \(a<0\), and otherwise equals \(S_D(g+a)/S_D(g)\).

Induced ordering: in a region of decreasing hazard, survival to an additional fixed horizon increases with attained service, because
\[
\partial_g\log\{S(g+s)/S(g)\}=a(g)-a(g+s)\ge0.
\]
Hence (9) increases there, and roughly equal \(c/k\) favors evicting **more**-generated requests. In an increasing-hazard region with the whole horizon inside that region, the inequality reverses and favors fewer-generated requests. Near the cap, preserve the requests that must finish imminently. Prompt size, partial blocks and terminal penalty can override all these attained-service comparisons. \(\blacksquare\)

Numerical formula checks for W-SMALL, \(\mu=\log256,\sigma=1,C=4096,h=128\):

| attained \(g\) | unbounded mean residual | rounded/capped mean residual | rounded/capped \(E\min(R,128)\) |
|---:|---:|---:|---:|
| 0 | 422.073 | 417.361 | 115.838 |
| 64 | 392.273 | 387.640 | 106.116 |
| 128 | 405.131 | 399.545 | 104.850 |
| 512 | 560.906 | 542.283 | 110.136 |
| 1024 | 757.671 | 701.413 | 114.704 |
| 2048 | 1101.427 | 851.141 | 118.888 |
| 3072 | 1408.745 | 681.814 | 120.912 |
| 4095 | 1694.317 | 1.000 | 1.000 |

These are distribution calculations, not sampled-manifest statistics or performance predictions. In particular, the mean residual falls from 2048 to 3072 while 128-step relief rises: replacing the conditional distribution by a mean can reverse the desired eviction ordering.

**No “nearly as good as C” conclusion follows.** For any permitted observation \(Z\), optimal Bayes local cost is \(\min_iE[J_i\mid Z]\), while clairvoyance attains \(E[\min_iJ_i\mid Z]\), which is no larger. Identical observable \((p,g,k)\) can hide one residual of 1 and one of at least \(h\), with exchangeable labels. An O rule then obtains expected frozen-block relief \(k(h+1)/2\), versus \(kh\) for C. Such length intervals have positive probability under the capped lognormal away from the cap. This conditional example disproves an observation-only near-oracle guarantee; it is not a claim that this event dominates the workload. The actual gap is a paired hardware question.

For a benchmark-prior O arm, fix the declared public \((\mu,\sigma,C)\) before seeing the manifest. For a learned arm, fit only completion records from *closed submission cohorts*, or handle completion-selection bias explicitly: fitting to the currently completed subset overweights short requests. Do not silently estimate the distribution from all manifest targets. The public-prior and learned-distribution arms answer different information questions.

## 5. Concrete, bounded F3 specification

The following is a **fully specified local proxy**, not a globally optimal makespan theorem. Its virtue is that it implements the derivation without an unconstrained optimizer or a mistaken universal sign.

Common configuration for C/O twins:

- Horizon \(H=128\) scheduler opportunities; no speculative multi-token decode.
- Proxy reconstruction cost \(\widehat R_i=c_i\), measured in recompute-token units. Do not convert it to ms using the isolated-prefill calibration.
- Early KV rent \(\widehat\lambda=256/H=2\) token-equivalents/(block-step). This is a declared proxy normalization: one fully occupied block's reconstruction costs roughly 256 tokens; it is not an empirically identified rent.
- While fresh undispatched or engine-waiting nonquarantined work remains, \(\widehat\nu=0\). In the final drain, \(\widehat\nu=256\) token-equivalents/terminal step. This deliberately protects terminal work and must be reported as a policy choice, not a calibrated conversion.
- In final drain set \(L_0=\lceil\max_j E[R_j\mid\mathcal I]\rceil\) over unfinished resident requests, taking C's distribution to be a point mass. This is a plug-in background horizon; O still evaluates each candidate's terminal expectation by (11), not at its mean. Log that O's random background maximum is approximated, not integrated.
- Score
\[
\widehat J_i=c_i+\widehat\nu E[\delta_i]-2E[B_i(H)].
\tag{12}
\]
Choose minimum among single-victim-feasible, unprotected candidates. Ties within relative/absolute tolerance \(10^{-12}\) use lower \(c_i/E[B_i]\), then the original FCFS tail rank. Never divide by zero: a zero-benefit candidate has efficiency \(+\infty\), but can still be chosen when eviction is mandatory and all candidates are bad.
- C uses point-mass survival from exact live \(d_i-g_i\); O uses the declared survival provider. Every other operation, guard, traversal convention and parameter is identical.

A conservative first deployment reorders only homogeneous one-token running steps, preserves original survivor order, and never globally sorts `running`. On a mixed/recompute or unaudited step it leaves the baseline order unchanged. Report eligible-step coverage and preemption coverage; an arm with no covered preemptions has not tested F3. A prospective-demand computation may select the candidate before `schedule()` but must be checked against the actual allocator result; it is not permission to edit `SchedulerOutput` afterward.

Pseudocode (all decisions at a safe scheduler boundary, using current engine state):

```text
construct(public_config, whitelisted_policy_view):
    assert no per-request target/oracle fields in O view
    build survival tables for the public distribution; C uses point masses
    initialize stable FCFS ranks, quarantine, guard state and cost counters

before_schedule(state):
    apply the common anti-thrash admission/hold rule below
    if not homogeneous_decode_audit_conditions(state): return baseline_order
    if no prospective KV deficit: return baseline_order
    read live generated counts and immediately releasable blocks
    compute final-drain L0 and common nu, if needed
    candidate = null
    for each unprotected resident with blocks >= deficit:
        obtain B and expected delta from its C/O survival provider
        J = context + nu * expected_delta - 2 * B
        update best candidate using the specified stable tie rule
    if no feasible singleton: use audited baseline multi-victim handling
    otherwise: stable-remove candidate, append candidate to running tail
    call the original schedule exactly once; never post-filter its result

observe_schedule_and_output(result):
    record actual victims, free-block delta, recompute, live progress
    update quarantine/guard state from actual victims, not predictions
    restore canonical survivor order only at the next audited safe boundary
    learn lengths solely from eligible completed cohorts
```

Multiple-victim F3 is intentionally not claimed optimal: a one-time tail ordering may need several pops, and the remaining deficit changes after each pop. The initial arm uses baseline behavior for predicted multi-victim events; records whether such events occurred unexpectedly; and cannot support a general F3 claim if that coverage is material. A subsequent separate-selector implementation can greedily re-evaluate (12) after each actual free, but that is a greedy set heuristic, not an exact knapsack solver.

Complexity: with \(H\le256\), each \(\lceil(c+s)/256\rceil\) has at most one upward transition within the horizon. Put \(q=256\lceil c/256\rceil-c+1\). Then
\[
E[B]=kE[\min(R,H)]+\sum_{s=q}^{H-1}P(R>s)
\]
when \(q<H\), otherwise just the first term. Survival cumulative sums evaluate this and (11) in \(O(1)\) per candidate. Precompute one unbounded rounded survival table through the public maximum cap \(U\); individual public caps cut off sums, so prompt-dependent caps do not require a table per request. Startup is \(O(U+n)\) time and memory; each schedule boundary is \(O(b)\), total \(O(U+n+Tb)\), plus \(O(n)\) quarantine state. Stable move-to-tail is \(O(b)\), not a full sort. Guard packing is a single FIFO pass, not a combinatorial search.

Hard planning allocation: 2 seconds for startup table/config work; reject an arm whose setup exceeds it rather than silently using a different estimator. Online decision work has a 1 ms per-call deadline and 5 seconds cumulative per run; after exhaustion use canonical FCFS plus the same guard and record the fallback count. Check deadlines every 32 candidates; commit only a completed decision, never a partially constructed permutation. These are admission deadlines for planner work, not a real-time OS guarantee against descheduling; a deadline overshoot is a recorded budget violation. Report measured startup and online overhead separately and include online overhead in makespan. An arm dominated by fallback is not evidence about the proposed scoring rule.

For the specified priors, cap survival is well away from floating-point underflow (about 0.0028 for SMALL and 0.0188 for LARGE). `erfc` and double-precision compensated cumulative sums suffice. Check computed probabilities against \([0,1]\) within \(10^{-12}\) and residual/relief bounds against their public caps. If a learned fit moves into severe tails, use log-survival and stable interval sums, not `1-CDF` or cancellation-prone mean-minus-age formulas. Reject invalid fits, retain the last valid completed-cohort model, and record this event.

## 6. Anti-thrash guard with a finite-completion proof

A fixed cooldown alone is **not** a termination proof: two large requests can alternate eviction and reconstruction forever after each cooldown. Merely reducing external dispatch is also insufficient when preempted requests are already in vLLM's waiting queue.

Use the following common guard in all attribution arms:

1. On the first actual preemption of a pressure episode, close fresh and waiting admissions. Quarantine every victim. Do not re-admit any victim while any member of the current running cohort remains. If more resident evictions occur, quarantine those too. The last feasible resident is not proactively evicted.
2. After the running cohort drains, serve quarantined requests in FIFO **protected packs**. Let \(U_i\) be a public upper bound on total decode length, independent of exact target; reserve \(K_i^{\max}=\lceil(p_i+U_i)/256\rceil\). Greedily form the FIFO prefix with total endpoint reservation at most 566. Admit only that pack, forbid further admissions, and let it finish. Repeat until quarantine is empty. A singleton is always possible under the individual-fit assumption.
3. Protected packs are never chosen as victims. This is safe only if the endpoint reservation accounts for actual allocator/in-flight overhead. If that invariant fails, stop the experimental arm as a feasibility/integration failure rather than silently ignoring a protected flag. Once quarantine empties, reopen normal admission.

The queue-eligibility gate must apply inside the scheduler to *already waiting* requests, not just the external dispatcher. A list-only victim permutation cannot enforce this guard. If the current wrapper has no eligibility hook, the orchestrator must add a proper pre-schedule holding queue with preserved request objects/status and exactly-once return to waiting; its loop's pending-work predicate must include held requests. Never cancel/recreate requests, lose generated output, or modify a completed `SchedulerOutput`. Quiesce the relevant asynchronous scheduling boundary before moving ownership of waiting entries. This is an explicit integration requirement, not something `throttle_prefills=True` is assumed to guarantee.

Termination proof, under finite \(n\), finite public length bounds, individual feasibility, bounded successful prefill execution, and fair progress of admitted feasible requests:

- In an episode's first drain, no request enters the running cohort. Each eviction strictly decreases its membership; an evicted member never returns during that drain. There are at most the initial cohort size minus one such evictions before a feasible singleton remains. Between evictions, only finite work remains; by fair progress that cohort drains.
- Every protected pack is feasible at every generated-token position by endpoint reservation and finishes after finite work. The finite quarantine shrinks by at least one completion per pack, so recovery ends.
- Any request evicted during normal operation is completed in protected recovery before normal admission resumes. Hence it is evicted at most once over the whole run. With finitely many requests, there are finitely many episodes, evictions and recomputations. All requests complete. \(\blacksquare\)

This is deliberately stronger than a performant cooldown and can lower batching during recovery. Do not attribute its gain or loss to victim selection. A less conservative guard needs a separate progress argument; “we have not observed thrashing” is not one. Public endpoint reservations also distinguish true safety from a probabilistic 128-step forecast.

## 7. Required validation before a victim-selection headline

### A. Two-request forced-preemption sign test

Run through the **installed scheduler on the orchestrator**, with deterministic outputs and a controlled one-block allocation shortfall; no third runnable request. At a quiescent decode boundary, arrange resident A and B with equal currently releasable block counts, both unfinished, each requiring a new block on this call. Choose the free-block budget so zero victims cannot satisfy both but evicting either leaves enough KV to schedule the other. Disable waiting admissions, speculation, prefix sharing and connector/deferred-free effects for this fixture; ensure the token budget exceeds both one-token demands. The test fixture, not Track O policy code, may know the actual targets.

First set score(A) < score(B), stable-move A to the tail, and call `schedule()` once. Assert exactly one reported preemption, victim A, survivor B scheduled exactly one token, A absent from the scheduled-token map, and actual free/block-account deltas consistent with the allocation. Rebuild an identical clean state, reverse the scores, and assert exactly B is evicted. Record pre-call running order, actual tail at pop, current allocation request, free blocks, block demand, victim ID, and full per-request scheduled map. Also exercise the self-victim path explicitly: the probe has a special branch for it.

The block-pressure fixture can use a reduced usable-pool test configuration or an allocator test hook; distinguish that scheduler sign fixture from a production hardware makespan arm. Do not infer correctness merely from a score printout or from an aggregate preemption count. If the engine's actual block convention moves the crossing, use allocator-reported demand to place the fixture; do not assume \(c\bmod256\) alone determines the next allocation.

### B. Side-effect audit: a gating test, not a foregone conclusion

Record/hash full `Scheduler` and `AsyncScheduler` method-resolution-order source in the orchestrator environment. Identify every `running` traversal, token-budget decrement/refund, KV failure loop, speculative/encoder skip, current-victim break, waiting-admission gate, and output update. The proof obligation for list-only F3 is restricted to a declared operating region:

- every running request requests exactly one useful decode token;
- budget covers all of them; no chunked prefill/recompute or speculative tokens;
- no request-specific blocking, deferred frees or cross-request KV sharing;
- at most one victim is required and every remaining request can be scheduled after its release;
- no waiting admission occurs on the preemption call; mutation is at a safe async boundary.

Under these conditions, all survivors can receive the same one token; stable move-to-tail changes their iteration positions but not their requested amounts. On a no-preemption call the entire scheduled-token map must be invariant. On a one-preemption call, *literal equality of maps with different victim IDs is impossible*: compare identical survivor maps and the one-victim replacement, and compare total scheduled tokens and batch cardinality. This is the precise meaning of “only victim changes,” not unchanged request IDs.

Use paired identical pre-call scheduler states, not successive calls on one mutated state. Compare order-preserving output maps, new/cached/resumed request sets, scheduled prefill/decode counts, total tokens, block deltas, final waiting membership, and `_update_after_schedule` state. Include all trace states in a short qualification run; any mismatch outside the intended victim delta disqualifies a victim-only label for that region. A finite test is evidence; the source-level conditions supply the general argument.

Actively test the counterexample region: two running prefills each request 1536 tokens with budget 2048 and ample KV. Forward order allocates 1536/512; reverse order allocates 512/1536 by request ID under a sequential capped-budget allocator. Totals agree, yet per-request token allocation differs. Also test decode plus recompute and token-budget exhaustion. If the installed build behaves differently, record its actual rule; it still must be inspected, not assumed order-independent. Mixed calls are expected to falsify a universal victim-only claim.

### C. Causal control, not merely another baseline

Preferred attribution uses independent traversal and victim controls, with **the same anti-thrash guard, gate, token budget, queue rules and logging** in every cell:

| cell | allocation traversal | actual victim selector |
|---|---|---|
| A | canonical FCFS | canonical FCFS tail |
| B | F3-permuted | canonical FCFS tail, explicitly forced independently of traversal |
| C | canonical FCFS | F3 score, explicitly forced independently of traversal |
| D | F3-permuted | F3 score |

C−A isolates victim selection under canonical traversal; D−B isolates it under permuted traversal; B−A measures traversal perturbation; the difference of those effects measures interaction. B is the essential sham-reordering control. It cannot be implemented just by moving the baseline victim back to the tail, because that changes the traversal being controlled.

Implement the independent selector at the scheduler's victim decision, including its existing bookkeeping for any already scheduled victim. The probe's priority branch demonstrates that such bookkeeping matters; changing only `.pop()` to remove an arbitrary already-scheduled request is unsafe. Do not claim a pure-victim result if no correct independent seam exists. In the restricted homogeneous-decode region, the audited invariance argument can justify a smaller A/D comparison, but the result must explicitly exclude mixed/unaudited calls. Add a no-preemption negative control: any systematic A/D difference there is a traversal or instrumentation effect, not victim improvement.

Hardware arm configuration: fixed identical manifest hash, TP16, 566 usable blocks, unchanged max-sequences/token-budget settings, arrival admission with cap256 (known to have nonzero preemptions), same native-prefill-brake setting throughout. Compare C/O score providers with \(H=128\), (12), the common guard and the same traversal controls; repeat on SMALL and LARGE. Also retain default-tail+guard versus default-tail without guard to quantify guard cost, separately from F3 attribution. Interleave repeated arms and report paired makespan differences with the observed cross-arm noise floor. Report useful output identity, prompt and recompute tokens, preemption count, unique victims, per-victim repeated evictions, KV freed, actual time-to-readmission, completion delay, final-tail duration, scheduled-token maps, eligible/preempted coverage, planning overhead and all budget/invariant failures.

Falsifiable predictions, not results:

1. The mixed-prefill counterexample/audit will reject an unconditional “reordering changes only the victim” assertion; pure one-token budget-slack calls should pass the restricted map/cardinality checks.
2. In the pressure-dominated middle phase of cap256, the C point-mass score should avoid more imminent-completion victims and produce more early useful KV relief than FCFS tail or a most-remaining rule without a horizon. Whether this decreases wall time is the hardware endpoint, not implied by relief alone.
3. O's conditional-survival score should outperform the fixed fewest-generated key on late attained-service pressure episodes away from the cap, measured by conditional relief and then whole-run makespan. Lack of whole-run gain is a possible outcome, especially with a small covered-preemption fraction.
4. The strong common guard should yield zero twice-evicted request IDs. Any repeated eviction of an ID falsifies implementation or its feasibility assumptions; it is not explained away by workload noise. Guard-induced loss of throughput is separately measurable.
5. A most-remaining-only victim arm can lengthen the terminal tail; the terminal-penalty arm should reduce that tail in controlled equal-cost two-residual fixtures. No fixed percentage makespan gain is predicted.

## Verification performed for this derivation

The constant-time block-growth identity was checked against direct survival-weighted sums over 24 age/context combinations; maximum absolute discrepancy was \(2.28\times10^{-13}\) block-steps. Formula (11) agreed with direct capped-PMF expectations across 36 age/horizon combinations to \(1.43\times10^{-14}\) steps. Both checks used a \(10^{-9}\) absolute tolerance. Direct subtraction of the flip example gave long-minus-short costs -480, 0 and 32 at \(\nu=0,3.75,4\), respectively, with \(\lambda=1\).

A standard-library Python calculation evaluated (6), (8) and (9), including the table above. Independent PMF summation including the cap atom agreed with survival-sum residuals at \(g=0,128,1024,4095\), with displayed absolute residual 0. Independent Simpson quadrature of the lognormal first moment agreed with (6) at \(g=32,256,2048\); maximum absolute discrepancy was \(5.28\times10^{-11}\) tokens (check tolerance \(10^{-8}\)). These are numerical checks, not proofs.

Separately, explicit finite growing-KV trajectories at \((c,r,h)=(1024,1,128),(1024,256,32),(257,600,100)\) all gave exactly zero signed lifetime occupancy difference under a pure time shift, corroborating the change-of-variables proof. Boundary checks: capped residual at \(C-1\) is 1, \(0<A_h\le h\), initial unbounded mean is \(e^{\mu+\sigma^2/2}\), and the terminal flip example has the stated threshold by direct subtraction. No implementation, scheduler audit, runtime performance or hardware validation is claimed by these formula checks.
