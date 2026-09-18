# Oracle information: conditional release laws, not a fixed missing 32 blocks

## Conclusions to carry into the experiment

- **The strongest legal Track O knows the workload law and conditions it on attained service.** At prompt/decode correlation zero, prompt length cannot rank decode lengths, but an unfinished request's generated count is still informative. Compare exact individual lengths against this opponent, not against an unconditional point estimate or a fixed watermark.
- **Knowing a decode length is not knowing a wall-clock or scheduler-step release time.** Exact releases additionally require a service schedule. Chunked prefill, skipped decode, and asynchronous progress can invalidate a release-credit admission proof in either track.
- **The existing watermark is an admission threshold, not a maintained reserve.** In `BK_BK_NA_watermark.json`, margin 32 produced zero preemptions, minimum free KV **2** blocks, mean free KV **202.9694**, mean scheduled batch **106.2816**, and measured wall time **345.6376 s**. It did not maintain 32 unused blocks.
- **Removing that threshold is not automatically worth 32 divided by mean request size.** For manifest `5f30f9fa9f`, exact useful-token block area is 2,462,156 block-steps under after-token allocation, so the relevant mean size is **3.41210 blocks per useful decode token**, not a per-request arithmetic mean. A continuously filled batch of 128 uses 436.749 average blocks, leaving 129.251 average blocks. Thus a hypothetical permanent 32-block reserve does not even bind the aggregate area bound at target 128. Peak packing and the terminal tail can still bind; their feasibility does not follow from this average.
- A first-passage calculation **does explain the scale of K=32 surprisingly well**: a frozen-hazard compound-Poisson approximation calibrated to this manifest gives K=32 for a nominal 1% whole-run overflow target. This is a check of a proposed formula, **not a safety certificate and not a hardware result**.
- There is no distribution-free mapping from Spearman correlation or MAE to oracle value. What matters is the **joint uncertainty of block-weighted releases at binding horizons**, including bias and correlation between prediction errors.

## 1. Information and objective

Let L=256 tokens/block, B=566 usable blocks, p_i be prompt length, d_i the total decode target, g_i the engine-observed number of generated tokens, and q_i the actually allocated blocks. Do not replace g_i with elapsed steps or computed-prefill counters. Pending prefill and pending recomputation reservations are additional live liabilities, not free space.

Track O's information sigma-field at a decision contains the public request features, past actions, engine-observed progress, actual allocation/free space, and actual completions. Track C additionally observes all d_i at construction. For an active request, define R_i=d_i-g_i>0. The future primitive supplied to the planner is

\[
 s_i(h)=\Pr(R_i\ge h\mid\mathcal I_t),\qquad h=1,2,\ldots.
\]

C supplies a point mass at its exact residual. O supplies a conditional survival distribution, not `estimated_d - steps_elapsed`.

The real optimization objective is measured end-to-end bulk wall time for completing every request of the identical manifest. For the restricted decode cost model, the planner minimizes

\[
 J=6.75T+0.0514\sum_t\operatorname{padded}(b_t),
\]

subject to all requests completing and live KV feasibility at every service epoch. The common term 0.292D can be omitted from this planning objective even when O does not know D: all policies ultimately perform the same realized useful decode work. No count times coefficient in this document is a measured makespan.

The count identity does **not** establish a physical 210.7-second irreducible wall-time floor. The coefficients were fit to median decode-only steps. Composition-dependent attention, mixed prefill, recomputation, host scheduling and different service clocks are separate channels. Use J as a candidate score; hardware measures the actual information value. BatchProfileTheory treats the cost-model qualification in more detail.

For genuinely optimal policies with the same action space and the same **ex-ante whole-run risk constraint**, information cannot hurt in expectation: C can ignore its extra information and execute any O policy. A constraint conditional on C's richer information is a different feasible-policy definition and must not be substituted into that argument. This does not imply that a particular clairvoyant heuristic beats its non-clairvoyant twin on every manifest. A comparison of different search algorithms is not an information-only measurement.

## 2. The exact margin object

First impose a **checkable service contract**: every ready decode resident advances once per logical decode epoch until completion; pending prefill receives no forecast release credit. The allocation/release timing below charges the next token's blocks before completion cleanup. Adjust the one-token convention to the installed engine; it changes the area slightly but not the argument.

For a proposed accepted resident set A, including a candidate and all pending liabilities, let

\[
 U_0=\sum_{i\in A}\left\lceil\frac{p_i+g_i}{L}\right\rceil,
 \quad a_i(h)=\left\lceil\frac{p_i+g_i+h}{L}\right\rceil,
\]
\[
 U(h;R)=\sum_{i\in A}a_i(h)\mathbf1\{R_i\ge h\},
 \qquad G_A(R)=\max\left(0,\max_{1\le h\le H}[U(h;R)-U_0]\right).
\]

H must cover complete drainage with **no additional admissions**, not just a 32- or 256-step lookahead. A public upper bound is max_i(d_cap,i-g_i), where d_cap,i=min(public workload cap, max_model_len-p_i). Pending requests must be treated according to the ready-service contract, with no speculative early release. The shorthand formulas above apply to ready decode residents.

**Exact gate:** with effective free space F=B-U_0 after reserving the candidate,

\[
 K_C=G_A(R_{\rm actual}),\qquad
 K_O(\alpha)=\inf\{k\ge0:\Pr(G_A\le k\mid\mathcal I_t)\ge1-\alpha\}.
\]

Admit if F>=K. C's K is not generally zero: deterministic future growth still needs space. The *uncertainty* cost is the difference in usable actions between these two applications of the same functional, not K_O itself.

**Safety proof.** If no further admission occurs, G_A<=F is exactly the trajectory capacity condition under the service contract. If each admission conditions on the entire history and has conditional failure probability at most alpha_j, the union bound gives whole-run failure probability at most sum_j alpha_j. Independence between admission decisions is unnecessary. For at most n one-time admissions, alpha_j=epsilon/n is sufficient. A failed risk event is not repaired by saying that the next decision will observe the failure earlier. \(\blacksquare\)

For a distribution-free zero-overflow guarantee, replace every R_i by its public maximum. U(h;R) is coordinatewise increasing in R, so this worst-cap trajectory is sufficient, and is necessary if all those endpoints remain jointly possible and all accepted residents must keep advancing. Reserving the sum of individual peak blocks is simpler but more conservative. With no finite length bound there is no finite universal margin permitting arbitrary simultaneous advancement.

**Counterexample to a one-step gate.** With B=3, two requests each at p=128,g=0 occupy two blocks. They are indistinguishable to O whether both targets are 1 or both are 256. In the short case the additional trajectory margin is zero; in the long case both need two blocks together at epoch 129, so the required margin is two and the state is unsafe. Checking the next allocation and later stopping *admissions* does not solve this.

If the engine can deliberately pause residents without discarding KV, there are other safe policies: one-resource Banker's safe-state tests using public maximum claims can reserve enough to finish requests in a safe serial order. Such policies trade batch concurrency for safety and refute any universal claim that O must maintain a particular fixed K. They are a different action-space experiment; they must not be silently granted to only one twin. Reordering a live list is not, by itself, an implemented pause mechanism.

### A computable certified upper envelope

Under conditional independence of individual lengths, set

\[
 \mu_h=\sum_i a_i(h)s_i(h)-U_0,\qquad
 v_h=\sum_i a_i(h)^2s_i(h)(1-s_i(h)).
\]

Let c_h be the maximum a_i(h) among requests with 0<s_i(h)<1, or zero if there are none. With m examined horizons and ell=log(m/alpha), Bernstein's inequality and a union bound give

\[
 K_{\rm cert}=\left\lceil\max\left(0,\max_h\left[
 \mu_h+\frac{c_h\ell}{3}
 +\sqrt{2v_h\ell+\left(\frac{c_h\ell}{3}\right)^2}
 \right]\right)\right\rceil.
\]

All integer h<=H are valid. It is enough to examine h=1 and every block-growth epoch: between them every sample path can only lose residents. With C point masses, v_h=c_h=0 and this **same code** reduces to the exact deterministic gate. Take the minimum with the known valid worst-cap margin to avoid a needlessly larger bound. Shared uncertain fitted parameters can correlate requests; in that case this independent-sum formula is not certified unless a separate simultaneous parameter-confidence bound is included.

A tighter option uses weighted-Bernoulli Chernoff tails at each h:

\[
 \Pr(U(h)\ge B+1)\le\inf_{\theta\ge0}
 \exp\left[-\theta(B+1)+\sum_i\log(1-s_i+s_i e^{\theta a_i})\right].
\]

Sum these bounds over horizons and require <=alpha. Use bisection on the convex derivative and log-sum-exp / log1p-expm1 evaluations. Do not silently call 32 or 128 sampled trajectories a certificate for alpha approximately 10^-5. Scenario samples are useful for the *objective*; a claimed tiny violation probability needs an analytical bound or adequate independent statistical certification.

## 3. A strong O release law and the hazard shape

For the actual rounded, lower-bounded, winsorized lognormal, let D=min(c,max(1,round(exp(mu+sigma Z)))), with Z standard normal. For an integer x,

\[
 S(x)=\Pr(D>x)=
 \begin{cases}
 1,&x<1,\\
 \overline\Phi((\log(x+1/2)-\mu)/\sigma),&1\le x<c,\\
 0,&x\ge c.
 \end{cases}
\]

Thus, for a live request with g emitted tokens,

\[
 s(h)=\frac{S(g+h-1)}{S(g)},\qquad
 \lambda(g)=\Pr(R=1\mid D>g)=1-\frac{S(g+1)}{S(g)}.
\]

This explicitly uses survival information at rho=0. Handle the cap atom: lambda(c-1)=1. Evaluate survival ratios with stable log-survival or erfc, not `1 - normal_cdf` in an extreme tail. Never normalize a zero-survival state; it is a state/contract inconsistency.

For the uncapped continuous law,

\[
 h(a)=\frac{\phi(z)}{\sigma a\overline\Phi(z)},\quad
 z=\frac{\log a-\mu}{\sigma},\quad
 \frac{d\log h}{d\log a}=-1+\frac{\phi(z)/\overline\Phi(z)-z}{\sigma}.
\]

The lognormal hazard rises and then falls. To justify uniqueness, write m(z)=phi(z)/overlinePhi(z): the derivative of m(z)-z equals minus the positive variance of a standard normal conditioned to exceed z. Also m(z)-z decreases from infinity to zero, by the normal-tail limits. Hence the derivative above changes sign exactly once. At median 256,sigma=1 its continuous maximum is near age **158.22**, not at the median. Numerical discrete next-token hazards are:

| generated g | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4095 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| completion probability next token | .001512 | .002619 | .003240 | .003109 | .002505 | .001797 | .001192 | 1 |

Consequences: young synchronized admissions initially provide little release credit; survivors far into the uncapped tail are not memoryless and should not be assigned the population mean residual; survivors at the hard cap are almost completely predictable. A memory controller should aggregate **block-weighted conditional releases**, not count completions equally.

### Which distributional knowledge is legal?

Use two clearly named O variants:

1. **O-law, the primary opponent:** public median, sigma, cap, prompt distribution/coupling specification, but no manifest PRNG seed or individual target. This isolates the value of exact individual lengths from the separate cost of learning a population law. At rho=0 the public law is already the correct conditional law.
2. **O-learn, a secondary deployment test:** a prior fixed before this manifest plus learning from completed requests only. Report its calibration/cold-start loss separately; do not label all of it irreducible individual information value.

The harness's `build_drawn_workload` uses a Gaussian latent coupling. With its public latent correlation r and prompt normal-score z_i computed from the known prompt ranking,

\[
 \log D_{i,\rm uncapped}\mid p_i
 \sim N(\mu+\sigma r z_i,\sigma^2(1-r^2)).
\]

Use the same rounding/capping and survival conditioning. The generator's parameter called rho is **not** the measured estimator's Spearman coefficient. Its ranking uses stable index tie-breaking; this is knowable from the prompt-only catalogue, without reconstructing any random draw. If distribution parameters are not designated public, this O-law arm must be reported as a known-population-law benchmark rather than silently substituting it for O-learn.

**Completed-only fitting has a selection trap.** In the matched rho=.5 artifact, the estimator has Spearman .458835 and MAE 265.657, but also mean error **-153.361 tokens** and five preemptions at the same K=32. This is consistent with serious underprediction; the artifact alone does not prove its cause. Training on whichever requests have finished first can overrepresent short jobs. A safe simple calibration design selects a fixed prompt-stratified cohort by a hash independent of lengths *before* execution and performs one fit only after that entire fixed cohort has completed. No unresolved cohort is skipped in favor of an earlier-finishing one. A completed-sample snapshot at an arbitrary time is not automatically an unbiased length sample. Do not fit a censored-data estimator on active requests if the stated track permits fitting on completed requests only; live g may still condition predictions from an already fitted law.

## 4. Why K=32 is a plausible scale, and why it is not portable

For an explanatory short-time approximation, freeze a representative busy state's growth rate c_b=b/L and completion rate Lambda=b/E[D]. Let completed requests release independent block marks Q and approximate completions by a Poisson process N(t). Additional occupancy is

\[
 X(t)=c_bt-\sum_{j=1}^{N(t)}Q_j.
\]

If Lambda E[Q]>c_b, there is a positive root theta of

\[
 c_b\theta+\Lambda(E[e^{-\theta Q}]-1)=0,
 \quad\text{equivalently}\quad
 \frac{\theta}{L}+\frac{E[e^{-\theta Q}]-1}{E[D]}=0.
\]

The compound-Poisson exponential martingale and Ville's inequality give

\[
 \Pr(\sup_{t\ge0}X(t)\ge K)\le e^{-\theta K}
\]

**for that frozen stochastic process**. Therefore the corresponding design heuristic is

\[
 K_{\rm fluid}=\left\lceil\frac{\log(n/\epsilon)}{\theta}\right\rceil.
\]

This shows why K is not a function of decode variance alone: terminal block marks depend on prompts and decode, while live nonstationary versions depend on ages and block phases. Replacing Q by its mean loses important release-size variability. If the frozen drift is nonnegative, there is no positive adjustment root; use the finite-horizon trajectory calculation, not this formula.

**Check against existing data, not a fitted hardware result.** The shared exact admission list for manifest 5f30f9fa9f has E[D]=721595/1763=409.29949 and terminal mark Q=ceil((p+d)/256) with mean 3.044810. Its Q:count histogram is

`1:383, 2:538, 3:339, 4:184, 5:124, 6:88, 7:40, 8:19, 9:9, 10:11, 11:8, 12:1, 13:5, 14:3, 15:3, 16:3, 17:5`.

Substituting that empirical mark distribution gives theta=**0.37833277053363135**, root residual below 1e-18. For n=1763,epsilon=.01,

\[
 \frac{\log(1763/.01)}{\theta}=31.9294,
 \qquad 1763e^{-32\theta}=0.00973647.
\]

So the proposed formula selects **K=32**, consistent in scale with the measured zero-preemption watermark run. No claim of calibration follows: synchronized young cohorts, changing b, non-Poisson completions, age-dependent marks, block-boundary bursts, mixed prefill and adaptive admissions violate the frozen process. One zero-preemption run cannot establish a 1% failure bound. These empirical lengths are used here for retrospective mathematical checking only, not passed to O.

For a fast *uncertified* feedback approximation, replace Lambda with sum_i lambda_i(g_i), use completion-probability-weighted current block marks, and estimate c_b over the next 256 tokens from the actual block phases. Recompute the adjustment root when the active composition changes. The certified trajectory gate above remains the primary matched test; this `fluid` approximation is a separately labeled lower-compute arm.

**W-LARGE sensitivity, entirely analytical:** with median 1024,sigma=1,cap8192, the rounded prior mean is 1605.5144. Applying this frozen formula to illustrative constant prompts p=512,768,1024 produces K=**209,157,131** respectively for the same n and epsilon. These are not recommended fixed margins or forecast hardware outcomes. They show that blindly carrying K=32 into W-LARGE is not justified. C's deterministic growth reserve also grows, so none of these K values is itself the oracle gap.

## 5. Residency and makespan: the correct conversion

For a no-preemption decode trajectory, define useful-token block area

\[
 A=\sum_i\sum_{k=1}^{d_i}\left\lceil\frac{p_i+k}{L}\right\rceil,
 \qquad \bar q=\frac{A}{D}.
\]

This is invariant under ordering. If every resident advances every epoch, average occupancy is A/T=bar_q times mean_b. If residents stall while retaining KV, add their idle block-time divided by T. Thus an additional **time-average unavailable capacity** Delta M produces approximately

\[
 \Delta\bar b\simeq\frac{\Delta M}{\bar q}
\]

only while memory is the binding constraint, packing efficiency and idle residence are comparable, and no batch cap is binding. A threshold K at admission is not Delta M. Approximately,

\[
 \bar b\lesssim\min\left(b^*,\frac{B-M-\overline U_{\rm idle}}{\bar q}\right).
\]

For an idealized common padding bucket P and nearly constant batches b_C and b_O=b_C-Delta M/bar_q,

\[
 \Delta t_{\rm pred}
 =\frac{D(6.75+0.0514P)}{1000}
 \left(\frac1{b_O}-\frac1{b_C}\right)\quad\text{seconds}.
\]

For actual varying profiles, use 6.75(T_O-T_C)+.0514(sum padded_O-sum padded_C); substituting padded(mean_b) is invalid. Crossing a bucket can even reverse a mean-batch improvement's model benefit.

**Conditional sensitivity table, not expected or measured outcomes:** D=721595,bar_q=3.41210236,b_C=128,P=128, with every extra unavailable block assumed to reduce batch.

| extra average unavailable blocks Delta M | idealized b_O | model-only penalty |
|---:|---:|---:|
| 8 | 125.655 | 1.402 s |
| 16 | 123.311 | 2.857 s |
| 32 | 118.622 | 5.941 s |
| 64 | 109.243 | 12.902 s |

The 32-block row is about 1.74% of the **measured reference** 340.8 s, not a prediction that C will beat that reference by 1.74%. In fact (B-32)/bar_q=156.502>128, so a static 32-block reserve does not preclude the target-128 area budget. The measurable variables needed to turn this sensitivity into a prediction are extra time-average free KV, idle-resident block-time, packing failures at admissions, and the terminal tail.

Long lengths can also help C by **starting otherwise unidentifiable long tail requests early**, not just by predicting releases of current residents. At rho=0, queued requests have exchangeable unknown lengths conditional on prompts; live age information cannot identify a long request that has never run. This is a genuine remaining information gap even for O-law. A fixed margin analysis misses it.

## 6. Estimator quality is a release-calibration problem

### No universal rho/MAE-to-K function

Spearman correlation is unchanged by strictly increasing rescaling of predictions: perfect ranking can coexist with catastrophically wrong release times. MAE does not specify upper-tail bias or dependence. An error of magnitude e on every request and an error of e/eta on an eta fraction have the same MAE and radically different reserve needs. Common-mode errors can make uncertainty scale like b instead of sqrt(b).

Nor can one divide MAE by L: predicting a completion one step too early can lose credit for the **whole q-block context**, not merely one token's block increment. Without more assumptions, Markov's inequality only gives P(|error|>x)<=MAE/x, which is far too weak near a binding release time.

The correct empirical target is calibrated s_i(h), or equivalent one-sided release-time quantiles, evaluated at the ages and horizons where the gate binds. Report probability calibration and block-weighted delayed-release errors there, including their simultaneous dependence. Global Spearman and MAE remain diagnostics, not safety specifications.

### Useful conditional mappings, with their assumptions visible

For an ideal continuous Gaussian copula with *population* Spearman rho_S,

\[
 r=2\sin(\pi\rho_S/6),\qquad
 \sigma_{\rm conditional}=\sigma\sqrt{1-r^2}.
\]

Plug that residual log-scale into the conditional survival and K formulas; do not replace K by K_0(1-rho_S). At rho_S=.459, the residual standard-deviation factor is **.8794**: useful ranking, but only a 12% reduction in log-scale uncertainty in this ideal model. Factors at .90,.95,.99 are .4190,.2988,.1345. The measured online predictor's rank coefficient is not automatically the copula parameter; these numbers are an explanatory model, not inferred achieved reserves.

Under an uncapped conditional lognormal with optimal conditional-median point prediction, the conditional relative MAE is 2Phi(sigma_conditional)-1. Capping and bias change this relation. In particular, MAE 266 on roughly 409 tokens is nowhere near exact endpoint prediction.

A more local mapping exposes the release discontinuity. Assume independent centered Gaussian residual-time errors with standard deviation tau, predicted release centers having a smooth density f_R(h) near a binding horizon, approximately common released size q_rel, and small tau. Then

\[
 v_h\simeq b q_{\rm rel}^2 f_R(h)\frac{\tau}{\sqrt\pi}
 =b q_{\rm rel}^2 f_R(h)\frac{e}{\sqrt2},
 \quad e=\operatorname{MAE}=\tau\sqrt{2/\pi}.
\]

The identity used is integral Phi(u)(1-Phi(u))du=1/sqrt(pi). Thus the Gaussian fluctuation component of the margin scales approximately as

\[
 \Delta M_{\rm uncertainty}\simeq
 z q_{\rm rel}\sqrt{b f_R(h)e/\sqrt2},
\]

not linearly in MAE. Add biased delayed-release demand and a simultaneous-horizon correction; z is a specified quantile, not a universal constant. This small-error expansion is not justified automatically at e=266.

**When does O close the gap?** Precisely for a given common candidate/action sequence, when its calibrated upper envelope fits within every exact-C accepted state's available slack, so both gates choose the same actions. Survival uncertainty can remain large away from binding horizons. No universal Spearman threshold exists; even nonzero MAE can give identical decisions, while a one-token error at a zero-slack release can change one.

For a model-time tolerance delta seconds within one bucket, the allowable extra *binding* margin is

\[
 \Delta M\le
 \frac{\bar q b_C^2\delta}
 {D(6.75+.0514P)/1000+b_C\delta}.
\]

In the small-workload sensitivity above, a 1-second tolerance corresponds to **5.736 blocks**, and 1% of the measured 340.8-second reference corresponds to **18.949 blocks**. In the local Gaussian release model, substitute these into

\[
 e\lesssim\frac{\sqrt2\,(\Delta M)^2}
 {z^2q_{\rm rel}^2 b f_R(h)}.
\]

This identifies an accuracy target only after measuring the binding horizon density, block marks, bias, dependence and available slack. Those quantities are missing from the two reported estimator summaries; manufacturing one numerical “gap closes at rho=X” would not be justified.

## 7. Implementable matched planner: law-aware feedback with a common gate

Use the OfflinePlanner calendar/search family in both tracks, not a bespoke C optimizer against a fixed-watermark O. The common search objective is J above; its probabilistic O evaluation uses draws from the conditional law, not the real hidden manifest. Future draws used during planning are internal hypothetical samples and must never be reported as experiments.

### Shared planner contract

- `LifetimeProvider.profiles(public_queue, observed_state)` returns immutable conditional lifetime profiles. C returns point masses, O-law returns the exact public rounded/capped law conditioned on p,g. O-learn returns its precommitted completed-only fit. Everything below this interface is shared.
- Common target b*: test **128 on W-SMALL** and **64 on W-LARGE**, with matched 64 and 32 target controls respectively. These are target ceilings, not claims that they are sustainable or optimal. Do not raise C's target alone.
- Common candidate pool size 24, calendar seed, relocate/swap/tail-pair proposal family, RNG stream, tie-breaking and drain objective. Candidate values can depend on the profile; both tracks use the same expressions. Queue items never receive true-d sorting in the common code.
- Common exact effective-free accounting, including pending prefills/recomputation and allocation timing. No filtering of SchedulerOutput after schedule().
- Common risk target epsilon=.01 and alpha=epsilon/n per committed admission for the certified arm. The risk bound applies only with a correct conditional law and the verified service contract.
- No release credit for pending prefills before they meet the ready-service contract. Reserve each pending request's endpoint blocks at the supremum of its provider's support until it becomes ready (exact endpoint in C, public cap in O); this reservation is part of every candidate certificate. Release that excess reservation only when replacing it by a valid ready-state trajectory certificate. If this contract cannot be enforced or verified, the zero-preemption claim is withdrawn; use the same conservative endpoint/pending treatment in both tracks, or explicitly model bounded service skips with a separately justified bound. It is not safe to accept a release-credit timetable and retroactively switch to endpoint reservation when it is already infeasible.

### Pseudocode

```text
construct(track, public_records, public_law, runtime_observation_port):
    assert_exact_public_schema(public_records, track)
    provider = PointMassTargets(C_private_targets) if track == C else ConditionalLaw(public_law)
    # Private targets are held only by the C provider, never common policy objects.
    risk_per_admission = 0.01 / n
    plan = common_calendar_search(provider, common_config, deadline)
    return shared_policy(plan, provider, risk_per_admission)

before_engine_schedule():
    state = observation_port.snapshot()  # real g, allocated blocks, readiness, pending liabilities
    check_ready_epoch_contract(state)    # correctness precondition, not a post-hoc assertion of safety
    incorporate_actual_completions(state)
    provider.condition_on_survival(state)
    # Learning updates, if enabled, consume approved completed-cohort outputs only.

    if recompute_due_or_cache_invalid:
        profiles = provider.profiles(public_queue, state)
        cache = aggregate_future_mean_variance_and_growth_events(profiles, state)
        common_receding_calendar_repair(plan, cache, remaining_planning_allowance)

    while pending_queue and resident_count < b_star:
        candidates = common_candidate_pool(plan, profiles, count=24)
        # Score expected remaining J; use common forecast samples only for this score.
        candidates = common_score_and_tie_break(candidates)
        accepted = false
        for candidate in candidates:
            if current_or_next_allocation_not_feasible(state, candidate): continue
            # Covers full no-new-admission drainage, not merely the next refresh window.
            certificate = common_tail_envelope(cache, candidate, risk_per_admission)
            if certificate.valid and certificate.peak_blocks <= B:
                dispatch(candidate)
                update_pending_liabilities_and_cache_without_rereading_stale_free_space()
                record_admission_certificate_and_risk_charge()
                accepted = true
                break
        if not accepted: break
    execute_engine_schedule_unchanged()

when_queue_empty():
    # Same rule in both tracks; no maintained admission watermark in the drain.
    stop_admission_search()
    continue_ready_decode_service_to_completion()
```

The initial and repaired calendar is a contingent admission plan: never dispatch an unsafe item just because its planned timestamp arrived. Both tracks rebase forecasts on the actual state. The gate certifies the accepted set's complete drainage, so merely reaching the next refresh time is not the safety argument. The controller reacts to the hazard profile of the actual age mixture and knows the exact block phase, unlike a fixed K or population-mean predictor. This is a strong legal contender, not a claim of Bayes-optimality. Bernstein/union-bound looseness can still reduce O's utilization; a measured gap for this implementation is not a lower bound on the gap against optimal O. Report the bound's looseness using exact small-state laws or independently validated release-tail diagnostics before interpreting it as unavoidable uncertainty.

### Complexity and hard limits

Let n be requests, T logical epochs, b maximum simultaneous residents, H maximum residual token horizon (4096 or 8192 here), k=24 candidates, S=32 fixed objective scenarios, E completed search proposals, and R runtime aggregate refreshes.

- A direct full calendar evaluation is O(Tb+n log n), or an equivalent exact event sweep. Objective sampling multiplies this by S; C's identical point-mass scenarios may be collapsed rather than computed redundantly.
- The simple all-horizon certificate costs O(Hb) to aggregate a resident set and O(H) per cached candidate addition. A straightforward implementation costs O(RHb+n k H) beyond search; memory is O(Hb+Sn+n) if profile tables are cached, reducible by sharing common-law tables and accumulating moments. Do not claim this CPU work is free.
- **Offline hard budget:** 55 seconds for n=1763, 110 seconds for n=3526, including survival-table setup and search; seed allocation at most 15/30 seconds. Cache the best fully evaluated valid policy at every boundary. No unbounded annealing or exhaustive action-tree search.
- **Runtime hard budget:** aggregate refresh normally every 32 verified decode epochs and on state deviations that invalidate a certificate; at most 4 ms of optional search/certificate rebuilding per refresh and at most 2 seconds cumulatively per run. Per-step exact occupancy accounting remains mandatory. If a new certificate cannot finish in budget, do not admit from an unfinished calculation; continue the already certified drain and use a cheap worst-cap endpoint certificate for new admission when possible. Always finish the workload. Record fallback frequency; a result dominated by this fallback measures implementation/CPU limits, not a tight information gap.
- Match the number of completed search proposals and scenario count, not just an ambiguous “same 55 seconds.” Preselect a common completed-proposal prefix feasible for both within the deadline. Collapsing C's duplicate scenarios does not entitle it to extra proposals in the information-only contrast.
- Report planning time outside wall-time measurement; runtime planning/dispatch overhead stays inside measured wall time. A 2-second online controller costs a material fraction of a predicted small-workload gain.

A separately labeled `IG-fluid` arm may replace the certified envelope with the age-weighted adjustment-root approximation for lower CPU cost. It has no whole-run probability guarantee under the actual workload. Compare matched C/O implementations of the same finite-horizon gate for the primary information estimate; do not silently compare exact C against this different O gate.

## 8. What makes the twin information-only

Required equality: workload SHA and D; prompt tokens; all engine/model/KV/padding/token-budget settings; warmup; prefill admission and throttle settings; target b*; preemption/victim behavior; readiness contract; risk definition; candidate/search family and completed proposal count; gate implementation; runtime refresh/budget rules; all tie-breaks; initial queue order; drain handling; measured timing boundaries and emitted-work accounting. The only policy capability difference is the lifetime provider's knowledge of individual targets.

O's construction proof needs a **whitelist**, not only `assert 'd' not in row`:

```text
assert all(keys(record) == PUBLIC_REQUEST_SCHEMA for record in policy_records)
assert no target/decode/max_tokens/oracle_d_sort/manifest_seed capability exists
assert observation_port exports only observed progress/allocation/readiness/completions
assert estimator_input_type is CompletedObservedRequest, never ManifestRequest
assert prediction_ledger and engine SamplingParams objects are unreachable from policy
```

Keep opaque ID, p, public arrival metadata and approved prompt features. Public d_cap is a common upper bound, not each request's `max_tokens`. Reject arbitrary objects whose properties or closures retain a manifest reference. The common policy's runtime-state DTO must be a copied/immutable whitelist, not a scheduler Request that exposes SamplingParams. A field-presence assertion alone cannot prove the absence of hidden references; the restricted constructor/capability boundary supplies that guarantee. The current `policy_view` stripping d is a useful start, not the complete proof for a richer planner.

The engine-side executor may still read d to set max_tokens, outside the O policy boundary. Never expose the manifest random seed to O: public generation law is allowed, reconstruction of its individual pseudorandom draws is oracle access by another route. Record observed completed lengths to train O only after completion, not from the scoring ledger.

Compare **C-matched versus O-law-matched** to estimate individual-length information. Report **C-best-offline versus C-matched** separately as extra search/mechanism headroom; its difference against O is not purely information. Also compare O-law versus O-learn, and O-law versus K32, to isolate population-learning and feedback gains. Changing the generator's rho changes the joint workload and E[pD]; compare twins within each manifest, never attribute between-manifest wall-time changes solely to estimator accuracy.

## 9. Falsifiable predictions and diagnostic arms

All ranges below are **unmeasured predictions**, not analytical benchmark results. Use paired manifests and repeats sufficient to estimate the paired noise floor.

1. **W-SMALL, sigma=1, rho=0, 10x/20x backlog:** test `IG-C-lawgate-cap128` against `IG-O-publiclaw-lawgate-cap128`, same certified gate and calendar search. Prior prediction: C's average advantage over this stronger O is **0–3%**, rather than the large residency gain suggested by comparing greedy LPT with K32. The sign can vary across heuristic runs. A reproducible advantage above 3% should correlate with reduced terminal low-batch steps or peak-packing improvements, not simply “32 blocks recovered.” Offline planning can reveal a positive gap missed by pure sorting, but the aggregate 128-target area slack makes a large mandatory reserve penalty implausible.
2. **O-law versus fixed K32:** same cap128, same candidate/search machinery, changing only the gate ablation. Prediction: O-law needs smaller reserves in high-hazard or cap-near states and larger reserves in young synchronized states; a constant reserve will misallocate slack in both directions. Measure conditional gate thresholds, minimum free space, overflow/preemption events and the full batch/padding profile. This is a mechanism comparison, not the information-only twin.
3. **Estimator signal:** repeat the exact twin at generator r=0 and .5; within each manifest compare population-only O-law with prompt-conditioned O-law. Prediction: .459-ish rank quality is useful but does not eliminate conditional release uncertainty; calibrated O-law beats a biased point-estimate trajectory policy on risk at matched admission aggressiveness. A zero oracle gap can nevertheless occur because both fit below a padding target; it would not establish near-perfect length prediction.
4. **W-LARGE, prompts >=512, median1024,sigma1,cap8192:** paired targets64 and32. Prior prediction: C's advantage over strong O is more often positive, roughly **2–8%** under sustained KV pressure and enough backlog to offer packing alternatives; it is not guaranteed merely by larger d. As sensitivity rather than a baseline forecast, bar_q≈9–12, b_C≈48–56 and an extra binding O margin of32–64 blocks give approximately **2–7% model-time penalties**. Larger margins or tail failures can exceed that range. Actual joint prompt/decode statistics and hardware wall time decide it. The frozen-law K estimates above predict that K32 will be much less reliable as a safety control here.
5. **Conditions likely to maximize information value:** moderate-to-large dispersion with many distinct, uncertain endpoints; high KV pressure at the desired padding target; enough pending requests for C to exchange long/short and large/small trajectories; lower b so fewer independent completions average out release uncertainty; and terminal long jobs whose length is not predictable from prompts. Sweep sigma=.3,1,1.5, workload scale and 2x/10x/20x oversubscription, always pairing manifests. Do not predict monotonicity to arbitrarily large sigma: winsorization creates a predictable cap atom, and the token-work/area composition changes.
6. **Vanishing-gap controls:** deterministic known d; or sufficient KV and sequence slots to admit every request and run the same simultaneous service trajectory. These should give identical gate actions apart from numerical/tie-break errors. If length bounds scale while B stays fixed, the relative gap can increase; if B scales with the workload so the same trajectories remain feasible, that argument disappears. At fixed bounded lengths, purely terminal O(d_cap) drain effects become a smaller fraction as n grows; larger oversubscription does not by itself guarantee a larger percentage oracle gain.

Record per decision: actual g/allocation, pending liabilities, block-weighted conditional completion probabilities, certificate horizon/peak, risk charge, candidate rejected reason, planning/fallback time. Record per epoch: useful b, padding, mixed-prefill status, actual KV, idle resident block-time, and ready-service violations. Record separately the suffix after the queue empties. These diagnostics distinguish information, tail ordering, host overhead and a failed cost/service model.

## Verification and artifact boundary

- Read the required context/goal, the actual `policy_view`, estimator, live progress/pending-prefill gate and workload-generation sections, `plan_buckets.json`, `plan_rho_matched.json`, and the BK/RHOm artifact summary sections. The harness changed during concurrent work; symbol semantics, not fixed line numbers, are cited. No benchmark harness was edited and no TPU/Ray job was submitted.
- Manifest area and terminal-mark histogram were extracted by OfflinePlanner from the existing BK cap128 admission list and shared. The mark histogram sums to1763; its mean and the adjustment root were independently recomputed here.
- Numerical mathematics only: exact discrete survival sums and PMF-weighted means agreed within 5e-13 tokens for both workload laws; probability masses were nonnegative and summed to one within 2e-16. The continuous capped moment agreed with rounded discrete means within 3e-7 tokens. The lognormal hazard maximum was searched on a dense deterministic grid; its rising/falling values and cap atom were checked separately.
- Independent derivation check: numerical quadrature of Phi(u)(1-Phi(u)) agreed with1/sqrt(pi) within2e-16. The two-request identical-information counterexample was enumerated directly and gave margins0 and2. On a separate exact three-request, four-point-lifetime example, all64 degenerate C certificates equaled directly enumerated path maxima; the stochastic certificates satisfied their stated bounds at alpha=.01,.1,.3. In that small example they selected the worst-cap margin, directly demonstrating that certification conservatism is real. Makespan sensitivity rows were computed directly from the two reciprocal batch costs; signs, units and positivity were checked.
- Existing artifacts establish K32's observed behavior, the two estimator diagnostics, D and the exact area inputs. They do **not** establish conditional completion independence, a lockstep service guarantee, cold-start estimator calibration, a stationary Poisson release process, or any W-LARGE outcome. Those are stated assumptions or proposed hardware checks, never claimed measurements.
