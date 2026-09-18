# Batch-profile theory: exact surrogate optimum, conditional KV bounds, and model falsification

## 1. The useful correction: the measured regression is not a hardware lower envelope

Let \(P(0)=0\) and, for integer \(1\le b\le512\), let \(P(b)\) be the least element of \(\{8,16,32,64,128,256,512\}\) not smaller than \(b\). Write \(\ell=0.292\), \(a=6.75\), \(c=0.0514\), and \(h(b)=a+cP(b)\), in milliseconds. All seconds calculated below are **values or bounds of this surrogate, not measured makespans**.

The exact mathematical problem associated with the proposed identity is
\[
 M_{\rm model}=\ell D+\sum_{t=1}^{T}h(b_t),\qquad \sum_{t=1}^{T}b_t=D.
\]
A fitted conditional median is not a pointwise lower bound. In particular, neither \(M_{\rm wall}\ge0.292D\) nor a 62% irreducible fraction of hardware makespan follows from the supplied evidence. The identity fixes the number of useful output events; it does not fix mixed-step input-token shapes, prefill/recompute work, host overhead, or nonlinear context-dependent costs.

**Artifact-supported scope warning.** `BK_BK_cap128.json` reports 340.802530 s, \(D=721595\), 8068 executed iterations, mean scheduled requests 89.440878, and zero preemptions. Its minimum free KV is 84 blocks and its mean free KV is 260.482524, so a cap of 128 did not itself identify the KV limit. OfflinePlanner's full-artifact extraction found:

- sum scheduled requests = 721609, differing from emitted outputs by 14;
- 1777 multi-token request occurrences comprising 429858 scheduled input tokens;
- 719832 single-token request occurrences;
- the first iteration schedules 9 requests and 2048 input tokens, with zero single-token requests and 8 emitted outputs.

Thus the supplied 0.04%-accurate count identity is entirely compatible with substantial mixed prefill/decode work. Do not apply the decode-only median regression to that work without validation. The extraction is an artifact calculation, not a new hardware run. The calibration artifact `COST_shapes_tp16.json` also begins with a different manifest, \(n=420,D=172072\), whereas the target BK artifact has \(n=1763,D=721595\); histogram transfers require matching workload hashes and trace scopes.

A useful replacement is a jointly measured step-cost function
\[
 \tau_t=F(P^{\rm req}_t,P^{\rm tok}_t,u_t,v_t,
            C^{\rm full}_t,C^{\rm slide}_t,\text{mixed}_t)+\epsilon_t,
\]
where \(u_t\) counts useful decode/output events, \(v_t\) counts prefill and recomputed input tokens, the two \(P\)'s are actual runner shapes, and the two \(C\)'s summarize full and sliding contexts. Use shape-stratified empirical costs or a small calibrated regression, selected by held-out whole-run error, not median-per-batch \(R^2\). Existing shape records include scheduled tokens, decode-only status, context sums, sliding context sums, and padded input/request shapes. Separate missing fields must be collected on the testing arms below. A proposed fit remains a surrogate, not a proved lower envelope.

There is a further invariant: for one-token decoding, the useful full-context token work is
\[
 \sum_i\sum_{g=0}^{d_i-1}(p_i+g)
 =\sum_i\left(d_ip_i+\frac{d_i(d_i-1)}2\right),
\]
and \(\sum_{i,g}\min(p_i+g,1024)\) is also order-invariant. This is a finite-sum identity. Therefore merely adding *additive* context costs does not create an ordering advantage; nonlinear shape/composition effects, stalls, or recomputation must supply that advantage. Conversely, fixed \(\{b_t\}\) does not imply fixed hardware time if those effects are present. For example, a hypothetical cost term \(b\max_i(p_i+g_i)\) distinguishes homogeneous and mixed-context batches with identical batch counts. Whether the TPU exhibits such an effect is a falsifiable hypothesis, not an asserted kernel property.

## 2. Correct variable-horizon formulation and exact profile solution

Initially ignore request precedence, prefill, and KV trajectories. This is a relaxation, not an executable request schedule. Assume \(D\) is a nonnegative integer. No zero-work steps are beneficial. With \(n_b\) the number of steps of size \(b\), the exact optimization is
\[
 H(D)=\min_{n_1,\ldots,n_{512}\in\mathbb Z_{\ge0}}
       \sum_{b=1}^{512}n_bh(b)
 \quad\text{subject to}\quad \sum_{b=1}^{512}bn_b=D,
 \qquad T=\sum_b n_b.
\]
Thus \(T\) is a decision variable. A Lagrangian with a previously fixed \(T\), or substitution of \(P(\bar b)\) for mean padding, changes the problem. Exactly,
\[
 M_{\rm model}=\ell D+D\frac{a+c\overline{P(b)}}{\bar b},
 \qquad \bar b=D/T.
\]
In general \(\overline{P(b)}\ne P(\bar b)\).

**Dyadic covering theorem.** Replace a step of actual size \(b\) by capacity \(P(b)\). Then \(H(D)\) is the minimum of
\[
 a\sum_q z_q+c\sum_q qz_q,
 \qquad \sum_q qz_q\ge D,\quad z_q\in\mathbb Z_{\ge0},
 \quad q\in\{8,16,\ldots,512\}.
\]
In an optimal cover there is at most one copy of each \(q<512\).

*Proof.* A profile gives a capacity cover at identical cost. An optimal cover has at most \(D\) steps when \(D>0\): otherwise deleting steps until \(D\) remain still leaves capacity at least \(8D\). Assign \(D\) positive integer token counts within its step capacities; actual padding cannot exceed the chosen capacities, so the profile optimum is no greater than the cover optimum. This proves equality of optima. Two copies of \(q\le256\) can be replaced by one \(2q\), preserving capacity and saving exactly \(a>0\); hence duplicates below 512 are impossible in an optimal cover. \(\blacksquare\)

For any fixed total capacity \(S\), a multiple of 8, repeated merging gives its minimum step count
\[
 m(S)=\left\lfloor S/512\right\rfloor+
       \operatorname{popcount}\big((S\bmod512)/8\big).
\]
Consequently an exact general implementation need evaluate at most 64 capacities between \(8\lceil D/8\rceil\) and \(512\lceil D/512\rceil\), minimizing \(am(S)+cS\). This handles the discontinuities without convexifying them.

For the specific coefficients there is an even simpler closed form. Define \(k=\lfloor D/512\rfloor\), \(r=D-512k\), and interpret the cost of an empty tail as zero:
\[
 H(D)=k h(512)+
 \begin{cases}
 0,&r=0,\\
 h(r),&1\le r\le256,\\
 h(256)+h(r-256),&257\le r\le320,\\
 h(512),&321\le r\le511.
 \end{cases}
\]
A canonical profile consists of \(k\) full 512 steps and the indicated one- or two-step remainder. Step order is irrelevant only in this relaxed surrogate problem.

*Proof of the coefficient-specific cutoffs.* The total capacity of all distinct buckets below 512 is 504, so every optimum has at least \(k\) full 512 buckets; more than \(k+1\) can be reduced. A residual of at most 256 is cheapest in one bucket. Indeed, using two or more smaller buckets can save at most 120 padded slots relative to its next bucket, worth \(120c=6.168<a\). A residual 257–320 is cheaper as 256 plus at most 64: its cost is at most \(h(256)+h(64)=29.948\), below \(h(512)=33.0668\). A residual at least 321 covered without a 512 bucket costs at least \(h(256)+h(128)=33.2376>33.0668\). These comparisons exhaust the dyadic normal forms. \(\blacksquare\)

**The constant-power-of-two assertion is false.** At \(D=257\), sizes \((256,1)\) cost 27.0696 ms of overhead, whereas a single 257 step costs 33.0668 ms. At \(D=384\), the single non-power-of-two step is optimal, costing 33.0668 ms versus 33.2376 ms for \((256,128)\). Repeated constant sizes need not be optimal either: at \(D=768\), \((512,256)\) beats \((384,384)\).

Here is an exact classification of *strictly constant* profiles for the relaxed problem with maximum 512. Let \(D=mb\), where \(m\ge1\) and \(1\le b\le512\) are integers. Such a profile is optimal if and only if one of these holds:

1. \(m=1\) and \(b\le256\);
2. \(P(b)=512\), \(m=\lceil D/512\rceil\), and \(D\bmod512\) is either 0 or at least 321.

For case 2 with remainder 0, necessarily \(b=512\). Non-power constants such as two steps of 511 also satisfy case 2. Necessity follows from merging repeated lower buckets and the residual formula; sufficiency follows by direct substitution. \(\blacksquare\)

For real requests, also require per-request precedence and one output per request per step, so \(T\ge\max_i d_i\); resident KV feasibility must hold at every allocation. None of the fungible-token constructions above proves those extra constraints feasible.

## 3. Constant-target optimum, all padding minima, and sensitivity

A sustained constant target, ignoring its final partial step, has the analytical objective
\[
 f_D(b)=D\left[\ell+\frac{a+cP(b)}b\right],\qquad 8\le b\le512.
\]
Within a padding bucket \((q/2,q]\),
\[
 f_D'(b)=-D\frac{a+cq}{b^2}<0.
\]
The continuous extension has one-sided local minima at every bucket endpoint \(8,16,32,64,128,256,512\), including the domain endpoints. Its upward right jump at any nonterminal endpoint is \(cD\). For **integer** batches, \(q\to q+1\) is an upward jump precisely when \(cq(q-1)>a\); this fails at 8 and holds at 16 and above. The integer local minima are therefore \(16,32,64,128,256,512\), not just 128 and 256. Among fully filled bucket endpoints, \(f_D(q)=D(\ell+c+a/q)\) decreases strictly, so the unrestricted sustained optimum is 512, not 128.

For a *hard pointwise feasible maximum* \(B\in[8,512]\), let \(q\) be the largest bucket endpoint no greater than \(B\). Only \(q\) and \(B\) need be compared. When \(q<B<2q\), choose \(B\) rather than \(q\) exactly when
\[
 B\ge B_{\rm cross}(q)
 =q\frac{a+2cq}{a+cq}.
\]
The crossovers for \(q=8,16,32,64,128,256\) are respectively 8.459364, 17.737679, 38.269786, 84.970397, **191.179906**, and 425.202467. Thus 129–191 loses to 128 as a sustained constant target; integer 192 first beats 128. In particular, a hard maximum of 134 selects 128.

For finite work, a constant *target* is not a strictly constant profile. Its exact one-residual cost is
\[
 F_D(b)=\ell D+\lfloor D/b\rfloor h(b)
        +\mathbf1_{D\bmod b>0}h(D\bmod b).
\]
It differs from \(f_D(b)\) by less than \(h(512)=33.0668\) ms in absolute value. This family can be evaluated at all 505 integer targets if exact finite-work selection is desired. For the target \(D=721595\), its integer local minima are exactly the same six endpoints above: inside each bucket the smooth decrease is at least 91.0218 ms per unit near the flattest endpoint, exceeding twice the finite-tail error bound; the boundary jump signs are unchanged. Direct exact-integer enumeration independently confirmed this.

| Target \(b\) | Sustained surrogate seconds \(f_D(b)/1000\) | Left/within-bucket derivative, s per additional request |
|---:|---:|---:|
| 8 | 856.641504 | -80.741971 |
| 16 | 552.218614 | -21.344555 |
| 32 | 400.007168 | -5.915670 |
| 64 | 323.901446 | -1.768683 |
| 128 | 285.848584 | **-0.587053** |
| 256 | 266.822154 | -0.219205 |
| 512 | 257.308938 | -0.091022 |

There is no two-sided derivative at 128. Exact finite targets 126, 127, 128, 129 give respectively 287.042068, 286.442254, 285.852480, 322.066750 surrogate seconds. Losing one filled lane below 128 costs 0.589774 s over this entire manifest; losing two costs 1.189588 s. Crossing upward to sustained 129 instead costs 36.214270 s in this finite-target family. The continuous-limit jump is 37.089983 s.

Do not apply those sustained-target penalties to a single transient crossing. For any two equal-work profiles the exact surrogate comparison is
\[
 \Delta M=a\Delta T+c\Delta S,\qquad S=\sum_tP(b_t).
\]
One 128-to-256 padding exposure costs only 6.5792 ms. A planner needs accurate *counts of crossings and steps*, not sub-request precision in an average batch estimate.

## 4. A proved KV step bound; why “1080 means maximum 134” is not proved

Assume: no KV sharing/prefix reuse; no context dropping beyond the stated full-context allocation geometry; at most one output per request per engine step; and a useful output at generated count \(g\) requires \(\lceil(p_i+g)/256\rceil\) live blocks. These assumptions are checkable against configuration and allocation/output traces. Recomputed input work can only consume additional resources and does not invalidate the following useful-work bound.

Define the **useful block-token work**
\[
 A_- =\sum_{i=1}^n\sum_{g=0}^{d_i-1}
          \left\lceil\frac{p_i+g}{256}\right\rceil.
\]
It is independent of order. At a step producing useful outputs, their required live blocks sum to at most 566. Summing over steps gives
\[
 T\ge L:=\max\left\{\max_i d_i,
       \left\lceil\frac D{\min(n,512)}\right\rceil,
       \left\lceil\frac{A_-}{566}\right\rceil\right\}.
\]
This is a physical step-count bound under the stated allocation assumptions, not a fitted-time claim. It remains valid with paused residents and preemptions: idle/resumed residents add occupancy not credited in \(A_-\). \(\blacksquare\)

OfflinePlanner extracted all \((p_i,d_i)\) from the target BK admissions:
\[
 n=1763,\quad D=721595,\quad\max_i d_i=4096,\quad
 A_-=2459546,\quad L=4346.
\]
The output-event-weighted allocated context is \(256A_-/D=872.572255\) token-equivalents, and the average-batch ceiling from this relaxation is \(566D/A_-=166.056162\), or \(D/4346=166.036585\) after integer steps. An allocation-before-service convention requiring one extra token gives \(A_+=2462156\), \(L=4351\); **use this stronger number only if the engine reserves that extra token before every useful output, including the final one**. The conservative \(A_-\) avoids silently assuming that convention.

Since \(P(b_t)\ge b_t\) and every padding size is a multiple of 8,
\[
 M_{\rm model}\ge \ell D+aL+c\,8\lceil D/8\rceil
 =\boxed{277.131480\ \text{surrogate seconds}}.
\]
Under the stronger verified-after-allocation convention the bound is 277.165230 surrogate seconds. These are **not** unconditional hardware makespan bounds. A pointwise lower-envelope cost model, or a separately justified hardware minimum per step and unit work, is the missing prerequisite for a bound in actual wall seconds. A high regression \(R^2\) cannot supply it.

The bound is tight for the aggregate relaxation, not necessarily for request/KV trajectories. Here \(S_0=8\lceil D/8\rceil=721600\), with \(m(S_0)=1411\). A dyadic capacity partition can have any number of steps between \(m(S_0)\) and \(S_0/8\): repeatedly split a non-8 bucket into two halves, raising the count by one without changing padding sum. Thus 4346 steps with padding sum 721600 exist as an abstract profile; distribute the five unused slots within one bucket. This proof does not schedule the actual requests or eliminate idle KV.

**What the approximate 1080 statement would imply under stronger hypotheses:**

- If *every step's active-set mean context* were at least 1080, then \(b_t\le\lfloor144896/1080\rfloor=134\) pointwise. Under this additional hard constraint, the exact surrogate minimum for this \(D\) is 285.852480 s: 5637 steps of 128 and one of 59. At most one step in 129–134 can occur in a cost optimum, because two such steps are dominated by two 128 steps plus a remainder of 2–12. Testing that one exceptional step adds 6.5792 ms for this particular remainder, so it does not help.
- If only the **useful-output-weighted** mean context were at least 1080, the conclusion would instead be \(T\ge\lceil1080D/144896\rceil=5379\). The corresponding tight aggregate surrogate bound is **284.104230 s**, not the constant-128 value. Mixing filled 128/256 buckets and a few boundary corrections can exploit the average constraint.
- An approximate request mean, an unweighted average of per-step means, or a mean observed under one policy proves neither hypothesis. For this manifest the exact useful block-token average is below 1080 even *after* rounding. Existing high-batch arms likewise rule out treating 134 as a universal pointwise ceiling.

This is why the discontinuous problem must not be replaced by “pick \(P(\bar b)\).” For example, half 128 steps and half 256 steps have \(\bar b=192\) and mean padding 192, whereas constant 192 pays padding 256. Under a mean-only constraint, variance in batch count can be beneficial. Under a hard pointwise cap it may be forbidden.

## 5. Drain: exact residual model, a tight excess bound, and a counterexample to mandatory collapse

At a time after the last admission, suppose \(m\le B\) residents remain and all make one useful decode token at every subsequent step until completion, with no stalls, recomputation, or further prefill. Their residuals must be taken from live generated counts: \(r_i=d_i-g_i\), not elapsed planner steps. This is the uninterrupted-service tail model, not a theorem about arbitrary vLLM service.

Order residuals \(r_{(1)}\ge\cdots\ge r_{(m)}>0\), put missing ranks equal to zero, and define \(R=r_{(1)}\), \(U=\sum_i r_i\). At tail step \(k\),
\[
 b_k=\#\{i:r_i\ge k\},\qquad 1\le k\le R.
\]
The exact padded exposure is the following seven-order-statistic formula:
\[
 S_{\rm tail}=8r_{(1)}+8r_{(9)}+16r_{(17)}+32r_{(33)}
              +64r_{(65)}+128r_{(129)}+256r_{(257)}.
\]
*Proof.* Expand the step function as
\(P(b)=8\mathbf1_{b\ge1}+8\mathbf1_{b\ge9}+16\mathbf1_{b\ge17}+\cdots+256\mathbf1_{b\ge257}\).
The number of tail steps with \(b_k\ge j\) is exactly \(r_{(j)}\). Sum this finite identity. \(\blacksquare\)

Therefore
\[
 M_{\rm tail}=\ell U+aR+cS_{\rm tail}.
\]
For a power-of-two target \(B\), let \(\lambda_B=h(B)/B\). The tail's excess over processing the same tokens at that filled target is
\[
 E_B=aR+cS_{\rm tail}-\lambda_BU
     =\sum_{k=1}^R\big[h(b_k)-\lambda_Bb_k\big]\ge0,
\]
because \(h(b)/b\ge h(B)/B\) for \(1\le b\le B\). Basic absolute overhead bounds are
\[
 aR+c\max(U,8R)\le aR+cS_{\rm tail}\le h(B)R.
\]
For \(B=128\), the piecewise-linear per-step excess is maximized at \(b=1\): evaluating the lower bucket endpoints \(1,9,17,33,65\) gives
\[
 0\le E_{128}\le7.057065625\,R\ {
m ms}.
\]
Since W-SMALL has \(r_i\le4096\),
\[
 E_{128}\le\boxed{28.9057408\ \text{surrogate seconds}}.
\]
This is tight for the residual model: one request with residual 4096 attains it. That tail's entire surrogate cost is 30.5283072 s. It is **not** an attainable improvement promise or a bound on all effects of changing a full admission schedule. Once only that request remains, there is no intra-request scheduling trick to produce more than one output per step; any improvement had to start it earlier or overlap it with other work.

A positive undersubscribed drain is **not mathematically unavoidable**. If \(B\) residents all have the same residual, they run at \(B\) until finishing simultaneously, so \(E_B=0\) and the profile drops directly from \(B\) to zero. Predicting and shaping residual order statistics is a distinct possible use of exact \(d_i\), beyond predicting KV releases.

Nor is uninterrupted service always padding-optimal even after the last admission. With 65 residents and residuals \((2,1,\ldots,1)\), uninterrupted service produces batches \((65,1)\). Delaying one short resident for one step gives \((64,2)\), preserving both \(D=66\) and \(T=2\) while reducing padding from 136 to 72, saving 3.2896 surrogate ms. The requests can be chosen with short prompts so KV is feasible in either schedule. This is a proof about a scheduler allowed to pause a resident without discarding its KV; implementing that requires a valid scheduler service hook, not filtering `SchedulerOutput`. It illustrates another consequence of padding nonconvexity, not a proposed large standalone hardware gain.

**Interpretation of the supplied small-bucket statistics.** If \(S\) is the number of steps with \(P(b)\le16\), the stated ranges imply 49.8–63.5% of those steps occur *outside* the final 10% of the run. Thus treating all small-bucket steps as the terminal drain is unsupported. Whether “10%” means wall time or step index must be recorded; if it means step index, every paired trace must satisfy \(S_{\rm final10}\le0.1T\). Do not cross-multiply unmatched extrema of the two reported ranges.

For each small-bucket step, the exact excess relative to a filled 128 target lies in \([5.90625,7.057065625]\) ms. If the stated 19.9–27.3% range applies to a particular trace, these steps contribute between **1.175344 and 1.926579 surrogate seconds per 1000 total steps** of that trace. This is an exposure accounting bound, not a hardware saving. The percentages alone do not determine residuals, the true last-admission time, or total drain cost. Log that time, queue-empty status, and the seven residual ranks. Optimizing a severe tail is potentially worthwhile; optimizing only the final 10% while ignoring early starvation is not justified by these data.

## 6. Implementable use, complexity, and falsifiable hardware arms

This work supplies an exact scoring/bounding library, not an admission-feasibility substitute. A feasible planner must additionally enforce request dependencies and, at every actual allocation, \(\sum_{i\in R_t}\lceil(p_i+g_i(t))/256\rceil\le566\), including retained paused residents and the engine's allocation lookahead. Exact lengths alone do not guarantee predicted release steps when mixed prefill skips decode service.

```text
choose_constant_target(B):               # 8 <= B <= 512, certified hard cap only
    q = greatest member of BUCKETS <= B
    return argmin over {q, B} of h(x)/x   # exact integer cross-products; ties prefer q

score_realized_profile(D, T, padding_sum):
    return 1460*D + 33750*T + 257*padding_sum   # units = 0.0002 ms

relaxed_profile(D):
    k, r = divmod(D, 512)
    if r == 0:    return counts(k x 512)
    if r <= 256:  return counts(k x 512, r)
    if r <= 320:  return counts(k x 512, 256, r-256)
    return counts(k x 512, r)
    # Last branch's actual r is 321..511, charged to padding 512.

F(m):                                    # m >= 0
    u, r = divmod(m, 256)
    return 128*u*(u+1) + (u+1)*r          # sum_{x=1}^m ceil(x/256)

manifest_bound(manifest):                 # auditing/Track C only; p_i >= 1
    if n == 0: return 0, 0
    D = sum(d_i)
    A = sum(F(p_i+d_i-1) - F(p_i-1))
    L = max(max(d_i), ceil(D/min(n,512)), ceil(A/566))
    return L, score_realized_profile(D, L, 8*ceil(D/8))

score_uninterrupted_tail(live_generated, exact_lengths, B):  # Track C only
    residuals = sort_descending(d_i - live_generated_i)
    R = max(residuals, default=0); U = sum(residuals)
    S = 8*r[1]+8*r[9]+16*r[17]+32*r[33]+64*r[65]+128*r[129]+256*r[257]
    return a*R+c*S, a*R+c*S-h(B)*U/B       # missing ranks are zero
```

Complexity: target selection and the closed-form profile are \(O(\log512)\) time, \(O(1)\) memory in compressed counts; expanding a profile costs \(O(T)\). Manifest bounds cost \(O(n)\) time and \(O(1)\) extra memory. Trace scoring costs \(O(T)\) time, \(O(1)\) memory. Sorting tail residuals costs \(O(b\log b)\) time and \(O(b)\) workspace; updating it on every step would cost \(O(Tb\log b)\), so use it at plan/endgame evaluation boundaries, not as needless per-step work. This library performs no combinatorial search and does not run the \(O(D\cdot512)\) verification DP in production.

**Hard planning budget:** 1.0 s per decoded in-memory manifest for this analytical preprocessing, with a monotonic deadline checked during the \(O(n)\) scan. On expiry, return the already computed constant-target recommendation and, if the complete \(D\) is available from authoritative manifest metadata or a finished scan, the generic non-KV bound; otherwise mark all numeric bounds unavailable. Mark the strengthened KV bound unavailable, never a partially accumulated bound labeled exact. Report parsing/startup time separately as well as total planning time. Per-decision target selection is constant work. Use integer arithmetic for the specified decimal coefficients; 64-bit checked arithmetic or arbitrary-precision integers avoid rounding-induced bucket ties. No fitted decimal is claimed physically exact.

Track O policy objects receive neither the manifest-bound inputs nor true residuals and assert exact lengths absent at construction. Their target selection uses only a genuinely enforced cap; any tail estimate must use live generated counts and an estimator trained on completed requests. True residual diagnostics may be produced by a separate evaluator, not passed back into Track O.

**Falsifiable predictions / arms for the orchestrator; none run here:**

1. **Model-invariance probe, before trusting a hardware lower bound.** Same model/TP16, prefix caching disabled, no preemption, 256 requests: 128 prompts of length 64 and 128 of length 512, all with 65 total outputs. Use an identical prefill phase in both arms and pause each request immediately after its first output. Begin decode capture only when all 256 live generated counts equal 1, leaving 64 outputs each. Enforce exactly 128 decode requests per step for 128 steps. `BP_context_cluster` serves the short half to completion then the long half; `BP_context_mix` serves a fixed 64+64 half-cohort to completion then the remaining 64+64. Both have identical manifest (16640 total outputs), identical captured decode work (16384 outputs), identical decode batch/padding profile, and peak KV at most \(128\cdot1+128\cdot3=512\) blocks. The count-only model predicts equal decode time. A reproducible timing difference rejects fixed-profile hardware order-invariance. Record whole-run wall time and instrumented decode-only intervals directly, not by subtracting a separate prefill run. A second probe at decode batch 64, with 64 prompts of length 64 and 64 of length 1024 and the same 65-total/64-remaining convention, uses at most 384 blocks and crosses the sliding-window boundary. Any custom pause mechanism must preserve KV and scheduler bookkeeping.
2. **Padding knee in the real workload.** Matched `BP_target128`, `BP_target134`, and `BP_target192` on W-SMALL, same admission/trajectory-safety mechanism, same request ordering, no changes except the target. Prediction conditional on actually sustained, safe targets and transferable step costs: 134 is worse than 128; 192 narrowly beats it. Failure to sustain a target is a feasibility result, not evidence against the arithmetic. Record exact \(T,S\), mixed-step work, allocation slack, and preemptions. Repeat on W-LARGE without importing W-SMALL's residency assumption.
3. **Endgame value of length information.** Same full-service trajectory gate and hard cap 128, same planning wall budget and evaluation count. Compare a Track C planner scored through completion (or using the exact tail formula beyond a truncated horizon) against the same planner whose truncated score omits the tail. Prediction: manifests with large residual ranks \(r_{(1)},r_{(9)},\ldots\) are the ones with meaningful reducible surrogate tail exposure; hardware benefit must exceed paired-arm noise and cannot be inferred from mean batch alone. Then run information-only twins with the same objective/mechanism, replacing true residuals by completed-request-only estimates. This tests a use of \(d_i\) distinct from a watermark or sort-key change. Full-horizon planners already account for the tail: do not add it twice.

## 7. Verification and exact scope

Executed pure mathematical checks, not a scheduler simulation or TPU benchmark:

- An independent integer-partition dynamic program over every \(D=0,\ldots,4096\), allowing all batches 1–512, agreed with the closed-form dyadic solution in all **4097 cases**, with **zero exact-integer discrepancies**. Costs used 0.0002 ms units; this is evidence corroborating the proof, not a replacement for it.
- Exact enumeration of all finite-target costs for \(D=721595\) gave local minima **16, 32, 64, 128, 256, 512** and the numerical sensitivity values above. The unconstrained relaxed optimum is 257.3167696 surrogate seconds; it is not feasible evidence for this manifest, whose longest request alone needs 4096 output steps.
- Direct layer-by-layer tail sums matched the seven-rank formula exactly for five cases: a 65-resident boundary crossing; one residual of 4096; 128 equal residuals; residuals 1–134 crossing multiple buckets; and a tied six-resident example. Their \((m,R,U,S)\) values were respectively \((65,2,66,136)\), \((1,4096,4096,32768)\), \((128,100,12800,12800)\), \((134,134,9045,12480)\), and \((6,7,22,56)\).
- Edge checks: \(D=0\) uses no steps; filled 512 profiles attain the asymptotic rate bound; constant equal-residual tails have zero excess; a serial residual attains the stated upper excess bound; the 257-token split disproves universal constant profiles. All count/capacity quantities are integers, costs are nonnegative, and comparisons keep milliseconds separate from seconds.

A shared-kernel name collision invalidated one intermediate arithmetic printout; all reported constants were recomputed and rechecked inside an isolated local scope. No hardware jobs, benchmark runs, harness edits, permanent tests, or executable scheduler changes were made. The deliverable's actual-wall conclusion is deliberately limited: the supplied measurements do not establish a numerical hardware lower bound or a globally optimal physical batch target. They do establish which exact surrogate comparisons to test, and the artifact-derived block-work invariant supplies a stronger, correctly weighted feasibility bound than the approximate 1080-context argument.
