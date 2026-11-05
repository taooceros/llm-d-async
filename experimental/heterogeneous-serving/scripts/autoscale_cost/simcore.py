"""Discrete-event simulator of a fleet of TP8 vLLM instances behind a least-outstanding router (numba).

Model (mirrors the real AutoscaleBurst harness, scripts/asb_loadgen.py in ~/hetero-wt/AutoscaleBurst):
  router    one FIFO admission queue; dispatch to the active instance with the fewest outstanding requests (tie:
            oldest last dispatch) while it has < cap outstanding; a request still queued after adm_to seconds is
            rejected (429); instances join when booted (state ACTIVE) and leave after draining.
  engine    vLLM continuous batching with chunked prefill at step granularity: every step, each decoding request
            emits one token and the remaining token budget (max_batched - n_decode) goes to prefilling requests in
            FCFS order; a request is admitted to the engine when its prompt fits in the free KV (kv_cap tokens).
            Step time = p0 + max(P, D) + p6*min(P, D), P = p2*bucket(n_dec + prefill_tokens)/1e3
                        + p4*sum(chunk*(done+chunk/2))/1e6, D = p7 + p1*n_dec + p3*max(ctx of decoding)/1e5
                        + p8*sum(ctx of decoding)/1e5,
                        bucket = next power of two >= 16 (vLLM-TPU token padding)   (fit in calibrate.py)
            The simulator jumps over runs of identical steps (no admission/completion inside) so it costs a few
            events per request. TTFT = arrival -> end of the step that finishes the prompt (+ p5 constant
            overhead); TPOT = (finish - first token)/(out - 1).
  control   ticks every 1 s. mode 0: the campaign's queue trigger (queue > thr in every sample for sustain s since
            the previous trigger, +1 instance, <= max_trig, boot time boot_list[k]); mode 1: reactive headroom
            target = max(min_inst, ceil((rate + h)/C)), scale up at once, scale down to the max target of the last
            T s (keep-alive); rate = arrivals in the last W s / W (or the next W s: clairvoyant oracle);
            mode 2: HPA timing (scrape every scrape_s, sync every sync_s, desired = ceil(rate/(u*C)) with
            tolerance, scale-up/-down stabilization windows Su/Sd, default k8s rate limits); mode 3: static.
  billing   an instance is billed from acquisition (boot start) to release (booting instances are cancelled
            first on scale-down; an active instance drains its outstanding requests before release).
"""
from __future__ import annotations

import numpy as np
from numba import njit

FREE, BOOT, ACTIVE, DRAIN = 0, 1, 2, 3
EMPTY, WAIT, PREF, DEC = 0, 1, 2, 3
INF = 1e300

# policy vector indices
P_MODE, P_H, P_T, P_W, P_C, P_MIN, P_MAX, P_U, P_SU, P_SD, P_SCRAPE, P_SYNC, P_TOL, P_CLAIR, P_THR, P_SUST, \
    P_MAXTRIG, P_N0, P_BLO, P_BHI, P_SEED, P_UPLIM = range(22)
NPOL = 22
# engine vector: p0..p8, cap, max_batched, kv_cap, adm_to
NPAR = 9
E_CAP, E_MAXB, E_KV, E_ADM, E_PB = 9, 10, 11, 12, 13
NENG = 14


@njit(cache=True)
def _sift_up(heap, pos, key, j):
    x = heap[j]
    kx = key[x]
    while j > 0:
        p = (j - 1) >> 1
        y = heap[p]
        if key[y] <= kx:
            break
        heap[j] = y
        pos[y] = j
        j = p
    heap[j] = x
    pos[x] = j


@njit(cache=True)
def _sift_down(heap, pos, key, j, n):
    x = heap[j]
    kx = key[x]
    while True:
        c = 2 * j + 1
        if c >= n:
            break
        if c + 1 < n and key[heap[c + 1]] < key[heap[c]]:
            c += 1
        y = heap[c]
        if key[y] >= kx:
            break
        heap[j] = y
        pos[y] = j
        j = c
    heap[j] = x
    pos[x] = j


@njit(cache=True)
def _set_key(heap, pos, key, i, v, n):
    old = key[i]
    key[i] = v
    if v < old:
        _sift_up(heap, pos, key, pos[i])
    elif v > old:
        _sift_down(heap, pos, key, pos[i], n)


@njit(cache=True)
def _step_time(eng, n_dec, ptok, ctxsum, attn, ctxmax):
    # vLLM-TPU pads the step's token count to a power-of-two bucket (>= 16): compute term on the bucket,
    # memory term (weights p7 + per decoding request + attention: p3 on the longest context, latency-bound
    # kernel parallel over sequences, p8 on the total context, bandwidth); the two overlap by (1 - p6)
    n = n_dec + ptok
    b = 16
    while b < n:
        b *= 2
    pc = eng[2] * b * 1e-3 + eng[4] * attn * 1e-6
    dm = eng[7] + eng[1] * n_dec + eng[3] * ctxmax * 1e-5 + eng[8] * ctxsum * 1e-5
    if pc >= dm:
        return eng[0] + pc + eng[6] * dm
    return eng[0] + dm + eng[6] * pc


@njit(cache=True)
def simulate(arr, inp, outp, eng, pol, boot_list, cum, n_ticks, t_end_arrivals):
    """Run one configuration. Returns per-request (ttft, tpot, status, inst) and per-tick logs.

    arr/inp/outp: requests sorted by arrival (s, tokens). cum[s] = #arrivals with arr < s (len n_ticks+W+2).
    status: 0 never finished, 1 ok, 2 rejected (admission timeout).
    """
    N = arr.size
    MAXI = int(pol[P_MAX]) + 8
    if MAXI < 8:
        MAXI = 8
    MAXS = int(eng[E_CAP])
    cap = MAXS
    maxb = int(eng[E_MAXB])
    kvcap = eng[E_KV]
    adm_to = eng[E_ADM]
    mode = int(pol[P_MODE])
    np.random.seed(int(pol[P_SEED]))

    ttft = np.full(N, np.nan, np.float32)
    tpot = np.full(N, np.nan, np.float32)
    status = np.zeros(N, np.int8)
    rinst = np.full(N, -1, np.int16)
    tq = np.zeros(n_ticks, np.int32)        # router queue at each tick
    tbill = np.zeros(n_ticks, np.int16)     # billed instances (boot + active + drain)
    tboot = np.zeros(n_ticks, np.int16)     # booting instances
    tact = np.zeros(n_ticks, np.int16)      # active (serving, not draining)
    ttgt = np.zeros(n_ticks, np.int16)      # policy target incl. headroom (before keep-alive), mode 1/2
    ttgt0 = np.zeros(n_ticks, np.int16)     # ceil(rate / C): load-following part without headroom

    # instance state
    st = np.zeros(MAXI, np.int8)
    t_acq = np.zeros(MAXI)
    t_ready = np.zeros(MAXI)
    outst = np.zeros(MAXI, np.int32)
    lastd = np.zeros(MAXI)
    seg_t0 = np.zeros(MAXI)
    seg_tau = np.zeros(MAXI)
    seg_k = np.zeros(MAXI, np.int64)
    seg_type = np.zeros(MAXI, np.int8)      # 0 idle, 1 uniform, 2 mixed single step
    seg_head = np.full(MAXI, -1, np.int32)  # slot of the head prefill in a uniform segment
    seg_ptok = np.zeros(MAXI, np.int64)
    # slots
    s_ph = np.zeros((MAXI, MAXS), np.int8)
    s_rid = np.zeros((MAXI, MAXS), np.int64)
    s_rem = np.zeros((MAXI, MAXS), np.int64)
    s_ctx = np.zeros((MAXI, MAXS), np.int64)
    s_seq = np.zeros((MAXI, MAXS), np.int64)
    s_tf = np.zeros((MAXI, MAXS))
    alloc = np.zeros(MAXS, np.int64)
    seqc = 0
    # heap over instances (key = next event time: boot completion or segment end)
    key = np.full(MAXI, INF)
    heap = np.arange(MAXI).astype(np.int64)
    pos = np.arange(MAXI).astype(np.int64)
    act = np.zeros(MAXI, np.int64)
    n_act = 0
    # logs
    n_acq = 0
    n_cancel = 0
    trig_t = np.full(64, np.nan)
    join_t = np.full(64, np.nan)
    n_trig = 0
    inst_trig = np.full(MAXI, -1, np.int64)

    # controller state
    high_since = -1.0
    ctl_since = -1e18
    W = int(pol[P_W])
    T = int(pol[P_T])
    last_ge = np.full(MAXI + 2, -10 ** 12, np.int64)  # keep-alive: last tick where target >= v
    hpa_metric = 0.0
    hpa_hist_t = np.zeros(4096)
    hpa_hist_v = np.zeros(4096, np.int64)
    hpa_n = 0
    hpa_last_sync = -1e18
    cur_tgt = 0
    cur_tgt0 = 0

    nxt = 0      # next arrival
    qh = 0       # queue head (requests nxt.. are not yet arrived; [qh, nxt) arrived, some dispatched?)
    # queue = requests [qh, nxt) that are not dispatched: FIFO dispatch keeps it contiguous
    tick = 1

    # initial instances
    n0 = int(pol[P_N0])
    for i in range(n0):
        st[i] = ACTIVE
        t_acq[i] = 0.0
        t_ready[i] = 0.0
        n_acq += 1
    n_act = 0
    for i in range(MAXI):
        if st[i] == ACTIVE:
            act[n_act] = i
            n_act += 1

    t = 0.0
    while True:
        ta = arr[nxt] if nxt < N else INF
        ti = key[heap[0]]
        tt = float(tick) if tick < n_ticks else INF
        if ta == INF and tt == INF:
            # drain remaining instance events only if requests outstanding
            if ti == INF:
                break
        if ta <= ti and ta <= tt:
            t = ta
            nxt += 1
            ev = 0
        elif ti <= tt:
            t = ti
            ev = 1
        else:
            t = tt
            ev = 2
        if ev == 2 and nxt >= N and qh >= N:
            # nothing left to arrive or queue: stop ticking once every request is done
            busy = False
            for j in range(n_act):
                if outst[act[j]] > 0:
                    busy = True
                    break
            if not busy:
                anyd = False
                for i in range(MAXI):
                    if st[i] == DRAIN:
                        anyd = True
                        break
                if not anyd:
                    break

        changed = False   # instance set changed -> rebuild act
        if ev == 1:
            i = heap[0]
            if st[i] == BOOT:
                st[i] = ACTIVE
                t_ready[i] = t
                _set_key(heap, pos, key, i, INF, MAXI)
                if inst_trig[i] >= 0 and inst_trig[i] < 64:
                    join_t[inst_trig[i]] = t
                changed = True
            else:
                # segment end: advance seg_k steps
                k = seg_k[i]
                if seg_type[i] == 1:
                    for s in range(MAXS):
                        if s_ph[i, s] == DEC:
                            s_rem[i, s] -= k
                            s_ctx[i, s] += k
                    h = seg_head[i]
                    if h >= 0:
                        s_rem[i, h] -= k * seg_ptok[i]
                        s_ctx[i, h] += k * seg_ptok[i]
                elif seg_type[i] == 2:
                    for s in range(MAXS):
                        if s_ph[i, s] == DEC:
                            s_rem[i, s] -= 1
                            s_ctx[i, s] += 1
                    # FCFS allocation identical to the plan
                    nd = 0
                    for s in range(MAXS):
                        if s_ph[i, s] == DEC:
                            nd += 1
                    left = maxb - nd
                    if nd > 0 and maxb > eng[E_PB]:
                        left = int(eng[E_PB]) - nd
                    while left > 0:
                        best = -1
                        for s in range(MAXS):
                            if s_ph[i, s] == PREF and s_rem[i, s] > 0 and alloc[s] == 0:
                                if best < 0 or s_seq[i, s] < s_seq[i, best]:
                                    best = s
                        if best < 0:
                            break
                        c = min(left, s_rem[i, best])
                        alloc[best] = c
                        left -= c
                    for s in range(MAXS):
                        if alloc[s] > 0:
                            s_rem[i, s] -= alloc[s]
                            s_ctx[i, s] += alloc[s]
                            alloc[s] = 0
                # completions
                for s in range(MAXS):
                    ph = s_ph[i, s]
                    if ph == PREF and s_rem[i, s] == 0:
                        r = s_rid[i, s]
                        ttft[r] = t - arr[r] + eng[5]
                        if outp[r] <= 1:
                            status[r] = 1
                            s_ph[i, s] = EMPTY
                            outst[i] -= 1
                        else:
                            s_ph[i, s] = DEC
                            s_rem[i, s] = outp[r] - 1
                            s_tf[i, s] = t
                    elif ph == DEC and s_rem[i, s] <= 0:
                        r = s_rid[i, s]
                        tpot[r] = (t - s_tf[i, s]) / (outp[r] - 1)
                        status[r] = 1
                        s_ph[i, s] = EMPTY
                        outst[i] -= 1
                seg_type[i] = 0
                # replan below
                if st[i] == DRAIN and outst[i] == 0:
                    st[i] = FREE
                    _set_key(heap, pos, key, i, INF, MAXI)
                    changed = True
                else:
                    _replan(i, t, eng, maxb, kvcap, MAXS, s_ph, s_rem, s_ctx, s_seq, inp, s_rid, seg_t0, seg_tau,
                            seg_k, seg_type, seg_head, seg_ptok, heap, pos, key, MAXI)

        if ev == 2:
            # controller tick
            # purge admission timeouts from the queue head
            while qh < nxt and t - arr[qh] > adm_to:
                status[qh] = 2
                qh += 1
            qd = nxt - qh
            nb = 0
            nbt = 0
            na = 0
            for i in range(MAXI):
                if st[i] != FREE:
                    nb += 1
                    if st[i] == BOOT:
                        nbt += 1
                    elif st[i] == ACTIVE:
                        na += 1
            want = -1   # desired (boot + active) count, -1 = no action
            if mode == 0:
                if qd > pol[P_THR]:
                    if high_since < 0:
                        high_since = t
                else:
                    high_since = -1.0
                if n_trig < int(pol[P_MAXTRIG]):
                    # every sample in (ctl_since, t] > thr, and high for >= sustain
                    # the real controller compares wall-clock sample times: with 1 s samples it needs
                    # sustain + 1 samples (observed: triggers 11 s apart), hence the strict + 0.5
                    if high_since >= 0 and t - high_since > pol[P_SUST] + 0.5:
                        # acquire one instance with boot_list[n_trig]
                        i = _free_slot(st, MAXI)
                        if i >= 0:
                            st[i] = BOOT
                            t_acq[i] = t
                            n_acq += 1
                            inst_trig[i] = n_trig
                            trig_t[n_trig] = t
                            _set_key(heap, pos, key, i, t + boot_list[n_trig], MAXI)
                            n_trig += 1
                            ctl_since = t
                            high_since = t
            elif mode == 1:
                if pol[P_CLAIR] > 0:
                    lo = tick
                    hi = min(tick + W, cum.size - 1)
                else:
                    lo = max(tick - W, 0)
                    hi = min(tick, cum.size - 1)
                rate = (cum[hi] - cum[lo]) / W
                tgt = int(np.ceil((rate + pol[P_H]) / pol[P_C] - 1e-9))
                tgt = max(tgt, int(pol[P_MIN]))
                tgt = min(tgt, int(pol[P_MAX]))
                cur_tgt = tgt
                cur_tgt0 = min(max(int(np.ceil(rate / pol[P_C] - 1e-9)), int(pol[P_MIN])), int(pol[P_MAX]))
                for v in range(1, tgt + 1):
                    last_ge[v] = tick
                eff = tgt
                v = min(nb + 1, MAXI + 1)
                while v > tgt:
                    if last_ge[v] > tick - T:
                        eff = v
                        break
                    v -= 1
                want = eff
            elif mode == 2:
                if tick % int(pol[P_SCRAPE]) == 0:
                    lo = max(tick - W, 0)
                    hi = min(tick, cum.size - 1)
                    hpa_metric = (cum[hi] - cum[lo]) / W
                lo = max(tick - W, 0)
                hi = min(tick, cum.size - 1)
                cur_tgt0 = min(max(int(np.ceil((cum[hi] - cum[lo]) / W / pol[P_C] - 1e-9)), int(pol[P_MIN])),
                               int(pol[P_MAX]))
                if tick % int(pol[P_SYNC]) == 0:
                    cur = nbt + na
                    raw = hpa_metric / (pol[P_U] * pol[P_C])
                    if cur > 0 and abs(raw / cur - 1.0) <= pol[P_TOL]:
                        rec = cur
                    else:
                        rec = int(np.ceil(raw - 1e-9))
                    rec = max(rec, int(pol[P_MIN]))
                    rec = min(rec, int(pol[P_MAX]))
                    hpa_hist_t[hpa_n % 4096] = t
                    hpa_hist_v[hpa_n % 4096] = rec
                    hpa_n += 1
                    # stabilization: scale-up uses the min recommendation of the last Su s, scale-down the max of Sd
                    upv = rec
                    dnv = rec
                    wmax = max(pol[P_SU], pol[P_SD])
                    j = hpa_n - 1
                    while j >= max(0, hpa_n - 4096):
                        tj = hpa_hist_t[j % 4096]
                        if t - tj > wmax + 1e-9:
                            break
                        vj = hpa_hist_v[j % 4096]
                        if t - tj <= pol[P_SU] + 1e-9 and vj < upv:
                            upv = vj
                        if t - tj <= pol[P_SD] + 1e-9 and vj > dnv:
                            dnv = vj
                        j -= 1
                    if upv > cur:
                        lim = max(2 * cur, cur + int(pol[P_UPLIM]))
                        want = min(upv, lim)
                    elif dnv < cur:
                        want = dnv
                    else:
                        want = cur
                    hpa_last_sync = t
                    cur_tgt = rec
            if want >= 0:
                cur = nbt + na
                if want > cur:
                    need = want - cur
                    # reclaim draining instances first (most outstanding first)
                    while need > 0:
                        best = -1
                        for i in range(MAXI):
                            if st[i] == DRAIN and (best < 0 or outst[i] > outst[best]):
                                best = i
                        if best < 0:
                            break
                        st[best] = ACTIVE
                        need -= 1
                        changed = True
                    while need > 0:
                        i = _free_slot(st, MAXI)
                        if i < 0:
                            break
                        b = pol[P_BLO] + (pol[P_BHI] - pol[P_BLO]) * np.random.random()
                        st[i] = BOOT
                        t_acq[i] = t
                        n_acq += 1
                        need -= 1
                        if b <= 0:
                            st[i] = ACTIVE
                            t_ready[i] = t
                            changed = True
                        else:
                            _set_key(heap, pos, key, i, t + b, MAXI)
                elif want < cur:
                    extra = cur - want
                    # cancel booting instances first (latest acquisition first)
                    while extra > 0:
                        best = -1
                        for i in range(MAXI):
                            if st[i] == BOOT and (best < 0 or t_acq[i] > t_acq[best]):
                                best = i
                        if best < 0:
                            break
                        st[best] = FREE
                        _set_key(heap, pos, key, best, INF, MAXI)
                        n_cancel += 1
                        extra -= 1
                    while extra > 0:
                        best = -1
                        for i in range(MAXI):
                            if st[i] == ACTIVE and (best < 0 or outst[i] < outst[best]):
                                best = i
                        if best < 0:
                            break
                        if outst[best] == 0:
                            st[best] = FREE
                            _set_key(heap, pos, key, best, INF, MAXI)
                        else:
                            st[best] = DRAIN
                        extra -= 1
                        changed = True
            # log (after actions)
            if tick < n_ticks:
                nb = 0
                nbt = 0
                na = 0
                for i in range(MAXI):
                    if st[i] != FREE:
                        nb += 1
                        if st[i] == BOOT:
                            nbt += 1
                        elif st[i] == ACTIVE:
                            na += 1
                tq[tick] = qd
                tbill[tick] = nb
                tboot[tick] = nbt
                tact[tick] = na
                if mode == 1 or mode == 2:
                    ttgt[tick] = cur_tgt
                    ttgt0[tick] = cur_tgt0
                else:
                    ttgt[tick] = na
                    ttgt0[tick] = na
            tick += 1

        if changed:
            n_act = 0
            for i in range(MAXI):
                if st[i] == ACTIVE:
                    act[n_act] = i
                    n_act += 1

        # dispatch
        while qh < nxt:
            if t - arr[qh] > adm_to:
                status[qh] = 2
                qh += 1
                continue
            best = -1
            for j in range(n_act):
                i = act[j]
                if outst[i] >= cap:
                    continue
                if best < 0 or outst[i] < outst[best] or (outst[i] == outst[best] and lastd[i] < lastd[best]):
                    best = i
            if best < 0:
                break
            i = best
            r = qh
            qh += 1
            outst[i] += 1
            lastd[i] = t
            rinst[r] = i
            # put into a free slot as WAIT
            for s in range(MAXS):
                if s_ph[i, s] == EMPTY:
                    s_ph[i, s] = WAIT
                    s_rid[i, s] = r
                    s_rem[i, s] = inp[r]
                    s_ctx[i, s] = 0
                    s_seq[i, s] = seqc
                    seqc += 1
                    break
            if seg_type[i] == 0:
                _replan(i, t, eng, maxb, kvcap, MAXS, s_ph, s_rem, s_ctx, s_seq, inp, s_rid, seg_t0, seg_tau,
                        seg_k, seg_type, seg_head, seg_ptok, heap, pos, key, MAXI)
            elif seg_type[i] == 1:
                j = int(np.ceil((t - seg_t0[i]) / seg_tau[i] - 1e-9))
                if j < 0:
                    j = 0
                if j < seg_k[i]:
                    # partial advance by j steps (no completion can happen inside a uniform segment)
                    for s in range(MAXS):
                        if s_ph[i, s] == DEC:
                            s_rem[i, s] -= j
                            s_ctx[i, s] += j
                    h = seg_head[i]
                    if h >= 0:
                        s_rem[i, h] -= j * seg_ptok[i]
                        s_ctx[i, h] += j * seg_ptok[i]
                    tc = seg_t0[i] + j * seg_tau[i]
                    seg_type[i] = 0
                    _replan(i, tc, eng, maxb, kvcap, MAXS, s_ph, s_rem, s_ctx, s_seq, inp, s_rid, seg_t0, seg_tau,
                            seg_k, seg_type, seg_head, seg_ptok, heap, pos, key, MAXI)
            # mixed single step: the new request joins at the segment end

    return (ttft, tpot, status, rinst, tq, tbill, tboot, tact, ttgt, ttgt0, trig_t[:n_trig], join_t[:n_trig], n_acq,
            n_cancel, t)


@njit(cache=True)
def _free_slot(st, MAXI):
    for i in range(MAXI):
        if st[i] == FREE:
            return i
    return -1


@njit(cache=True)
def _replan(i, t, eng, maxb, kvcap, MAXS, s_ph, s_rem, s_ctx, s_seq, inp, s_rid, seg_t0, seg_tau, seg_k, seg_type,
            seg_head, seg_ptok, heap, pos, key, MAXI):
    # admit WAIT slots (FCFS) while the prompt fits in the free KV
    used = 0
    for s in range(MAXS):
        ph = s_ph[i, s]
        if ph == PREF:
            used += inp[s_rid[i, s]]
        elif ph == DEC:
            used += s_ctx[i, s]
    while True:
        best = -1
        for s in range(MAXS):
            if s_ph[i, s] == WAIT and (best < 0 or s_seq[i, s] < s_seq[i, best]):
                best = s
        if best < 0:
            break
        need = inp[s_rid[i, best]]
        if used + need > kvcap and used > 0:
            break
        s_ph[i, best] = PREF
        s_ctx[i, best] = 0
        used += need
    n_dec = 0
    dmin = 1 << 60
    ctxsum = 0
    ctxmax = 0
    head = -1
    for s in range(MAXS):
        ph = s_ph[i, s]
        if ph == DEC:
            n_dec += 1
            ctxsum += s_ctx[i, s]
            if s_ctx[i, s] > ctxmax:
                ctxmax = s_ctx[i, s]
            if s_rem[i, s] < dmin:
                dmin = s_rem[i, s]
        elif ph == PREF:
            if head < 0 or s_seq[i, s] < s_seq[i, head]:
                head = s
    if n_dec == 0 and head < 0:
        seg_type[i] = 0
        _set_key(heap, pos, key, i, INF, MAXI)
        return
    budget = maxb - n_dec
    if n_dec > 0 and maxb > eng[E_PB]:
        budget = int(eng[E_PB]) - n_dec
    if head < 0:
        k = dmin
        tau = _step_time(eng, n_dec, 0, ctxsum + n_dec * (k - 1) * 0.5, 0.0, ctxmax + (k - 1) * 0.5)
        seg_type[i] = 1
        seg_head[i] = -1
        seg_ptok[i] = 0
    else:
        remh = s_rem[i, head]
        if remh > budget:
            q = (remh - 1) // budget
            k = q if n_dec == 0 else min(q, dmin)
            done_mid = s_ctx[i, head] + budget * (k - 1) * 0.5
            attn = budget * (done_mid + budget * 0.5)
            tau = _step_time(eng, n_dec, budget, ctxsum + n_dec * (k - 1) * 0.5, attn, ctxmax + (k - 1) * 0.5)
            seg_type[i] = 1
            seg_head[i] = head
            seg_ptok[i] = budget
        else:
            # one mixed step: FCFS allocation over the prefilling requests
            k = 1
            left = budget
            ptok = 0
            attn = 0.0
            taken = np.zeros(MAXS, np.bool_)
            while left > 0:
                best = -1
                for s in range(MAXS):
                    if s_ph[i, s] == PREF and not taken[s]:
                        if best < 0 or s_seq[i, s] < s_seq[i, best]:
                            best = s
                if best < 0:
                    break
                taken[best] = True
                c = min(left, s_rem[i, best])
                ptok += c
                attn += c * (s_ctx[i, best] + c * 0.5)
                left -= c
            tau = _step_time(eng, n_dec, ptok, ctxsum, attn, ctxmax)
            seg_type[i] = 2
            seg_head[i] = -1
            seg_ptok[i] = ptok
    if tau < 1e-4:
        tau = 1e-4
    seg_t0[i] = t
    seg_tau[i] = tau
    seg_k[i] = k
    _set_key(heap, pos, key, i, t + k * tau, MAXI)


def make_pol(**kw) -> np.ndarray:
    p = np.zeros(NPOL)
    d = dict(mode=3, h=0.0, T=0, W=60, C=0.7, min_inst=1, max_inst=64, u=0.7, Su=0, Sd=0, scrape=10, sync=15,
             tol=0.1, clair=0, thr=2, sustain=10, max_trig=3, n0=1, blo=0.0, bhi=0.0, seed=1, uplim=4)
    d.update(kw)
    names = ["mode", "h", "T", "W", "C", "min_inst", "max_inst", "u", "Su", "Sd", "scrape", "sync", "tol", "clair",
             "thr", "sustain", "max_trig", "n0", "blo", "bhi", "seed", "uplim"]
    for j, n in enumerate(names):
        p[j] = d[n]
    return p


def make_eng(p, cap=8, max_batched=2048, kv_cap=205 * 256, adm_to=600.0, pb=None) -> np.ndarray:
    """pb: prefill-token budget per step while requests are decoding (None = max_batched)."""
    e = np.zeros(NENG)
    e[:NPAR] = p
    e[E_CAP], e[E_MAXB], e[E_KV], e[E_ADM] = cap, max_batched, kv_cap, adm_to
    e[E_PB] = max_batched if pb is None else pb
    return e


def cum_counts(arr: np.ndarray, n_ticks: int, W: int) -> np.ndarray:
    """cum[s] = number of arrivals with arr < s, s = 0 .. n_ticks + W + 1."""
    edges = np.arange(n_ticks + W + 2, dtype=np.float64)
    return np.searchsorted(arr, edges, side="left").astype(np.int64)


def run(arr, inp, outp, eng, pol, boot_list=None, n_ticks=None):
    arr = np.ascontiguousarray(arr, np.float64)
    inp = np.ascontiguousarray(inp, np.int64)
    outp = np.ascontiguousarray(outp, np.int64)
    if n_ticks is None:
        n_ticks = int(np.ceil(arr[-1])) + 2
    W = int(pol[P_W])
    cum = cum_counts(arr, n_ticks, W)
    bl = np.zeros(64) if boot_list is None else np.asarray(list(boot_list) + [0.0] * (64 - len(boot_list)), float)
    res = simulate(arr, inp, outp, eng, pol, bl, cum, n_ticks, float(arr[-1]))
    keys = ["ttft", "tpot", "status", "inst", "q", "bill", "boot", "act", "tgt", "tgt0", "trig_t", "join_t", "n_acq",
            "n_cancel", "t_end"]
    return dict(zip(keys, res))
