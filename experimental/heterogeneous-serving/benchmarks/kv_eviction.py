"""F3's CPU policy, native selector seam, and real-request drain guard.

Integration (at a quiescent scheduler boundary, before installing trace wrappers):

* Construct F3Scorer with the C/O FutureModel, charging model construction via
  ``startup_elapsed_s``. Only the scorer sees this model; it never sees requests,
  the scheduler, the engine, a manifest, or a target-derived O snapshot.
* Construct GuardAdapter with public endpoint block reservations, including the
  allocator's group geometry/lookahead overhead. ``other_blocks`` accounts for
  outstanding allocations outside these requests. Individual endpoints must fit.
* Install EvictionController. ``snapshot(request, rank, protected)`` MUST return
  exactly {index,p,g,blocks,rank,protected}; g is emitted output length, not
  num_computed_tokens minus prompt. blocks is freshly releasable allocator state.
  ``block_deficit(scheduler, request, num_new_tokens)`` is evaluated AFTER the
  native failed allocation, never from an earlier step. It must use the actual
  allocator convention and include in-flight/group overhead.
* Gate external dispatch on ``guard.allow_arrivals``. Keep driving/flushing the
  engine while external work, native work, OR ``guard.has_pending()`` remains.
  Guard-held requests remain in scheduler.requests: the guard owns eligibility,
  not request lifetime. Observe real completion/status updates before each call.
  Never cancel/recreate a held request. Call close() in finally to return every
  held object and restore the exact original instance methods/queue objects.

No SchedulerOutput is edited. F3 is a local proxy and greedy repeated-victim
heuristic, not an optimal eviction-set solver. Reorder arms are qualified ONLY by
an explicit homogeneous audit; this module never asserts traversal isolation.
The installed_scheduler_sign_fixture is a CPU scheduler fixture, NOT throughput.
"""

import ast
import copy
import hashlib
import inspect
import math
import textwrap
import time
import types
from collections import deque
import numpy as np


_SNAPSHOT_FIELDS = frozenset(('index', 'p', 'g', 'blocks', 'rank', 'protected'))
_MISSING = object()
_HORIZON = 128
_BLOCK_SIZE = 256


class GuardInvariantError(RuntimeError):
    """The experimental arm must stop: its ownership/feasibility proof failed."""


class SelectorShapeError(RuntimeError):
    """Installed scheduler control flow is not the audited native shape."""


class _Deadline(Exception):
    pass


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _snapshots(rows, prompts=None):
    assert type(rows) in (tuple, list), 'Snapshots must be plain public containers'
    indices, ranks = set(), set()
    for row in rows:
        assert type(row) is dict and row.keys() == _SNAPSHOT_FIELDS, (
            'Snapshot whitelist violation (including d, engine, or max_tokens)')
        index, p, g, blocks, rank = row['index'], row['p'], row['g'], row['blocks'], row['rank']
        if not (type(index) is int and type(p) is int and type(g) is int
                and type(blocks) is int and type(rank) is int
                and min(index, p, g, blocks, rank) >= 0):
            raise ValueError('Snapshot indices, prompts, observations, blocks and ranks must be nonnegative integers')
        assert type(row['protected']) is bool, 'protected must be public bool'
        if prompts is not None and prompts.get(index) != p:
            raise ValueError('Snapshot prompt differs from the public catalogue')
        if index in indices or rank in ranks:
            raise ValueError('Live indices and canonical ranks must be unique')
        indices.add(index)
        ranks.add(rank)
    return rows


def _tail(rows):
    candidates = (row for row in rows if not row['protected'])
    result = max(candidates, key=lambda row: row['rank'], default=None)
    if result is None:
        raise GuardInvariantError('No unprotected native victim')
    return result


def f3_terms(profile, context, background_horizon, terminal_weight):
    """Equation (12), O(1) survival sums; no engine or target inspection.

    B uses the declared ceil((p+g+s)/256) proxy; freshly allocated blocks are
    separately used for actual singleton feasibility, not substituted for B.
    """
    if profile.block_size != _BLOCK_SIZE:
        raise ValueError('F3 normalization requires 256-token logical blocks')
    k = -(-context // _BLOCK_SIZE)
    growth_at = _BLOCK_SIZE * k - context + 1
    duration = profile.survival_sum(1, _HORIZON + 1)
    relief = k * duration
    if growth_at == 1:
        # The conditional resident is certainly alive for its first token.
        relief += duration - 1.0
    elif growth_at < _HORIZON:
        relief += profile.survival_sum(growth_at + 1, _HORIZON + 1)
    delta = (profile.survival_sum(background_horizon - _HORIZON + 1,
                                  background_horizon + 1)
             if terminal_weight else 0.0)
    upper = -(-(context + _HORIZON - 1) // _BLOCK_SIZE) * _HORIZON
    if not (math.isfinite(relief) and -1e-12 <= relief <= upper + 1e-9
            and math.isfinite(delta) and -1e-12 <= delta <= _HORIZON + 1e-12):
        raise ValueError('Invalid future relief or terminal expectation')
    return {'score': context + terminal_weight * delta - 2.0 * relief,
            'context': context, 'expected_B128': relief, 'expected_delta': delta,
            'efficiency': context / relief if relief > 0 else math.inf}


class F3Scorer:
    """Shared C/O policy. Input rows are validated at every information boundary."""

    def __init__(self, future_model, *, startup_elapsed_s=0.0):
        started = time.monotonic()
        if not math.isfinite(startup_elapsed_s) or startup_elapsed_s < 0:
            raise ValueError('startup_elapsed_s must include nonnegative model setup time')
        if future_model.track not in ('C', 'O') or future_model.block_size != _BLOCK_SIZE:
            raise ValueError('F3 needs a C/O FutureModel with block_size=256')
        if future_model.track == 'O':
            assert all(set(row) == {'index', 'p', 'arrival_index'}
                       for row in future_model.requests), 'O model contains forbidden fields'
        self.model = future_model
        self.online_s = 0.0
        self.decisions = 0
        self.fallbacks = 0
        self.budget_violations = 0
        self.last_decision = None
        self._prompts = {row['index']: row['p'] for row in future_model.requests}
        self._positions = {index: position for position, index in enumerate(future_model.indices)}
        future_model.prepare_survival_batches(deadline=started + 2.0 - startup_elapsed_s)
        for n, index in enumerate(future_model.indices):
            # Warm the provider's shared lazy tables INSIDE the startup charge.
            profile = future_model.future_profile(index)
            f3_terms(profile, profile.p, 0, 0)
            if n % 32 == 31 and time.monotonic() - started + startup_elapsed_s > 2.0:
                raise TimeoutError('F3 startup exceeds 2 seconds; reject this arm')
        self.startup_s = time.monotonic() - started + startup_elapsed_s
        if self.startup_s > 2.0:
            raise TimeoutError('F3 startup exceeds 2 seconds; reject this arm')

    def choose(self, snapshots, *, fresh_backlog, deficit, started_at=None):
        """Return an index plus audit data; only complete decisions may be committed.

        fresh_backlog means undispatched OR nonquarantined waiting work, including
        the guard-held queues. An empty quarantine alone is not fresh backlog.
        started_at includes harness snapshot/deficit work in the 1 ms deadline.
        """
        start = time.monotonic() if started_at is None else started_at
        if start > time.monotonic() or not math.isfinite(start):
            raise ValueError('started_at must be a monotonic timestamp in the past')
        rows = _snapshots(snapshots, self._prompts)
        if type(fresh_backlog) is not bool:
            raise ValueError('fresh_backlog must be a public boolean')
        deficit = _integer(deficit, 'deficit', 1)
        tail = _tail(rows)
        deadline = min(start + 0.001, start + max(0.0, 5.0 - self.online_s))
        result = None
        reason = None
        nu = 0 if fresh_backlog else 256
        horizon = 0
        try:
            if time.monotonic() >= deadline:
                raise _Deadline
            size = len(rows)
            positions = np.fromiter((self._positions[row['index']] for row in rows), np.int64, size)
            generated = np.fromiter((row['g'] for row in rows), np.int64, size)
            contexts = np.fromiter((row['p'] + row['g'] for row in rows), np.int64, size)
            sums = self.model.survival_sums
            if nu:
                means = sums(positions, generated, 1, self.model.max_model_len + 1)
                if np.any(~np.isfinite(means)) or np.any(means < 1 - 1e-12):
                    raise ValueError('Invalid conditional residual mean')
                horizon = math.ceil(float(means.max()))
            k = (contexts + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            growth_at = _BLOCK_SIZE * k - contexts + 1
            # Two windows share validation and law grouping; values remain the same queries.
            windows = sums(np.concatenate((positions, positions)),
                           np.concatenate((generated, generated)),
                           np.concatenate((np.ones(size, dtype=np.int64),
                                           np.minimum(growth_at + 1, _HORIZON + 1))),
                           _HORIZON + 1)
            duration, growth = windows[:size], windows[size:]
            relief = k * duration + np.where(growth_at == 1, duration - 1.0, growth)
            delta = (sums(positions, generated, horizon - _HORIZON + 1, horizon + 1)
                     if nu else np.zeros(size))
            upper = ((contexts + _HORIZON - 1 + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _HORIZON
            if (np.any(~np.isfinite(relief)) or np.any(relief < -1e-12)
                    or np.any(relief > upper + 1e-9) or np.any(~np.isfinite(delta))
                    or np.any(delta < -1e-12) or np.any(delta > _HORIZON + 1e-12)):
                raise ValueError('Invalid future relief or terminal expectation')
            scores = contexts + nu * delta - 2.0 * relief
            best = None
            best_score = best_efficiency = math.inf
            best_rank = 0
            eligible = np.fromiter((not row['protected'] and row['blocks'] >= deficit
                                    for row in rows), bool, size)
            if np.any(eligible):
                minimum = float(np.min(scores, where=eligible, initial=np.inf))
                # Match math.isclose, not numpy's asymmetric additive tolerance.
                close = np.abs(scores - minimum) <= np.maximum(
                    1e-12, 1e-12 * np.maximum(np.abs(scores), abs(minimum)))
                near = np.flatnonzero(eligible & close)
                if np.all(scores[near] == minimum):
                    # A unique minimum or exact ties cannot acquire an order-dependent
                    # approximate-tie chain. Preserve efficiency then canonical-rank ties.
                    efficiencies = np.full(len(near), np.inf)
                    np.divide(contexts[near], relief[near], out=efficiencies, where=relief[near] > 0)
                    ranks = np.fromiter((rows[n]['rank'] for n in near), np.int64, len(near))
                    selected = int(np.lexsort((-ranks, efficiencies))[0])
                    best = int(near[selected])
                    best_score, best_efficiency = minimum, float(efficiencies[selected])
                    best_rank = rows[best]['rank']
                else:
                    # Approximate ties are nontransitive: retain the original ordered fold.
                    for n, row in enumerate(rows):
                        if eligible[n]:
                            score = float(scores[n])
                            efficiency = float(contexts[n] / relief[n]) if relief[n] > 0 else math.inf
                            tied = math.isclose(score, best_score, rel_tol=1e-12, abs_tol=1e-12)
                            if (best is None or tied and (efficiency, -row['rank']) < (best_efficiency, -best_rank)
                                    or not tied and score < best_score):
                                best, best_score, best_efficiency, best_rank = n, score, efficiency, row['rank']
                        if n % 32 == 31 and time.monotonic() >= deadline:
                            raise _Deadline
            if best is not None:
                result = {'index': rows[best]['index'], 'rank': best_rank, 'score': best_score,
                          'context': int(contexts[best]), 'expected_B128': float(relief[best]),
                          'expected_delta': float(delta[best]), 'efficiency': best_efficiency}
            if time.monotonic() >= deadline:
                raise _Deadline
            if result is None:
                reason = 'no_feasible_singleton_native_multivictim_fallback'
        except _Deadline:
            reason = 'online_budget_exhausted_canonical_tail'
        elapsed = time.monotonic() - start
        self.online_s += elapsed
        self.decisions += 1
        violation = elapsed > 0.001 or self.online_s > 5.0
        self.budget_violations += int(violation)
        if violation:
            reason = 'online_budget_exhausted_canonical_tail'
        if reason:
            result = {'index': tail['index'], 'rank': tail['rank']}
            self.fallbacks += 1
        result.update(fallback=reason is not None, reason=reason or 'f3', nu=nu,
                      L0=horizon, elapsed_s=elapsed, budget_violation=violation,
                      background_maximum='ceil_max_mean_not_integrated' if nu else None)
        self.last_decision = result
        return result

    def summary(self):
        return {'startup_s': self.startup_s, 'online_s': self.online_s,
                'decisions': self.decisions, 'fallbacks': self.fallbacks,
                'budget_violations': self.budget_violations,
                'startup_limit_s': 2.0, 'decision_limit_s': 0.001, 'online_limit_s': 5.0}


# Fail closed on ANY control-flow change, not just the presence of .pop/max.
_NATIVE_BRANCH = '''
if self.policy == SchedulingPolicy.PRIORITY:
    preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
    self.running.remove(preempted_req)
    if preempted_req in scheduled_running_reqs:
        preempted_req_id = preempted_req.request_id
        scheduled_running_reqs.remove(preempted_req)
        token_budget += num_scheduled_tokens.pop(preempted_req_id)
        req_to_new_blocks.pop(preempted_req_id)
        scheduled_spec_decode_tokens.pop(preempted_req_id, None)
        preempted_encoder_inputs = scheduled_encoder_inputs.pop(preempted_req_id, None)
        if preempted_encoder_inputs:
            num_embeds_to_restore = sum(preempted_req.get_num_encoder_embeds(i)
                                       for i in preempted_encoder_inputs)
            encoder_compute_budget += num_embeds_to_restore
        req_index -= 1
else:
    preempted_req = self.running.pop()
'''
_NATIVE_AFTER = '''
self._preempt_request(preempted_req, scheduled_timestamp)
preempted_reqs.append(preempted_req)
if preempted_req == request:
    break
'''


def _dump(node):
    return ast.dump(node, include_attributes=False)


def transform_native_schedule(source):
    """Return a compiled-ready AST and audit hashes; unknown native shapes fail."""
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise SelectorShapeError('Expected precisely one native schedule function')
    function = tree.body[0]
    if function.name != 'schedule' or function.decorator_list:
        raise SelectorShapeError('Expected undecorated native schedule, not a wrapper')
    expected = ast.parse(_NATIVE_BRANCH).body[0]
    tests = [node for node in ast.walk(function)
             if isinstance(node, ast.If) and _dump(node.test) == _dump(expected.test)]
    if len(tests) != 1 or _dump(tests[0]) != _dump(expected):
        raise SelectorShapeError('Priority/FCFS selector/refund AST is not the unique audited shape')
    branch = tests[0]
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    parent = parents[branch]
    if not (isinstance(parent, ast.While) and isinstance(parent.test, ast.Constant)
            and parent.test.value is True):
        raise SelectorShapeError('Selector is not directly inside native allocation retry loop')
    position = parent.body.index(branch)
    after = ast.parse(_NATIVE_AFTER).body
    if [_dump(n) for n in parent.body[position + 1:]] != [_dump(n) for n in after]:
        raise SelectorShapeError('Native preempt/append/current-victim branch changed')
    allocation = ast.parse('''new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)''').body[0]
    success = ast.parse('if new_blocks is not None:\n    break').body[0]
    if position != 2 or [_dump(n) for n in parent.body[:position]] != [
            _dump(allocation), _dump(success)]:
        raise SelectorShapeError('Native allocation failure entry changed')
    replacement = copy.deepcopy(branch.body)
    replacement[0].value = ast.parse(
        '_kv_f3_select_victim(self, request, num_new_tokens)', mode='eval').body
    parent.body[position:position + 1] = replacement
    ast.fix_missing_locations(tree)
    # All refunds and all code outside the selection branch are copied verbatim.
    return tree, {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'transformed_ast_sha256': hashlib.sha256(_dump(tree).encode()).hexdigest(),
                  'native_refunds_retained': True, 'unique_match_count': 1}


def _restore_attribute(obj, name, original):
    if original is _MISSING:
        delattr(obj, name)
    else:
        setattr(obj, name, original)


class NativeSelectorSeam:
    """Instance-local seam: native priority bookkeeping, independent callback victim."""

    def __init__(self, scheduler, select_victim):
        self.scheduler = scheduler
        self.original = scheduler.schedule
        self._attribute = vars(scheduler).get('schedule', _MISSING)
        self._installed = False
        function = getattr(self.original, '__func__', None)
        if function is None or function.__closure__:
            raise SelectorShapeError('A native, closure-free bound schedule method is required')
        tree, self.audit = transform_native_schedule(inspect.getsource(function))
        namespace = dict(function.__globals__)
        if '_kv_f3_select_victim' in namespace:
            raise SelectorShapeError('Selector namespace collision')

        def checked_select(owner, current, num_new_tokens):
            if owner is not scheduler:
                raise GuardInvariantError('Selector attached to a different scheduler')
            victim = select_victim(owner, current, num_new_tokens)
            if not any(victim is request for request in owner.running):
                raise GuardInvariantError('Selector returned a nonresident object')
            return victim

        namespace['_kv_f3_select_victim'] = checked_select
        code = compile(tree, inspect.getsourcefile(function) or '<native-selector>', 'exec')
        exec(code, namespace)
        self.patched = types.MethodType(namespace['schedule'], scheduler)
        scheduler.schedule = self.patched
        self._installed = True

    def restore(self):
        if self._installed:
            _restore_attribute(self.scheduler, 'schedule', self._attribute)
            self._installed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


def _queue_add(queue, request):
    if hasattr(queue, 'add_request'):
        queue.add_request(request)
    else:
        queue.append(request)


def _queue_pop(queue):
    return queue.pop_request() if hasattr(queue, 'pop_request') else queue.popleft()


def _queue_prepend(queue, request):
    if hasattr(queue, 'prepend_request'):
        queue.prepend_request(request)
    else:
        queue.appendleft(request)


def _queue_remove(queue, request):
    if not any(item is request for item in queue):
        raise GuardInvariantError('Native preemption did not enqueue its actual victim')
    if hasattr(queue, 'remove_request'):
        queue.remove_request(request)
    else:
        queue.remove(request)


class GuardAdapter:
    """Own real queue eligibility, never request lifetime or engine output.

    Queue objects and request statuses are retained unchanged. Native _preempt_request
    runs once and then its actual victim is transferred into FIFO quarantine. On a
    predicted pressure call, pass pressure_expected=True to before_schedule to hold
    queues BEFORE calling native schedule. An unpredicted first failure closes them
    inside _preempt_request; the audited native no-preemption waiting gate guarantees
    that neither old waiting queue can admit on that call. All subsequent calls are
    held before schedule. This distinction is recorded, never hidden as a prediction.
    """

    def __init__(self, scheduler, *, index_of, endpoint_blocks, capacity,
                 other_blocks, is_finished=None, event_sink=None):
        self.scheduler = scheduler
        self.index_of = index_of
        self.endpoint_blocks = dict(endpoint_blocks)
        self.capacity = _integer(capacity, 'capacity', 1)
        for index, blocks in self.endpoint_blocks.items():
            _integer(index, 'endpoint index')
            if _integer(blocks, 'public endpoint blocks', 1) > self.capacity:
                raise GuardInvariantError('A public singleton endpoint exceeds capacity')
        self.other_blocks = other_blocks
        self.is_finished = is_finished or (lambda request: request.is_finished())
        self.event_sink = event_sink
        self.events = []
        self.state = 'open'
        self._held = None
        self._quarantine = deque()
        self._cohort = {}
        self._pack = {}
        self._seen_victims = set()
        self._before_running = ()
        self._installed = False
        self._preempt = scheduler._preempt_request
        self._preempt_attribute = vars(scheduler).get('_preempt_request', _MISSING)
        self._add = getattr(scheduler, 'add_request', None)
        self._add_attribute = vars(scheduler).get('add_request', _MISSING)

    def _event(self, event, **fields):
        record = {'event': event, 'at_monotonic_s': time.monotonic(), **fields}
        self.events.append(record)
        if self.event_sink is not None:
            self.event_sink(record)

    @property
    def allow_arrivals(self):
        return self.state == 'open'

    def has_pending(self):
        return bool(self._quarantine or self._pack or self._cohort
                    or self._held and any(self._held))

    def has_fresh_waiting(self):
        queues = self._held if self._held is not None else (
            self.scheduler.waiting, self.scheduler.skipped_waiting)
        protected = self._pack
        return any(not self.is_finished(request) and id(request) not in protected
                   for queue in queues for request in queue)

    def is_protected(self, request):
        return id(request) in self._pack

    def _hold(self):
        if self._held is not None:
            return
        waiting, skipped = self.scheduler.waiting, self.scheduler.skipped_waiting
        if waiting is skipped:
            raise GuardInvariantError('Waiting and skipped queues alias')
        # Construct replacements before transferring ownership, so failure is atomic.
        empty_waiting, empty_skipped = type(waiting)(), type(skipped)()
        self._held = (waiting, skipped)
        self.scheduler.waiting, self.scheduler.skipped_waiting = empty_waiting, empty_skipped

    def _release(self):
        if self._held is None:
            return
        if self.scheduler.waiting or self.scheduler.skipped_waiting:
            raise GuardInvariantError('Cannot reopen with unreconciled active waiting requests')
        self.scheduler.waiting, self.scheduler.skipped_waiting = self._held
        self._held = None

    def _prune(self):
        for members in (self._cohort, self._pack):
            for identity, request in tuple(members.items()):
                if self.is_finished(request):
                    del members[identity]

    def _advance(self):
        self._prune()
        if self.state == 'drain' and self._cohort:
            return
        if self.state == 'protected' and self._pack:
            return
        if self.state not in ('drain', 'protected'):
            return
        if self.scheduler.running or self.scheduler.waiting or self.scheduler.skipped_waiting:
            raise GuardInvariantError('Drain ended while native runnable ownership remains')
        while self._quarantine and self.is_finished(self._quarantine[0]):
            self._quarantine.popleft()
        if not self._quarantine:
            self._release()
            self.state = 'open'
            self._event('guard_reopen')
            return
        liability = _integer(self.other_blocks(self.scheduler), 'other allocated blocks')
        reserved = liability
        pack = []
        while self._quarantine:
            request = self._quarantine[0]
            if self.is_finished(request):
                self._quarantine.popleft()
                continue
            endpoint = self.endpoint_blocks[self.index_of(request)]
            if reserved + endpoint > self.capacity:
                break
            reserved += endpoint
            pack.append(self._quarantine.popleft())
        if not pack:
            if liability:
                self._event('guard_wait_allocator_liability', other_blocks=liability)
                return
            raise GuardInvariantError('FIFO protected singleton cannot fit')
        self.state = 'protected'
        self._pack = {id(request): request for request in pack}
        for request in pack:
            _queue_add(self.scheduler.waiting, request)
        self._event('guard_protected_pack', indices=[self.index_of(r) for r in pack],
                    reserved_blocks=reserved, other_blocks=liability, capacity=self.capacity)

    def before_schedule(self, *, pressure_expected=False):
        if type(pressure_expected) is not bool:
            raise ValueError('pressure_expected must be a public boolean')
        self._advance()
        self._before_running = tuple(self.scheduler.running)
        if self.state == 'open' and pressure_expected:
            self._hold()
            self.state = 'armed'
            self._event('guard_hold_before_pressure')
        elif self.state in ('drain', 'protected'):
            self._hold()
        if self.state == 'drain':
            if self.scheduler.waiting or self.scheduler.skipped_waiting:
                raise GuardInvariantError('Admissions appeared during resident drain')
            if any(id(r) not in self._cohort for r in self.scheduler.running):
                raise GuardInvariantError('A fresh request entered the resident drain')
        if self.state == 'protected':
            active = (*self.scheduler.running, *self.scheduler.waiting,
                      *self.scheduler.skipped_waiting)
            if any(id(r) not in self._pack for r in active):
                raise GuardInvariantError('An arrival entered a protected pack')
            reservation = sum(self.endpoint_blocks[self.index_of(r)] for r in self._pack.values())
            if reservation + self.other_blocks(self.scheduler) > self.capacity:
                raise GuardInvariantError('Protected endpoint/in-flight reservation was violated')

    def after_schedule(self):
        if self.state == 'armed':
            self._release()
            self.state = 'open'
            self._event('guard_pressure_prediction_no_victim')
        elif self.state == 'drain':
            if self.scheduler.waiting or self.scheduler.skipped_waiting:
                raise GuardInvariantError('A native victim escaped quarantine')
            if any(id(r) not in self._cohort for r in self.scheduler.running):
                raise GuardInvariantError('A native arrival escaped the drain gate')
        self._prune()

    def _on_preempt(self, request, *args, **kwargs):
        if self.is_protected(request) or self.state == 'protected':
            self._event('guard_protected_preemption_failure', index=self.index_of(request))
            raise GuardInvariantError('Native allocator tried to evict a protected pack member')
        if self.index_of(request) in self._seen_victims:
            raise GuardInvariantError('A request was evicted twice under the drain guard')
        if self.state in ('open', 'armed'):
            held_before = self._held is not None
            self._hold()
            self._cohort = {id(r): r for r in self._before_running if not self.is_finished(r)}
            self.state = 'drain'
            self._event('guard_begin_drain', held_before_schedule=held_before,
                        cohort_size=len(self._cohort))
        if id(request) not in self._cohort:
            raise GuardInvariantError('Preempted request was not in the resident cohort')
        self._preempt(request, *args, **kwargs)
        _queue_remove(self.scheduler.waiting, request)
        self._cohort.pop(id(request))
        self._quarantine.append(request)
        self._seen_victims.add(self.index_of(request))
        self._event('guard_quarantine', index=self.index_of(request),
                    quarantine_size=len(self._quarantine))

    def install(self):
        if self._installed:
            raise GuardInvariantError('Guard already installed')
        self.scheduler._preempt_request = self._on_preempt
        if self._add is not None:
            def guarded_add(*args, **kwargs):
                if not self.allow_arrivals:
                    raise GuardInvariantError('External dispatch attempted during closed admissions')
                return self._add(*args, **kwargs)
            self.scheduler.add_request = guarded_add
        self._installed = True
        return self

    def restore(self):
        """Return held/quarantined objects once, including an interrupted arm."""
        # Native removal precedes the guard hook. If a protected-victim invariant
        # aborts that call, return its still-RUNNING object to native ownership.
        owned = {id(r) for r in (*self.scheduler.running, *self.scheduler.waiting,
                                  *self.scheduler.skipped_waiting, *self._quarantine)}
        for request in (*self._cohort.values(), *self._pack.values()):
            if id(request) not in owned and not self.is_finished(request):
                self.scheduler.running.append(request)
                owned.add(id(request))
        if self._held is not None:
            waiting, skipped = self._held
            active_waiting = []
            while self.scheduler.waiting:
                active_waiting.append(_queue_pop(self.scheduler.waiting))
            while self.scheduler.skipped_waiting:
                _queue_add(skipped, _queue_pop(self.scheduler.skipped_waiting))
            pending = active_waiting + list(self._quarantine)
            if len({id(r) for r in pending}) != len(pending):
                raise GuardInvariantError('Duplicate queue ownership on guard restoration')
            for request in reversed(pending):
                if not self.is_finished(request):
                    _queue_prepend(waiting, request)
            self._quarantine.clear()
            self._release()
        self._cohort.clear()
        self._pack.clear()
        self.state = 'open'
        if self._installed:
            _restore_attribute(self.scheduler, '_preempt_request', self._preempt_attribute)
            if self._add is not None:
                _restore_attribute(self.scheduler, 'add_request', self._add_attribute)
            self._installed = False


_HOMOGENEOUS_FLAGS = frozenset((
    'safe_boundary', 'no_request_blocking', 'no_deferred_frees', 'no_kv_sharing',
    'no_waiting_admission', 'at_most_one_victim'))


class EvictionController:
    """Native fcfs / selector_sham / f3 / reorder_f3 / reorder_sham audit arms.

    reorder_sham changes the same F3 traversal as reorder_f3 but forces the
    canonical (pre-permutation) tail independently. ``reorder_audit(scheduler)``
    returns {deficit:int, <all _HOMOGENEOUS_FLAGS>:bool}; unknown/failed checks
    leave traversal canonical. Per-request one-token demand, no speculation or
    encoder input, and budget slack are checked here as well. A failed/missing
    audit is logged and cannot become a traversal-isolation claim.
    """

    MODES = frozenset(('fcfs', 'selector_sham', 'f3', 'reorder_f3', 'reorder_sham'))

    def __init__(self, scheduler, *, mode, guard, scorer, snapshot, block_deficit,
                 fresh_backlog, reorder_audit=None, pressure_expected=None, event_sink=None):
        if mode not in self.MODES or guard.scheduler is not scheduler:
            raise ValueError('Unknown arm or mismatched guard scheduler')
        if mode in ('f3', 'reorder_f3', 'reorder_sham') and scorer is None:
            raise ValueError('An F3 scorer is required for selection or F3 traversal')
        self.scheduler, self.mode, self.guard, self.scorer = scheduler, mode, guard, scorer
        self.snapshot, self.block_deficit = snapshot, block_deficit
        self.fresh_backlog, self.reorder_audit = fresh_backlog, reorder_audit
        self.pressure_expected, self.event_sink = pressure_expected, event_sink
        self.events = []
        self.seam = None
        self._canonical = ()
        self._permuted = False
        self._ranks = {}
        self._attribute = vars(scheduler).get('schedule', _MISSING)
        self._native = scheduler.schedule
        self._installed = False
        self.schedule_calls = self.eligible_steps = self.eligible_preemptions = 0
        self.preemptions = 0
        self._eligible = False

    def _event(self, event, **fields):
        record = {'event': event, 'mode': self.mode, 'at_monotonic_s': time.monotonic(), **fields}
        self.events.append(record)
        if self.event_sink is not None:
            self.event_sink(record)

    def _rows(self):
        return [self.snapshot(request, self._ranks[id(request)], self.guard.is_protected(request))
                for request in self.scheduler.running]

    def _fresh(self):
        value = self.fresh_backlog()
        if type(value) is not bool:
            raise ValueError('fresh_backlog() must return bool, not an engine object')
        return value or self.guard.has_fresh_waiting()

    def _select(self, scheduler, current, num_new_tokens):
        start = time.monotonic()
        rows = self._rows()
        deficit = _integer(self.block_deficit(scheduler, current, num_new_tokens), 'actual deficit', 1)
        snapshot_s = time.monotonic() - start
        if self.mode in ('selector_sham', 'reorder_sham'):
            _snapshots(rows)
            tail = _tail(rows)
            decision = {'index': tail['index'], 'reason': 'canonical_tail_sham', 'fallback': False}
        else:
            decision = self.scorer.choose(rows, fresh_backlog=self._fresh(),
                                          deficit=deficit, started_at=start)
        victim = next(request for request, row in zip(scheduler.running, rows)
                      if row['index'] == decision['index'])
        if self.guard.is_protected(victim):
            raise GuardInvariantError('Policy selected protected request')
        self.preemptions += 1
        self.eligible_preemptions += int(self._eligible)
        self._event('victim_decision', **decision, deficit=deficit, snapshot_s=snapshot_s,
                    current_index=next(row['index'] for request, row in zip(scheduler.running, rows)
                                       if request is current),
                    native_tail_index=rows[-1]['index'], eligible_reorder=self._eligible)
        return victim

    def _restore_order(self, *, require_safe=False):
        if not self._permuted:
            return
        running = self.scheduler.running
        identities = {id(request) for request in running}
        previous = {id(request) for request in self._canonical}
        restored = ([request for request in self._canonical if id(request) in identities]
                    + [request for request in running if id(request) not in previous])
        if any(left is not right for left, right in zip(running, restored)):
            if require_safe:
                audit = self.reorder_audit(self.scheduler) if self.reorder_audit else None
                if type(audit) is not dict or audit.get('safe_boundary') is not True:
                    self._event('reorder_restore_unsafe', traversal_isolation_claim=False)
                    raise GuardInvariantError('Cannot restore traversal outside an audited safe boundary')
            running[:] = restored
        self._permuted = False

    def _maybe_reorder(self):
        self._eligible = False
        if self.mode not in ('reorder_f3', 'reorder_sham'):
            return
        start = time.monotonic()
        audit = self.reorder_audit(self.scheduler) if self.reorder_audit else None
        running = self.scheduler.running
        valid = type(audit) is dict and all(audit.get(flag) is True for flag in _HOMOGENEOUS_FLAGS)
        valid = (valid and len(running) > 1
                 and self.scheduler.max_num_scheduled_tokens >= len(running)
                 and not self.scheduler.num_lookahead_tokens
                 and not getattr(self.scheduler, 'connector', None)
                 and not getattr(self.scheduler, 'ec_connector', None)
                 and all(not r.spec_token_ids and not r.has_encoder_inputs
                         and not r.num_output_placeholders and not r.is_prefill_chunk
                         and r.num_tokens_with_spec - r.num_computed_tokens == 1 for r in running))
        if not valid:
            self._event('reorder_ineligible', audit=audit, traversal_isolation_claim=False)
            return
        self._eligible = True
        self.eligible_steps += 1
        deficit = _integer(audit.get('deficit'), 'prospective deficit')
        if not deficit:
            self._event('reorder_no_pressure', traversal_isolation_claim=False)
            return
        rows = self._rows()
        decision = self.scorer.choose(rows, fresh_backlog=self._fresh(), deficit=deficit, started_at=start)
        if not decision['fallback']:
            victim = next(r for r, row in zip(running, rows) if row['index'] == decision['index'])
            if victim is not running[-1]:
                running.remove(victim)
                running.append(victim)
                self._permuted = True
        self._event('reorder_decision', **decision, reordered=self._permuted,
                    traversal_isolation_claim=False)

    def _schedule(self, *args, **kwargs):
        self._restore_order(require_safe=True)
        pressure = self.pressure_expected(self.scheduler) if self.pressure_expected else False
        self.guard.before_schedule(pressure_expected=pressure)
        self._canonical = tuple(self.scheduler.running)
        self._ranks = {id(request): rank for rank, request in enumerate(self._canonical)}
        self._maybe_reorder()
        self.schedule_calls += 1
        output = self._native(*args, **kwargs)  # Exactly once, no postfilter/refund reimplementation.
        if self._eligible and output.preempted_req_ids:
            victims = set(output.preempted_req_ids)
            services = {self.guard.index_of(request): output.num_scheduled_tokens.get(
                request.request_id, 0) for request in self._canonical
                if request.request_id not in victims}
            self._event('restricted_survivor_service_audit', services=services,
                        holds=all(tokens == 1 for tokens in services.values()),
                        victim_indices=[self.guard.index_of(request) for request in self._canonical
                                        if request.request_id in victims],
                        traversal_isolation_claim=False)
        self.guard.after_schedule()
        return output

    def install(self):
        if self._installed:
            raise GuardInvariantError('Eviction controller already installed')
        try:
            if self.mode != 'fcfs':
                self.seam = NativeSelectorSeam(self.scheduler, self._select)
                self._native = self.scheduler.schedule
            self.guard.install()
            self.scheduler.schedule = self._schedule
            self._installed = True
        except BaseException:
            if self.seam:
                self.seam.restore()
            self.guard.restore()
            raise
        return self

    def summary(self):
        return {'mode': self.mode, 'schedule_calls': self.schedule_calls,
                'eligible_steps': self.eligible_steps, 'selector_preemptions': self.preemptions,
                'eligible_preemptions': self.eligible_preemptions,
                'traversal_isolation_claim': False,
                'seam': self.seam.audit if self.seam else None,
                'scorer': self.scorer.summary() if self.scorer else None,
                'events': self.events, 'guard_events': self.guard.events,
                'unique_evicted_requests': len(self.guard._seen_victims)}

    def close(self):
        self._restore_order()
        try:
            self.guard.restore()
        finally:
            if self.seam:
                self.seam.restore()
            if self._installed and self.seam is None:
                _restore_attribute(self.scheduler, 'schedule', self._attribute)
            self._installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()


class FixtureDependencyError(RuntimeError):
    """An installed constructor/interface needs an explicit actor-side factory."""


def _construct_known(constructor, supplied):
    signature = inspect.signature(constructor)
    kwargs = {name: supplied[name] for name, parameter in signature.parameters.items()
              if name in supplied and parameter.kind != inspect.Parameter.POSITIONAL_ONLY}
    missing = [name for name, parameter in signature.parameters.items()
               if parameter.default is inspect.Parameter.empty
               and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                           inspect.Parameter.VAR_KEYWORD)
               and name not in kwargs]
    if missing:
        raise FixtureDependencyError(
            f'{constructor.__module__}.{constructor.__qualname__}{signature}: '
            f'audit installed source and provide required values for {missing}')
    return constructor(**kwargs)


def installed_scheduler_sign_fixture(llm, *, scheduler_factory=None):
    """On-actor CPU-only fixture using installed Scheduler, Request and block pool.

    NEVER calls llm.step/generate, an executor, model runner, Ray or TPU execution.
    Configs are copied; every case constructs a NEW Scheduler and actual allocator.
    The allocator test hook reserves real free pool blocks, not a fabricated count.
    scheduler_factory, if needed for an unfamiliar constructor, accepts
    (copied_vllm_config, copied_kv_cache_config) and must construct a fresh installed
    Scheduler. It must not reuse ANY live request, cache manager, or block pool.

    Includes both signs, an already-scheduled victim/refund, two current-victim
    positions, and mixed/budget-exhausted traversal counterexamples. Returned token
    maps and counters are observations from installed schedule(), not predictions
    or hardware throughput measurements. A failed assertion rejects qualification.
    """
    from vllm import SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request, RequestStatus

    engine = llm.llm_engine
    live = engine.engine_core.engine_core.scheduler
    original_config = engine.vllm_config
    original_kv_config = live.kv_cache_config
    source = inspect.getsource(Scheduler.schedule)
    _, source_audit = transform_native_schedule(source)
    mro_source_hashes = {}
    for cls in type(live).__mro__:
        if cls is not object:
            class_source = inspect.getsource(cls)
            mro_source_hashes[f'{cls.__module__}.{cls.__qualname__}'] = (
                hashlib.sha256(class_source.encode()).hexdigest())
    block_size = original_config.cache_config.block_size
    if block_size != 256:
        raise FixtureDependencyError('This F3 fixture requires the mounted 256-token block geometry')

    def new_scheduler(token_budget=8):
        config = copy.deepcopy(original_config)
        kv_config = copy.deepcopy(original_kv_config)
        # Scheduler-only controls: no execution, connectors, speculation, cache
        # sharing or asynchronous/deferred ownership can contaminate the fixture.
        config.cache_config.enable_prefix_caching = False
        config.scheduler_config.policy = 'fcfs'
        config.scheduler_config.max_num_seqs = 8
        config.scheduler_config.max_num_batched_tokens = token_budget
        config.scheduler_config.long_prefill_token_threshold = 0
        for name in ('speculative_config', 'kv_transfer_config', 'ec_transfer_config',
                     'kv_events_config'):
            if hasattr(config, name):
                setattr(config, name, None)
        if hasattr(config.scheduler_config, 'async_scheduling'):
            config.scheduler_config.async_scheduling = False
        config.parallel_config.pipeline_parallel_size = 1
        # These are metadata-only CPU blocks, not accelerator memory allocations.
        kv_config.num_blocks = max(128, 32 * len(kv_config.kv_cache_groups) + 1)
        config.cache_config.num_gpu_blocks = kv_config.num_blocks
        if scheduler_factory is None:
            scheduler = _construct_known(Scheduler, {
                'vllm_config': config, 'kv_cache_config': kv_config,
                'structured_output_manager': None, 'block_size': block_size,
                'log_stats': False, 'include_finished_set': False,
                'has_kernel_block_size_mismatch': False})
        else:
            scheduler = scheduler_factory(config, kv_config)
        if not isinstance(scheduler, Scheduler) or scheduler is live:
            raise GuardInvariantError('Fixture factory must return a new installed Scheduler')
        if (scheduler.kv_cache_manager is live.kv_cache_manager
                or scheduler.kv_cache_manager.block_pool is live.kv_cache_manager.block_pool
                or scheduler.requests is live.requests or scheduler.running is live.running):
            raise GuardInvariantError('Fixture allocator/request state aliases the live engine')
        if scheduler.running or scheduler.waiting or scheduler.skipped_waiting or scheduler.requests:
            raise GuardInvariantError('Fixture scheduler is not initially empty')
        if scheduler.connector is not None or getattr(scheduler, 'ec_connector', None) is not None:
            raise FixtureDependencyError('Fixture constructor retained an active connector')
        if getattr(scheduler, 'defer_block_free', False):
            raise FixtureDependencyError('Fixture constructor retained deferred frees; disable in factory')
        if scheduler.num_lookahead_tokens or scheduler.num_spec_tokens:
            raise FixtureDependencyError('Fixture constructor retained speculation')
        if scheduler.max_num_scheduled_tokens != token_budget:
            raise FixtureDependencyError('Fixture constructor did not honor its token budget')
        return scheduler

    def request(request_id, prompt_length, computed=0, generated=0):
        params = SamplingParams(temperature=0.0, max_tokens=512, ignore_eos=True)
        req = _construct_known(Request, {
            'request_id': request_id, 'prompt_token_ids': [0] * prompt_length,
            'sampling_params': params, 'pooling_params': None,
            'arrival_time': 0.0 if request_id == 'A' else 1.0,
            'lora_request': None, 'cache_salt': None, 'mm_features': None,
            'prompt_embeds': None, 'block_hasher': None, 'priority': 0})
        if generated:
            req.append_output_token_ids([0] * generated)
        req.num_computed_tokens = computed
        req.is_prefill_chunk = computed < prompt_length
        req.status = RequestStatus.RUNNING
        return req

    def block_ids(scheduler, req):
        groups = scheduler.kv_cache_manager.get_blocks(req.request_id).get_block_ids()
        null = scheduler.kv_cache_manager.block_pool.null_block.block_id
        return sorted({block for group in groups for block in group if block != null})

    def demand(scheduler, req, tokens=1):
        manager = scheduler.kv_cache_manager
        total = req.num_computed_tokens
        return _construct_known(manager.coordinator.get_num_blocks_to_allocate, {
            'request_id': req.request_id, 'num_tokens': total + tokens,
            'new_computed_blocks': manager.empty_kv_cache_blocks.blocks,
            'num_encoder_tokens': 0, 'total_computed_tokens': total,
            'num_local_computed_tokens': total, 'num_tokens_main_model': total + tokens})

    results = []
    for name, scores, free_mode in (
            ('negative_A_scheduled_victim_refund', {'A': -1, 'B': 1}, 'one_short'),
            ('negative_B_current_victim', {'A': 1, 'B': -1}, 'one_short'),
            ('first_current_victim_native_break', {'A': -1, 'B': 1}, 'zero')):
        scheduler = new_scheduler()
        manager, pool = scheduler.kv_cache_manager, scheduler.kv_cache_manager.block_pool
        a, b = request('A', block_size), request('B', block_size)
        for req in (a, b):
            scheduler.requests[req.request_id] = req
            scheduler.running.append(req)
            allocated = manager.allocate_slots(req, block_size, num_lookahead_tokens=0)
            if allocated is None:
                raise FixtureDependencyError('Reduced pool cannot initialize two resident requests')
            req.num_computed_tokens = block_size
            req.append_output_token_ids([0])
            req.is_prefill_chunk = False
            assert req.num_in_flight_tokens == req.num_output_placeholders == 0
        before_blocks = {r.request_id: block_ids(scheduler, r) for r in (a, b)}
        demands = {r.request_id: demand(scheduler, r) for r in (a, b)}
        if not (len(before_blocks['A']) == len(before_blocks['B']) > 0
                and demands['A'] == demands['B'] > 0):
            raise FixtureDependencyError(
                f'Actual allocator crossing not equal/nonzero: blocks={before_blocks}, demand={demands}')
        target_free = sum(demands.values()) - 1 if free_mode == 'one_short' else 0
        reserve = pool.get_num_free_blocks() - target_free
        if reserve < 0:
            raise FixtureDependencyError('Reduced allocator pool has insufficient fixture headroom')
        held_blocks = pool.get_new_blocks(reserve) if reserve else []
        free_before = pool.get_num_free_blocks()
        assert free_before == target_free
        selection = []
        before_update = {}
        original_update = scheduler._update_after_schedule

        def capture_update(output):
            frame = inspect.currentframe().f_back
            try:
                local = frame.f_locals
                before_update.update(
                    token_budget=local['token_budget'],
                    new_block_request_ids=sorted(local['req_to_new_blocks']),
                    scheduled_running_ids=[r.request_id for r in local['scheduled_running_reqs']],
                    spec_request_ids=sorted(local['scheduled_spec_decode_tokens']),
                    encoder_request_ids=sorted(local['scheduled_encoder_inputs']))
            finally:
                del frame
            return original_update(output)

        scheduler._update_after_schedule = capture_update

        def select(owner, current, tokens):
            frame = inspect.currentframe().f_back.f_back
            try:
                scheduled_before = dict(frame.f_locals['num_scheduled_tokens'])
            finally:
                del frame
            victim = min(owner.running, key=lambda req: scores[req.request_id])
            selection.append({'running': [r.request_id for r in owner.running],
                              'tail': owner.running[-1].request_id,
                              'current': current.request_id, 'victim': victim.request_id,
                              'free_blocks': pool.get_num_free_blocks(),
                              'current_demand': demand(owner, current, tokens),
                              'scheduled_before': scheduled_before,
                              'victim_blocks_at_selection': len(block_ids(owner, victim))})
            return victim

        original_schedule = scheduler.schedule
        with NativeSelectorSeam(scheduler, select) as seam:
            output = scheduler.schedule()
            seam_audit = seam.audit
        assert scheduler.schedule == original_schedule
        victim = 'A' if scores['A'] < scores['B'] else 'B'
        survivor = 'B' if victim == 'A' else 'A'
        expected_map = {survivor: 1} if free_mode == 'one_short' else {}
        observed_map = dict(output.num_scheduled_tokens)
        assert len(selection) == 1 and selection[0]['victim'] == victim, selection
        assert set(output.preempted_req_ids) == {victim}, output.preempted_req_ids
        assert observed_map == expected_map, (name, observed_map, expected_map)
        assert output.total_num_scheduled_tokens == sum(expected_map.values())
        assert before_update['token_budget'] == scheduler.max_num_scheduled_tokens - sum(expected_map.values())
        assert before_update['new_block_request_ids'] == sorted(expected_map)
        assert before_update['scheduled_running_ids'] == list(expected_map)
        assert before_update['spec_request_ids'] == before_update['encoder_request_ids'] == []
        assert [r.request_id for r in scheduler.waiting] == [victim]
        assert not scheduler.skipped_waiting
        assert [r.request_id for r in scheduler.running] == [survivor]
        victim_req, survivor_req = (a, b) if victim == 'A' else (b, a)
        assert victim_req.status == RequestStatus.PREEMPTED and victim_req.num_computed_tokens == 0
        assert survivor_req.num_computed_tokens == block_size + int(bool(expected_map))
        assert len(block_ids(scheduler, victim_req)) == 0
        if name == 'negative_A_scheduled_victim_refund':
            assert selection[0]['scheduled_before'] == {'A': 1}
            assert selection[0]['victim_blocks_at_selection'] == len(before_blocks['A']) + demands['A']
        else:
            assert selection[0]['current'] == victim
        final_blocks = {r.request_id: block_ids(scheduler, r) for r in (a, b)}
        expected_free = free_before + sum(map(len, before_blocks.values())) - sum(map(len, final_blocks.values()))
        assert pool.get_num_free_blocks() == expected_free
        results.append({'case': name, 'scores': scores, 'running_before': ['A', 'B'],
                        'blocks_before': before_blocks, 'demand_before': demands,
                        'free_before': free_before, 'allocator_reserved_blocks': len(held_blocks),
                        'selection': selection, 'num_scheduled_tokens': observed_map,
                        'preempted_req_ids': sorted(output.preempted_req_ids),
                        'native_before_update': before_update,
                        'computed_after': {r.request_id: r.num_computed_tokens for r in (a, b)},
                        'blocks_after': final_blocks, 'free_after': pool.get_num_free_blocks(),
                        'seam': seam_audit})

    traversal = []
    for case, specs, token_budget in (
            ('mixed_prefills', ((1536, 0, 0), (1536, 0, 0)), 2048),
            ('decode_recompute', ((256, 256, 1), (1536, 0, 0)), 1024),
            ('decode_budget_exhaustion', ((256, 256, 1), (256, 256, 1)), 1),
            ('homogeneous_no_preemption', ((256, 256, 1), (256, 256, 1)), 8)):
        paired = []
        for reverse in (False, True):
            scheduler = new_scheduler(token_budget)
            reqs = [request(name, *spec) for name, spec in zip(('A', 'B'), specs)]
            for req, (_, computed, _) in zip(reqs, specs):
                scheduler.requests[req.request_id] = req
                if computed:
                    req.num_computed_tokens = 0
                    assert scheduler.kv_cache_manager.allocate_slots(req, computed) is not None
                    req.num_computed_tokens = computed
            scheduler.running[:] = list(reversed(reqs)) if reverse else reqs
            output = scheduler.schedule()
            assert not output.preempted_req_ids, 'Traversal negative control unexpectedly preempted'
            paired.append(dict(output.num_scheduled_tokens))
        if case == 'homogeneous_no_preemption':
            assert paired == [{'A': 1, 'B': 1}, {'B': 1, 'A': 1}], paired
        else:
            assert paired[0] != paired[1], (case, paired)
        if case == 'mixed_prefills':
            assert paired == [{'A': 1536, 'B': 512}, {'B': 1536, 'A': 512}], paired
        traversal.append({'case': case, 'forward': paired[0], 'reverse': paired[1],
                          'equal_by_request': paired[0] == paired[1]})
    return {'kind': 'installed_scheduler_CPU_fixture_NOT_throughput_measurement',
            'live_engine_execution_calls': 0, 'source_audit': source_audit,
            'live_scheduler_mro_source_sha256': mro_source_hashes,
            'scheduler_class': f'{Scheduler.__module__}.{Scheduler.__qualname__}',
            'scheduler_signature': str(inspect.signature(Scheduler)),
            'request_signature': str(inspect.signature(Request)),
            'sign_cases': results, 'traversal_controls': traversal,
            'universal_traversal_isolation': False}

