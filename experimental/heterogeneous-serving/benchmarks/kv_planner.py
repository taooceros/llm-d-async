"""Exact sparse decode calendars and bounded, CPU-only timetable optimization.

The supplied d is a legal *total* service count, not necessarily an oracle value.
Every resident receives one useful token per epoch; q(a)=ceil((p+a+1)/L).
These are surrogate service calendars, never hardware timing measurements or an
execution certificate for an engine which skips residents or delays KV frees.

``surrogate_J`` is an integer in 1e-4 ms units. Starts and queue_indices refer to
input positions; ties use arrival_index, then input position. ``singleton_tail``
is the final batch-one drain *after the last admission*, as in the residual-work
identity. Empty gaps are removed by both public functions. Calendar rows contain
post-event occupancy, including the final zero-occupancy completion boundary.
"""

import heapq
import json
import math
import multiprocessing
import random
import statistics
import time
from collections import deque
from fractions import Fraction


_PAD = (0,) + tuple(max(8, 1 << (b - 1).bit_length()) for b in range(1, 513))
_H = (0,) + tuple(67500 + 514 * _PAD[b] for b in range(1, 513))


class _Expired(Exception):
    pass


def _check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise _Expired


def _integer(value, name, minimum=0, maximum=None):
    if (not isinstance(value, int) or isinstance(value, bool)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError('%s must be an integer in [%s, %s]' %
                         (name, minimum, maximum if maximum is not None else 'inf'))
    return value


def _prefix_area(x, block_size):
    q, r = divmod(x, block_size)
    return block_size * q * (q + 1) // 2 + r * (q + 1)


class _Job:
    __slots__ = ('p', 'd', 'arrival', 'first', 'last', 'area', 'events')

    def __init__(self, p, d, arrival, block_size):
        self.p, self.d, self.arrival = p, d, arrival
        self.first = p // block_size + 1
        self.last = (p + d + block_size - 1) // block_size
        self.area = _prefix_area(p + d, block_size) - _prefix_area(p, block_size)
        self.events = ((0, self.first, 1),) + tuple(
            (m * block_size - p, 1, 0) for m in range(self.first, self.last)
        ) + ((d, -self.last, -1),)


class _Problem:
    def __init__(self, manifest, usable_blocks, block_size, batch_limit):
        self.capacity = _integer(usable_blocks, 'usable_blocks', 1)
        self.block_size = _integer(block_size, 'block_size', 1)
        self.batch_limit = _integer(batch_limit, 'batch_limit', 1, 512)
        self.jobs = []
        self.area = self.tokens = self.dmax = 0
        self.individually_feasible = True
        try:
            entries = iter(manifest)
        except TypeError as exc:
            raise ValueError('manifest must contain [p, d, arrival_index] triples') from exc
        for index, entry in enumerate(entries):
            try:
                p, d, arrival = entry
            except (TypeError, ValueError) as exc:
                raise ValueError('manifest[%d] must be a triple' % index) from exc
            p = _integer(p, 'manifest[%d].p' % index, 1)
            d = _integer(d, 'manifest[%d].d' % index, 1)
            arrival = _integer(arrival, 'manifest[%d].arrival_index' % index)
            last = (p + d + block_size - 1) // block_size
            # Do not materialize an arbitrarily large, individually impossible profile.
            if last > usable_blocks:
                self.individually_feasible = False
                self.jobs.append((p, d, arrival))
                continue
            job = _Job(p, d, arrival, block_size)
            self.jobs.append(job)
            self.area += job.area
            self.tokens += d
            self.dmax = max(self.dmax, d)
        self.order = (sorted(range(len(self.jobs)),
                             key=lambda i: (self.jobs[i].arrival, i))
                      if self.individually_feasible else [])


def _evaluate(problem, starts, deadline=None):
    """Grouped O(n + D/L) events; no visit to individual service epochs."""
    events = {}
    for index, (job, start) in enumerate(zip(problem.jobs, starts)):
        if index % 32 == 0:
            _check(deadline)
        for k, (age, dr, db) in enumerate(job.events):
            if k % 128 == 0:
                _check(deadline)
            epoch = start + age
            previous = events.get(epoch)
            if previous is None:
                events[epoch] = [dr, db]
            else:
                previous[0] += dr
                previous[1] += db
    _check(deadline)
    times = sorted(events)
    _check(deadline)
    blocks = batch = previous_time = gap = 0
    area = tokens = padding = active = low_batch = peak = peak_batch = 0
    canonical = {}
    calendar = []
    for index, epoch in enumerate(times):
        if index % 128 == 0:
            _check(deadline)
        dt = epoch - previous_time
        area += dt * blocks
        tokens += dt * batch
        if batch:
            active += dt
            padding += dt * _PAD[batch]
            if batch <= 8:
                low_batch += dt
        else:
            gap += dt
        dr, db = events[epoch]
        blocks += dr
        batch += db
        if not (0 <= blocks <= problem.capacity and 0 <= batch <= problem.batch_limit):
            return None
        peak = max(peak, blocks)
        peak_batch = max(peak_batch, batch)
        new_epoch = epoch - gap
        canonical[epoch] = new_epoch
        row = [new_epoch, blocks, batch]
        if calendar and calendar[-1][0] == new_epoch:
            calendar[-1] = row
        else:
            calendar.append(row)
        previous_time = epoch
    assert blocks == batch == 0
    assert area == problem.area and tokens == problem.tokens
    new_starts = []
    for index, start in enumerate(starts):
        if index % 128 == 0:
            _check(deadline)
        new_starts.append(canonical[start])
    tq = max(new_starts, default=0)
    r1 = r2 = 0
    residual_min = None
    for index, (job, start) in enumerate(zip(problem.jobs, new_starts)):
        if index % 128 == 0:
            _check(deadline)
        remaining = start + job.d - tq
        if remaining > 0:
            residual_min = remaining if residual_min is None else min(residual_min, remaining)
            if remaining >= r1:
                r2, r1 = r1, remaining
            elif remaining > r2:
                r2 = remaining
    _check(deadline)
    return {
        'starts': new_starts,
        'surrogate_J': 67500 * active + 514 * padding,
        'peak_blocks': peak,
        'peak_batch': peak_batch,
        'area': area,
        'total_decode_tokens': tokens,
        'singleton_tail': r1 - r2,
        'active_epochs': active,
        'padding_sum': padding,
        'makespan': active,
        'queue_empty_epoch': tq,
        'residual_max': r1,
        'residual_second_max': r2,
        'low_batch_epochs': low_batch,
        'final_completion_spread': r1 - (residual_min or 0),
        'calendar': calendar,
    }


class _Node:
    """AVL node with the sums and maximum prefixes of both resource deltas."""
    __slots__ = ('key', 'dr', 'db', 'left', 'right', 'height',
                 'sum_r', 'sum_b', 'peak_r', 'peak_b')

    def __init__(self, key, dr, db):
        self.key, self.dr, self.db = key, dr, db
        self.left = self.right = None
        self.height = 1
        self.sum_r, self.sum_b = dr, db
        self.peak_r, self.peak_b = max(0, dr), max(0, db)


def _pull(node):
    left, right = node.left, node.right
    r = (left.sum_r if left else 0) + node.dr
    b = (left.sum_b if left else 0) + node.db
    node.sum_r = r + (right.sum_r if right else 0)
    node.sum_b = b + (right.sum_b if right else 0)
    node.peak_r = max(left.peak_r if left else 0, r + (right.peak_r if right else 0))
    node.peak_b = max(left.peak_b if left else 0, b + (right.peak_b if right else 0))
    node.height = 1 + max(left.height if left else 0, right.height if right else 0)
    return node


def _rotate_left(node):
    top = node.right
    node.right, top.left = top.left, node
    _pull(node)
    return _pull(top)


def _rotate_right(node):
    top = node.left
    node.left, top.right = top.right, node
    _pull(node)
    return _pull(top)


def _balance(node):
    _pull(node)
    left, right = node.left, node.right
    difference = (left.height if left else 0) - (right.height if right else 0)
    if difference > 1:
        if (left.left.height if left.left else 0) < (left.right.height if left.right else 0):
            node.left = _rotate_left(left)
        return _rotate_right(node)
    if difference < -1:
        if (right.right.height if right.right else 0) < (right.left.height if right.left else 0):
            node.right = _rotate_right(right)
        return _rotate_left(node)
    return node


def _remove_min(node):
    if node.left is None:
        return node.right
    node.left = _remove_min(node.left)
    return _balance(node)


def _add(node, key, dr, db):
    if node is None:
        return _Node(key, dr, db)
    if key < node.key:
        node.left = _add(node.left, key, dr, db)
    elif key > node.key:
        node.right = _add(node.right, key, dr, db)
    else:
        node.dr += dr
        node.db += db
        if node.dr == node.db == 0:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            successor = node.right
            while successor.left is not None:
                successor = successor.left
            node.key, node.dr, node.db = successor.key, successor.dr, successor.db
            node.right = _remove_min(node.right)
    return _balance(node)


class _Calendar:
    def __init__(self):
        self.root = None
        self.origin_r = self.origin_b = 0

    def profile(self, job, start, sign, deadline):
        for age, dr, db in job.events:
            _check(deadline)
            self.root = _add(self.root, start + age, sign * dr, sign * db)

    def fits(self, capacity, batch_limit):
        root = self.root
        return (self.origin_r + (root.peak_r if root else 0) <= capacity
                and self.origin_b + (root.peak_b if root else 0) <= batch_limit)

    def advance(self, epoch, deadline):
        # The removed prefix is retained as the offset for the certified suffix.
        while self.root is not None:
            _check(deadline)
            first = self.root
            while first.left is not None:
                first = first.left
            if first.key > epoch:
                break
            self.origin_r += first.dr
            self.origin_b += first.db
            self.root = _remove_min(self.root)

    def snapshot(self, epoch, deadline):
        """One shared resident view per fill round, never a candidate calendar copy."""
        blocks, batch = self.origin_r, self.origin_b
        rows = [[epoch, blocks, batch]]
        stack = []
        node = self.root
        while stack or node is not None:
            _check(deadline)
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            blocks += node.dr
            batch += node.db
            if node.key <= epoch:
                rows[0][1:] = blocks, batch
            else:
                rows.append([node.key, blocks, batch])
            node = node.right
        return rows


def _charge(job, start, rows, capacity, target, deadline):
    """Exact rational marginal price, merged over all growth/resident intervals."""
    end = start + job.d
    epoch = start
    row_index = 0
    growth_index = 1
    q = job.first
    delta_j = 0
    pressure = {}
    while epoch < end:
        _check(deadline)
        _, blocks, batch = rows[row_index]
        resident_end = rows[row_index + 1][0] if row_index + 1 < len(rows) else end
        growth_end = start + job.events[growth_index][0]
        following = min(end, resident_end, growth_end)
        dt = following - epoch
        delta_j += dt * (_H[batch + 1] - _H[batch])
        denominator = capacity + 1 - blocks
        pressure[denominator] = pressure.get(denominator, 0) + dt * q
        epoch = following
        if epoch == resident_end and row_index + 1 < len(rows):
            row_index += 1
        if epoch == growth_end and epoch < end:
            q += 1
            growth_index += 1
    price = Fraction(delta_j)
    h_target = _H[target]
    for denominator, amount in pressure.items():
        _check(deadline)
        price += Fraction(h_target * amount, denominator)
    return price / job.d


class _Candidates:
    def __init__(self, jobs):
        self.placed = bytearray(len(jobs))
        self.views = [
            [(i, i) for i in range(len(jobs))],
            [(-job.d, job.arrival, i) for i, job in enumerate(jobs)],
            [(Fraction(job.area, job.d), job.arrival, i) for i, job in enumerate(jobs)],
        ]
        for view in self.views:
            heapq.heapify(view)

    def candidates(self, deadline):
        candidates = []
        seen = set()
        for view, count in zip(self.views, (16, 4, 4)):
            taken = []
            while view and len(taken) < count:
                _check(deadline)
                item = heapq.heappop(view)
                index = item[-1]
                if self.placed[index]:
                    continue
                taken.append(item)
                if index not in seen:
                    candidates.append(index)
                    seen.add(index)
            for item in taken:
                heapq.heappush(view, item)
        return candidates


def _seed(problem, deadline, stats):
    jobs = problem.jobs
    target = min(128, problem.batch_limit)
    goal = max(problem.dmax, (problem.tokens + target - 1) // target,
               (problem.area + problem.capacity - 1) // problem.capacity)
    pending = _Candidates(jobs)
    calendar = _Calendar()
    completions = []
    starts = [0] * len(jobs)
    epoch = 0
    remaining = len(jobs)
    while remaining:
        _check(deadline)
        rows = calendar.snapshot(epoch, deadline)
        fits = []
        if rows[0][2] < target:
            for index in pending.candidates(deadline):
                _check(deadline)
                job = jobs[index]
                calendar.profile(job, epoch, 1, deadline)
                fit = calendar.fits(problem.capacity, target)
                calendar.profile(job, epoch, -1, deadline)
                stats['seed_candidates'] += 1
                if fit:
                    slack = goal - epoch - job.d
                    critical = 2 * job.d >= goal - epoch or slack <= problem.block_size
                    fits.append((index, critical, slack))
            if fits:
                critical = [entry for entry in fits if entry[1]]
                if critical:
                    slack = min(entry[2] for entry in critical)
                    eligible = [entry[0] for entry in critical if entry[2] == slack]
                else:
                    eligible = [entry[0] for entry in fits]
                # A dominated deadline/slack cannot win: do not compute its price.
                choice = min(eligible, key=lambda i: (
                    _charge(jobs[i], epoch, rows, problem.capacity, target, deadline),
                    jobs[i].arrival, i))
                calendar.profile(jobs[choice], epoch, 1, deadline)
                starts[choice] = epoch
                pending.placed[choice] = 1
                remaining -= 1
                stats['seed_placed'] += 1
                heapq.heappush(completions, (epoch + jobs[choice].d, choice))
                continue
        while completions and completions[0][0] <= epoch:
            _check(deadline)
            heapq.heappop(completions)
        assert completions, 'an individually fitting candidate must fit an empty calendar'
        epoch = completions[0][0]
        calendar.advance(epoch, deadline)
    result = _evaluate(problem, starts, deadline)
    assert result is not None
    return result


class _Random(random.Random):
    def __init__(self, seed):
        super().__init__(seed)
        self.draws = 0

    def random(self):
        self.draws += 1
        return super().random()

    def getrandbits(self, k):
        self.draws += 1
        return super().getrandbits(k)


def _relocate(problem, current, proposal, rng):
    jobs = problem.jobs
    index = rng.randrange(len(jobs))
    old = proposal[index]
    category = rng.randrange(8)
    if category < 4:
        new = old + (-1, 1, -problem.block_size, problem.block_size)[category]
    elif category == 4:
        new = 0
    elif category == 5:
        new = current['makespan'] - jobs[index].d
    elif category == 6:
        other = rng.randrange(len(jobs))
        new = proposal[other] + (jobs[other].d if rng.randrange(2) else 0)
    else:
        new = rng.randrange(max(0, current['makespan'] + problem.dmax - jobs[index].d) + 1)
    proposal[index] = max(0, new)


def _proposal(problem, current, rng, stats, deadline):
    starts = current['starts']
    proposed = starts.copy()
    jobs = problem.jobs
    draw = rng.randrange(10)
    move = 'relocate' if draw < 4 else 'swap' if draw < 6 else 'permute' if draw < 8 else 'tail_pair'
    stats['moves'][move] += 1
    if move == 'relocate' or (move == 'swap' and len(jobs) < 2):
        _relocate(problem, current, proposed, rng)
    elif move == 'swap':
        i, j = rng.sample(range(len(jobs)), 2)
        proposed[i], proposed[j] = proposed[j], proposed[i]
    elif move == 'permute':
        peaks = []
        rows = current['calendar']
        for k in range(len(rows) - 1):
            if k % 128 == 0:
                _check(deadline)
            if rows[k][1] == current['peak_blocks'] and rows[k + 1][0] > rows[k][0]:
                peaks.append(rows[k][0])
        epoch = rng.choice(peaks)
        overlap = []
        for i, job in enumerate(jobs):
            if i % 128 == 0:
                _check(deadline)
            if starts[i] <= epoch < starts[i] + job.d:
                overlap.append(i)
        if len(overlap) < 2:
            stats['relocate_fallbacks'] += 1
            _relocate(problem, current, proposed, rng)
        else:
            chosen = rng.sample(overlap, min(4, len(overlap)))
            values = [starts[i] for i in chosen]
            rng.shuffle(values)
            for i, value in zip(chosen, values):
                proposed[i] = value
    else:
        # Stable-ID tie breaking makes the unique last-finisher selection reproducible.
        for rank, i in enumerate(problem.order):
            if rank % 128 == 0:
                _check(deadline)
            if starts[i] + jobs[i].d == current['makespan']:
                break
        shorter = []
        for j, job in enumerate(jobs):
            if j % 128 == 0:
                _check(deadline)
            if job.d < jobs[i].d and starts[j] + job.d < current['makespan']:
                shorter.append(j)
        if not shorter:
            return None
        j = rng.choice(shorter)
        options = [s for s in (0, starts[i] - problem.block_size,
                               starts[j] + jobs[j].d - jobs[i].d) if s >= 0]
        earlier = rng.choice(options)
        filler = earlier + jobs[i].d - jobs[j].d
        if earlier >= starts[i] or filler < 0:
            return None
        proposed[i], proposed[j] = earlier, filler
    _check(deadline)
    return proposed


def _better(candidate, incumbent, order):
    candidate_key = candidate['surrogate_J'], candidate['singleton_tail']
    incumbent_key = incumbent['surrogate_J'], incumbent['singleton_tail']
    if candidate_key != incumbent_key:
        return candidate_key < incumbent_key
    return tuple(candidate['starts'][i] for i in order) < tuple(incumbent['starts'][i] for i in order)


def _new_stats():
    return {
        'seed_complete': False, 'seed_aborted': False, 'seed_placed': 0,
        'seed_candidates': 0, 'seed_s': 0.0, 'proposals': 0,
        'completed_evaluations': 0, 'feasible_evaluations': 0,
        'infeasible_evaluations': 0, 'rejected_evaluations': 0,
        'aborted_evaluations': 0, 'invalid_proposals': 0,
        'unchanged_proposals': 0, 'accepted_moves': 0, 'improvements': 0,
        'random_draws': 0, 'relocate_fallbacks': 0,
        'moves': dict.fromkeys(('relocate', 'swap', 'permute', 'tail_pair'), 0),
        'quota_exhausted': False, 'worker_terminated': False,
    }


def _optimize(problem, incumbent, seed_deadline, cutoff, search, seed, connection):
    stats = _new_stats()
    began = time.monotonic()
    try:
        seeded = _seed(problem, seed_deadline, stats)
        stats['seed_complete'] = True
        stats['seed_J'] = seeded['surrogate_J']
        if _better(seeded, incumbent, problem.order):
            incumbent = seeded
            stats['improvements'] += 1
            connection.send(('best', incumbent))
    except _Expired:
        stats['seed_aborted'] = True
    stats['seed_s'] = time.monotonic() - began
    connection.send(('stats', stats))
    if not search:
        connection.send(('done', stats))
        return
    current = incumbent
    rng = _Random(seed)
    started_search = time.monotonic()
    search_span = max(1e-12, cutoff - started_search)
    tau0 = max(_H[1] * problem.block_size, incumbent['surrogate_J'] * 0.01)
    recent = deque(maxlen=32)
    cycles = 0
    while stats['completed_evaluations'] < 10000:
        remaining = cutoff - time.monotonic()
        if remaining <= 0:
            break
        if cycles >= 8 and int(remaining / (2 * max(recent))) < 1:
            stats['quota_exhausted'] = True
            break
        cycle_start = time.monotonic()
        stats['proposals'] += 1
        evaluating = False
        try:
            proposed = _proposal(problem, current, rng, stats, cutoff)
            if proposed is None:
                stats['invalid_proposals'] += 1
            elif proposed == current['starts']:
                stats['unchanged_proposals'] += 1
            else:
                evaluating = True
                connection.send(('evaluating',))
                candidate = _evaluate(problem, proposed, cutoff)
                stats['completed_evaluations'] += 1
                evaluating = False
                if candidate is None:
                    stats['infeasible_evaluations'] += 1
                    stats['rejected_evaluations'] += 1
                else:
                    stats['feasible_evaluations'] += 1
                    delta = candidate['surrogate_J'] - current['surrogate_J']
                    accept = delta <= 0
                    if not accept:
                        x = min(1.0, (time.monotonic() - started_search) / search_span)
                        tau = tau0 * (0.001 ** x)
                        u = rng.random()
                        while u == 0:
                            _check(cutoff)
                            u = rng.random()
                        accept = math.log(u) < -delta / tau
                    if accept:
                        current = candidate
                        stats['accepted_moves'] += 1
                    else:
                        stats['rejected_evaluations'] += 1
                    if _better(candidate, incumbent, problem.order):
                        incumbent = candidate
                        stats['improvements'] += 1
                        connection.send(('best', incumbent))
        except _Expired:
            if evaluating:
                stats['aborted_evaluations'] += 1
            stats['random_draws'] = rng.draws
            break
        elapsed = max(time.monotonic() - cycle_start, 1e-12)
        recent.append(elapsed)
        cycles += 1
        stats['random_draws'] = rng.draws
        connection.send(('cycle', elapsed, stats))
    stats['search_s'] = time.monotonic() - started_search
    stats['quota_guard_s'] = 2 * max(recent) if recent else 0.0
    connection.send(('done', stats))


def _worker(problem, incumbent, seed_deadline, cutoff, search, seed, sender, receiver):
    receiver.close()
    try:
        _optimize(problem, incumbent, seed_deadline, cutoff, search, seed, sender)
    except Exception as exc:
        sender.send(('error', '%s: %s' % (type(exc).__name__, exc)))
    finally:
        sender.close()


def _finish(problem, result, stats, entered, cycle_times=()):
    result['queue_indices'] = sorted(range(len(problem.jobs)),
                                     key=lambda i: (result['starts'][i], problem.jobs[i].arrival, i))
    stats['cycle_median_s'] = statistics.median(cycle_times) if cycle_times else 0.0
    stats['cycle_max_s'] = max(cycle_times, default=0.0)
    stats['completed_cycles'] = len(cycle_times)
    result['stats'] = stats
    result['planning_s'] = time.monotonic() - entered
    # Include the actual JSON encoding cost in the planner clock, not just search.
    serialization_start = time.monotonic()
    json.dumps(result, separators=(',', ':'), allow_nan=False)
    stats['serialization_s'] = time.monotonic() - serialization_start
    result['planning_s'] = time.monotonic() - entered
    if 'budget_s' in stats:
        stats['budget_overrun_s'] = max(0.0, result['planning_s'] - stats['budget_s'])
    return result


def evaluate(manifest, starts, usable_blocks, block_size=256, batch_limit=128):
    """Return exact canonicalized metrics, None if infeasible; invalid inputs raise.

    Empty manifests are legal. p and d must be positive integers, arrival_index
    and starts nonnegative integers, and batch_limit must be in [1, 512].
    Arrival indices need not be dense; input positions disambiguate ties.
    """
    entered = time.monotonic()
    problem = _Problem(manifest, usable_blocks, block_size, batch_limit)
    try:
        starts = list(starts)
    except TypeError as exc:
        raise ValueError('starts must be an iterable of nonnegative integers') from exc
    if len(starts) != len(problem.jobs):
        raise ValueError('starts must have one entry per manifest request')
    for index, start in enumerate(starts):
        _integer(start, 'starts[%d]' % index)
    if not problem.individually_feasible:
        return None
    result = _evaluate(problem, starts)
    if result is None:
        return None
    return _finish(problem, result, {'evaluation_only': True}, entered)


def plan(manifest, usable_blocks, block_size=256, batch_limit=128,
         search=True, budget_s=None, seed=42):
    """Build a full feasible incumbent before bounded pressure packing/search.

    Budgets include validation, profiles, serial evaluation, process startup,
    optimization, IPC and JSON serialization. Default hard budgets are 55s for
    n<=1763 and 110s otherwise; seed/search cutoffs are 15/53 and 30/106s after
    entry. Explicit budgets scale both cutoffs proportionally. A budget smaller
    than mandatory validation/serial output cannot be met; return that complete
    serial plan and report the overrun rather than emit a partial plan.

    Only the CPU optimization child is terminated at the search cutoff. The
    parent always owns the last fully evaluated incumbent. Linux fork also works
    when this module was deployed by value, without importing it in the child.
    """
    entered = time.monotonic()
    if not isinstance(search, bool):
        raise ValueError('search must be a boolean')
    _integer(seed, 'seed')
    if budget_s is not None:
        if (isinstance(budget_s, bool) or not isinstance(budget_s, (int, float))
                or not math.isfinite(budget_s) or budget_s < 0):
            raise ValueError('budget_s must be a finite nonnegative number')
        budget_s = float(budget_s)
    problem = _Problem(manifest, usable_blocks, block_size, batch_limit)
    if not problem.individually_feasible:
        return None
    if budget_s is None:
        budget_s = 55.0 if len(problem.jobs) <= 1763 else 110.0
    seed_deadline = entered + budget_s * 15 / 55
    cutoff = entered + budget_s * 53 / 55
    starts = []
    epoch = 0
    for job in problem.jobs:
        starts.append(epoch)
        epoch += job.d
    incumbent = _evaluate(problem, starts)
    assert incumbent is not None
    initialization_s = time.monotonic() - entered
    stats = _new_stats()
    cycle_times = []
    evaluating = False
    worker_done = False
    if problem.jobs and time.monotonic() < seed_deadline:
        # No accelerator handles or work are sent to this child. Only this exact
        # process handle is ever terminated; no process-group signal is used.
        receiver = sender = process = None
        try:
            context = multiprocessing.get_context('fork')
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(target=_worker, args=(
                problem, incumbent, seed_deadline, cutoff, search, seed, sender, receiver),
                name='kv-calendar-cpu-planner', daemon=True)
            process.start()
            sender.close()
            while True:
                remaining = cutoff - time.monotonic()
                if remaining <= 0:
                    break
                if receiver.poll(min(0.02, remaining)):
                    try:
                        message = receiver.recv()
                    except EOFError:
                        break
                    kind = message[0]
                    if kind == 'best':
                        incumbent = message[1]
                    elif kind == 'evaluating':
                        evaluating = True
                    elif kind in ('stats', 'done'):
                        stats = message[1]
                        evaluating = False
                        if kind == 'done':
                            worker_done = True
                            break
                    elif kind == 'cycle':
                        cycle_times.append(message[1])
                        stats = message[2]
                        evaluating = False
                    elif kind == 'error':
                        stats['worker_error'] = message[1]
                        break
                elif not process.is_alive():
                    break
        except (OSError, ValueError, RuntimeError, AssertionError) as exc:
            stats['worker_error'] = '%s: %s' % (type(exc).__name__, exc)
        finally:
            if process is not None and process.pid is not None:
                # A stalled worker must not consume the serialization reserve
                # in a graceful join after the search cutoff.
                grace = min(0.005, max(0.0, cutoff - time.monotonic())) if worker_done else 0
                process.join(timeout=grace)
                if process.is_alive():
                    stats['worker_terminated'] = True
                    if not stats['seed_complete']:
                        stats['seed_aborted'] = True
                    if evaluating:
                        stats['aborted_evaluations'] += 1
                    stats['interrupted_cycle_counts_complete'] = False
                    process.terminate()
                    process.join(timeout=min(0.1, max(0.0, entered + budget_s - time.monotonic())))
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=0.05)
                if not process.is_alive():
                    stats['worker_exitcode'] = process.exitcode
                    process.close()
            if receiver is not None:
                receiver.close()
            if sender is not None:
                sender.close()
    stats.update({
        'budget_s': budget_s, 'initialization_s': initialization_s,
        'seed_cutoff_s': budget_s * 15 / 55, 'search_cutoff_s': budget_s * 53 / 55,
        'search_enabled': search, 'seed': seed,
        'initialization_exhausted_seed_budget': initialization_s >= budget_s * 15 / 55,
        'serial_J': _H[1] * problem.tokens,
    })
    return _finish(problem, incumbent, stats, entered, cycle_times)
