"""Legal C/O decode futures and a shared KV trajectory certificate.

The service contract is one generated token per ready resident per logical epoch,
with next-token allocation charged before completion cleanup. Pending prefill or
recomputation receives only a constant terminal reservation, never release credit.
Track O is the known-public-law arm; completed observations do not refit its law.
"""

from dataclasses import dataclass, field
import hashlib
import math
from statistics import NormalDist
from types import MappingProxyType
import numpy as np
import time


_REQUEST_FIELDS = frozenset(('index', 'p', 'arrival_index'))
_DRAW_FIELDS = frozenset((
    'n', 'sigma', 'median', 'd_max', 'rho', 'catalogue', 'min_prompt_tokens',
    'prompt_median', 'prompt_sigma', 'prompt_max',
))
_SQRT_TWO = math.sqrt(2.0)
_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _probability(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be a finite probability in [0, 1]')
    return float(value)


def _copy_requests(requests, track):
    # Exact builtin types prevent subclasses from carrying hidden instance state.
    assert type(requests) in (list, tuple, dict), 'Requests must be plain public containers'
    rows = requests.values() if type(requests) is dict else requests
    allowed = _REQUEST_FIELDS | {'d'} if track == 'C' else _REQUEST_FIELDS
    copied = []
    seen = set()
    for row in rows:
        assert type(row) is dict, 'Each request must be a plain public dictionary'
        assert all(type(key) is str for key in row), 'Request field names must be plain strings'
        assert set(row) == allowed, f'Track {track} request fields must be {sorted(allowed)}'
        assert all(type(value) is int for value in row.values()), (
            'Request fields must be plain integers, with no nested or attached state')
        item = {key: _integer(row[key], key) for key in _REQUEST_FIELDS}
        if item['index'] in seen:
            raise ValueError(f'Duplicate request index {item["index"]}')
        seen.add(item['index'])
        if track == 'C':
            item['d'] = _integer(row['d'], 'd', 1)
        copied.append(MappingProxyType(item))
    if type(requests) is dict:
        assert all(type(key) is int and key == row['index']
                   for key, row in requests.items()), 'Request mapping keys must be public indices'
    return tuple(copied)


def _copy_law(law):
    assert type(law) is dict, 'The public draw law must be a plain dictionary'
    assert all(type(key) is str for key in law), 'Public draw field names must be plain strings'
    assert set(law) <= _DRAW_FIELDS, 'Unknown or secret field in public draw law'
    assert all(type(value) in (int, float, str, type(None)) for value in law.values()), (
        'The public draw law must have scalar values, with no nested or attached state')
    return MappingProxyType({key: law[key] for key in _DRAW_FIELDS if key in law})


def _log_mills_ratio(z):
    """Log(Q(z)/phi(z)); Laplace's continued fraction is fast for z >= 8."""
    tail = 0.0
    for numerator in range(32, 0, -1):
        tail = numerator / (z + tail)
    return -math.log(z + tail)


def _log_normal_survival(z):
    if z < 8.0:
        return math.log(0.5 * math.erfc(z / _SQRT_TWO))
    return -0.5 * z * z - _LOG_SQRT_TWO_PI + _log_mills_ratio(z)


@dataclass(frozen=True, slots=True)
class _SurvivalSums:
    cap: int
    mu: float
    sigma: float
    _log_sums: tuple[float, ...] | None = field(default=None, repr=False, compare=False)
    _log_survivals: tuple[float, ...] | None = field(default=None, repr=False, compare=False)

    def interval(self, start, stop, given):
        # Immutable tuples retain the normalizers already computed during startup.
        # Scalar lookup avoids per-query unpack tuples and repeated log/erfc calls.
        if self._log_sums is None:
            values = [0.0] * (self.cap + 1)
            survivals = [0.0] * self.cap
            values[self.cap] = -math.inf
            for x in range(self.cap - 1, -1, -1):
                log_s = (0.0 if x == 0 else
                         _log_normal_survival((math.log(x + 0.5) - self.mu) / self.sigma))
                survivals[x] = log_s
                left, right = max(log_s, values[x + 1]), min(log_s, values[x + 1])
                values[x] = left + math.log1p(math.exp(right - left)) if left != -math.inf else left
            object.__setattr__(self, '_log_survivals', tuple(survivals))
            object.__setattr__(self, '_log_sums', tuple(values))
        left, right = self._log_sums[start], self._log_sums[stop]
        if left == -math.inf:
            return 0.0
        total = math.exp(left - self._log_survivals[given]) * -math.expm1(right - left)
        return min(float(stop - start), max(0.0, total))


@dataclass(frozen=True, slots=True)
class FutureProfile:
    """An immutable live residual law. Quantiles are REMAINING, not total tokens.

    ``actual_g`` and ``actual_blocks`` must come from engine observations. Existing
    over-allocation is retained until release; an omitted allocation (zero) still
    reserves the prompt/generated context implied by the observation.
    """

    index: int
    p: int
    actual_g: int
    actual_blocks: int
    block_size: int
    cap_remaining: int
    terminal_blocks: int
    current_blocks: int
    _exact_remaining: int | None
    _mu: float
    _sigma: float
    _sums: _SurvivalSums | None = field(repr=False, compare=False)

    def _log_survival(self, h):
        if h <= 1:
            return 0.0
        if h > self.cap_remaining:
            return -math.inf
        if self._exact_remaining is not None:
            return 0.0 if h <= self._exact_remaining else -math.inf
        x = self.actual_g + h - 1
        z = (math.log(x + 0.5) - self._mu) / self._sigma
        if self.actual_g == 0:
            return _log_normal_survival(z)
        base_z = (math.log(self.actual_g + 0.5) - self._mu) / self._sigma
        if base_z >= 8.0:
            # Do not subtract two huge negative log tails. This factored square
            # difference also works when erfc(base_z / sqrt(2)) underflows.
            delta = math.log1p((h - 1) / (self.actual_g + 0.5)) / self._sigma
            if not math.isfinite(base_z) or not math.isfinite(delta):
                return -math.inf
            return min(0.0, -delta * (base_z + 0.5 * delta)
                       + _log_mills_ratio(base_z + delta) - _log_mills_ratio(base_z))
        return min(0.0, _log_normal_survival(z) - _log_normal_survival(base_z))

    def survival(self, h):
        """Return P(D - actual_g >= h | D > actual_g), for integer h."""
        if type(h) is not int:
            raise ValueError('h must be an integer service horizon')
        return math.exp(self._log_survival(h))

    def survival_sum(self, start, stop):
        """Sum survival(h) for integer h in [start, stop), in O(1) after caching.

        The first nontrivial O query builds one O(public cap) immutable
        prefix table per distinct conditional public law, not per live profile.
        C uses an exact interval length and never builds a table.
        """
        if type(start) is not int or type(stop) is not int or stop < start:
            raise ValueError('Survival-sum endpoints must be integers with start <= stop')
        stop = min(stop, self.cap_remaining + 1)
        if stop <= start:
            return 0.0
        if self._exact_remaining is not None:
            return float(stop - start)
        certain = max(0, min(stop, 2) - start)
        start = max(start, 2)
        if stop <= start:
            return float(certain)
        return certain + self._sums.interval(self.actual_g + start - 1,
                                             self.actual_g + stop - 1, self.actual_g)

    def quantile(self, q):
        """Smallest residual whose CDF reaches q; endpoints select support bounds."""
        q = _probability(q, 'q')
        if self._exact_remaining is not None:
            return self._exact_remaining
        if q == 0:
            return 1
        if q == 1:
            return self.cap_remaining
        log_tail = math.log1p(-q)
        low, high = 1, self.cap_remaining
        while low < high:
            middle = (low + high) // 2
            if self._log_survival(middle + 1) <= log_tail:
                high = middle
            else:
                low = middle + 1
        return low


def fixed_cohort_indices(requests, size=64, strata=8):
    """Choose a fixed prompt-stratified hash sample using only (p, index).

    Selection ignores decode lengths, arrivals and input order. Strata are nearly
    equal-sized prompt-rank intervals, with proportional integer sample quotas.
    The returned order is fixed too, so completion order cannot change a fit.
    """
    size = _integer(size, 'size', 1)
    strata = _integer(strata, 'strata', 1)
    ranked = sorted(((_integer(row['p'], 'p'), _integer(row['index'], 'index'))
                     for row in requests), key=lambda item: (item[0], item[1]))
    count = len(ranked)
    if len({index for _, index in ranked}) != count:
        raise ValueError('Cohort request indices must be unique')
    if not count:
        return ()
    size = min(size, count)
    strata = min(strata, size)
    boundaries = [position * count // strata for position in range(strata + 1)]
    widths = [boundaries[position + 1] - boundaries[position] for position in range(strata)]
    quotas = [size * width // count for width in widths]
    remainders = sorted(range(strata), key=lambda position: (-(size * widths[position] % count),
                                                            position))
    for position in remainders[:size - sum(quotas)]:
        quotas[position] += 1
    selected = set()
    for position, quota in enumerate(quotas):
        candidates = ranked[boundaries[position]:boundaries[position + 1]]
        candidates.sort(key=lambda item: (
            hashlib.sha256(f'kv-future-cohort-v1:{item[1]}'.encode('ascii')).digest(), item[1]))
        selected.update(index for _, index in candidates[:quota])
    return tuple(index for _, index in ranked if index in selected)


class FutureModel:
    """Whitelist-only information boundary for a full public request catalogue.

    ``requests`` is a sequence of plain {index, p, arrival_index} dictionaries (or
    a dictionary keyed by index). C additionally requires d; O rejects it and all
    unknown fields, including nested payloads. ``law`` uses build_drawn_workload's
    raw draw-spec names: sigma, median, d_max, rho and public prompt metadata.
    Neither a manifest nor its PRNG seed is retained or accepted through this API.
    """

    def __init__(self, track, requests, law, max_model_len, block_size=256):
        if type(track) is not str or track not in ('C', 'O'):
            raise ValueError('track must be C or O')
        self.track = track
        self.max_model_len = _integer(max_model_len, 'max_model_len', 1)
        self.block_size = _integer(block_size, 'block_size', 1)
        self.requests = _copy_requests(requests, track)
        self.law = _copy_law(law)
        self.indices = tuple(row['index'] for row in self.requests)
        self._by_index = MappingProxyType({row['index']: row for row in self.requests})
        median = _integer(self.law.get('median', 256), 'median', 1)
        public_cap = _integer(self.law.get('d_max', 4096 if track == 'O' else max_model_len),
                              'd_max', 1)
        sigma = self.law.get('sigma', 1.0 if track == 'C' else None)
        rho = self.law.get('rho', 0.0)
        if type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('The public draw law requires finite sigma > 0')
        if type(rho) not in (int, float) or not math.isfinite(rho) or not -1 < rho < 1:
            raise ValueError('The public draw law requires -1 < rho < 1')
        if track == 'O' and rho != 0 and self.law.get('prompt_median') is not None:
            raise ValueError('Correlated synthesized prompts hide the original catalogue rank; '
                             'the public final prompt lengths do not determine this draw law')
        if 'n' in self.law and _integer(self.law['n'], 'n', 1) != len(self.requests):
            raise ValueError('The public draw count must match the full request catalogue')
        scale = float(sigma) * math.sqrt(1.0 - float(rho) ** 2)
        if scale == 0 or not math.isfinite(scale):
            raise ValueError('Conditional public-law scale is not representable')
        means = {}
        mu = math.log(median)
        if track == 'O' and rho != 0:
            # The generator ranks BEFORE its final arrival shuffle. Its stable
            # tie-break is original request index, not arrival_index or row order.
            ranked = sorted(self.requests, key=lambda row: (row['p'], row['index']))
            normal = NormalDist()
            for rank, row in enumerate(ranked):
                means[row['index']] = mu + sigma * rho * normal.inv_cdf((rank + 0.5) / len(ranked))
        self._parameters = {}
        sums_by_law = {}
        for row in self.requests:
            cap = min(public_cap, max_model_len - row['p'])
            if cap < 1:
                raise ValueError(f'Prompt {row["index"]} leaves no decode budget')
            if track == 'C' and row['d'] > cap:
                raise ValueError(f'Request {row["index"]} exceeds its public decode cap')
            parameters = (cap, means.get(row['index'], mu), scale)
            sums = None
            if track == 'O':
                sums = sums_by_law.get(parameters)
                if sums is None:
                    sums = _SurvivalSums(*parameters)
                    sums_by_law[parameters] = sums
            self._parameters[row['index']] = (*parameters, sums)
        self.cohort_indices = fixed_cohort_indices(self.requests)
        self._cohort_set = frozenset(self.cohort_indices)
        self._cohort_completions = {}
        self._resolved_sample = () if not self.cohort_indices else None
        self._batch_state = None

    def prepare_survival_batches(self, deadline=None):
        """Cache the same legal laws for vector queries; no per-decision profile objects."""
        if self._batch_state is not None:
            return
        if self.track == 'O':
            assert all('d' not in row for row in self.requests), 'O batch provider contains targets'
        def frozen(values, dtype):
            return np.frombuffer(np.asarray(values, dtype=dtype).tobytes(), dtype=dtype)
        limits, law_ids, laws, identities = [], [], [], {}
        for row in self.requests:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Batch profile startup exceeded its deadline')
            cap, _, _, sums = self._parameters[row['index']]
            limits.append(row['d'] if self.track == 'C' else cap)
            if sums is None:
                law_ids.append(-1)
                continue
            identity = id(sums)
            if identity not in identities:
                sums.interval(0, 1, 0)
                identities[identity] = len(laws)
                laws.append((frozen(sums._log_sums, np.float64),
                             frozen(sums._log_survivals, np.float64)))
            law_ids.append(identities[identity])
        self._batch_state = (frozen(limits, np.int64), frozen(law_ids, np.int64), tuple(laws))

    def survival_sums(self, positions, actual_g, starts, stops):
        """Vector equivalent of future_profile(...).survival_sum(start, stop).

        Positions index ``self.indices``. Input buffers are never retained. The
        returned array owns its values and cannot expose an input buffer's base.
        """
        self.prepare_survival_batches()
        positions, actual_g, starts, stops = map(np.asarray, (positions, actual_g, starts, stops))
        if any(a.dtype.kind != 'i' for a in (positions, actual_g, starts, stops)):
            raise ValueError('Batch profile queries require signed integer arrays or scalars')
        if positions.ndim != 1 or actual_g.shape != positions.shape:
            raise ValueError('Positions and generated counts must be matching vectors')
        limits, law_ids, laws = self._batch_state
        if np.any(positions < 0) or np.any(positions >= len(limits)):
            raise ValueError('Unknown batch profile position')
        remaining = limits[positions] - actual_g
        if np.any(actual_g < 0) or np.any(remaining <= 0) or np.any(stops < starts):
            raise ValueError('Invalid attained service or survival interval')
        starts = np.broadcast_to(starts, positions.shape)
        stops = np.minimum(np.broadcast_to(stops, positions.shape), remaining + 1)
        if self.track == 'C':
            return np.maximum(0, stops - starts).astype(np.float64)
        result = np.maximum(0, np.minimum(stops, 2) - starts).astype(np.float64)
        starts = np.maximum(starts, 2)
        active = stops > starts
        selected_laws = law_ids[positions]
        for law_id, (prefix, normalizer) in enumerate(laws):
            selected = np.flatnonzero(active & (selected_laws == law_id))
            if not len(selected):
                continue
            g = actual_g[selected]
            left = prefix[g + starts[selected] - 1]
            right = prefix[g + stops[selected] - 1]
            finite = left != -np.inf
            selected, g, left, right = selected[finite], g[finite], left[finite], right[finite]
            values = np.exp(left - normalizer[g]) * -np.expm1(right - left)
            result[selected] += np.clip(values, 0.0, stops[selected] - starts[selected])
        return result

    def future_profile(self, index, actual_g=0, actual_blocks=0):
        """Build a live profile from observed generated tokens and allocated blocks."""
        index = _integer(index, 'index')
        actual_g = _integer(actual_g, 'actual_g')
        actual_blocks = _integer(actual_blocks, 'actual_blocks')
        row = self._by_index[index]
        cap, mu, sigma, sums = self._parameters[index]
        total = row['d'] if self.track == 'C' else cap
        if actual_g >= total:
            raise ValueError(f'Request {index} has zero survival at actual_g={actual_g}')
        current = max(actual_blocks, -(-(row['p'] + actual_g) // self.block_size))
        terminal = max(actual_blocks, -(-(row['p'] + total) // self.block_size))
        remaining = total - actual_g
        return FutureProfile(index, row['p'], actual_g, actual_blocks, self.block_size,
                             remaining, terminal, current,
                             remaining if self.track == 'C' else None, mu, sigma, sums)

    def planner_manifest(self, quantile=0.5):
        """Return [p, representative TOTAL d, arrival_index] in input request order.

        This is the initial catalogue, not a residual live-state manifest. C uses
        exact totals and O uses legal public-law quantiles; the common planner
        must not replace total d with a remaining-token quantile.
        """
        quantile = _probability(quantile, 'quantile')
        return [[row['p'], self.future_profile(row['index']).quantile(quantile), row['arrival_index']]
                for row in self.requests]

    def observe_completion(self, index, actual_d):
        """Record an engine-observed completion; return whether the fixed cohort resolved.

        Non-cohort completions do not enter calibration. Repeated identical
        observations are harmless; contradictory observed lengths are rejected.
        No observation changes the primary O-law profiles or planner manifest.
        """
        index = _integer(index, 'index')
        actual_d = _integer(actual_d, 'actual_d', 1)
        row = self._by_index[index]
        cap = self._parameters[index][0]
        if actual_d > cap or (self.track == 'C' and actual_d != row['d']):
            raise ValueError('Observed completion contradicts the request length contract')
        if index in self._cohort_set:
            previous = self._cohort_completions.get(index)
            if previous is not None and previous != actual_d:
                raise ValueError('Contradictory completion observations')
            self._cohort_completions[index] = actual_d
            if self._resolved_sample is None and len(self._cohort_completions) == len(self.cohort_indices):
                self._resolved_sample = tuple(
                    (member, self._by_index[member]['p'], self._cohort_completions[member],
                     self._by_index[member]['arrival_index']) for member in self.cohort_indices)
        return self._resolved_sample is not None

    def calibration_sample(self):
        """Return None until the WHOLE cohort resolves, then immutable (index,p,d,arrival) rows.

        This is the only sample exposed for optional secondary fitting/calibration;
        the primary public-law model deliberately does no completed-only fitting.
        """
        return self._resolved_sample


def _liabilities(pending_terminal_blocks, other_blocks):
    return (_integer(pending_terminal_blocks, 'pending_terminal_blocks')
            + _integer(other_blocks, 'other_blocks'))


def endpoint_peak(profiles, pending_terminal_blocks=0, other_blocks=0):
    """Separate, release-free gate: sum individual terminal claims plus liabilities."""
    return (_liabilities(pending_terminal_blocks, other_blocks)
            + sum(profile.terminal_blocks for profile in profiles))


def certified_peak(profiles, pending_terminal_blocks=0, other_blocks=0, alpha=0.01):
    """Certified TOTAL peak blocks, not an additional margin or a sampled estimate.

    For conditionally independent ready residents, use a union-Bernstein envelope
    over h=1 and every block-growth horizon through complete public-cap drainage.
    Take its minimum with the exact worst-cap trajectory. C point masses use this
    same calculation with zero variance and reduce to the exact event-sweep peak.
    alpha is PER DECISION; the caller must allocate a whole-run risk budget. Zero
    alpha selects the deterministic worst-cap bound. Actual allocation, pending
    terminal reservations and other live liabilities receive no premature credit.
    """
    alpha = _probability(alpha, 'alpha')
    if alpha == 1:
        raise ValueError('alpha must be less than 1')
    liability = _liabilities(pending_terminal_blocks, other_blocks)
    profiles = tuple(profiles)
    if not profiles:
        return liability
    horizons = {1}
    events = {}
    current = 0
    point_masses = True
    for profile in profiles:
        current += profile.current_blocks
        point_masses &= profile._exact_remaining is not None or profile.cap_remaining == 1
        first_growth = profile.current_blocks * profile.block_size - profile.p - profile.actual_g + 1
        for h in range(first_growth, profile.cap_remaining + 1, profile.block_size):
            horizons.add(h)
            events[h] = events.get(h, 0) + 1
        release = profile.cap_remaining + 1
        events[release] = events.get(release, 0) - profile.terminal_blocks
    # Group simultaneous growth and cleanup at their proper next-token epochs.
    # This is the same peak functional evaluated at the public cap vector; for
    # point masses that vector IS the entire distribution, with v_h = c_h = 0.
    used = worst_cap = current
    for h in sorted(events):
        used += events[h]
        worst_cap = max(worst_cap, used)
    if not alpha or point_masses:
        return liability + worst_cap
    ell = math.log(len(horizons)) - math.log(alpha)
    certified = current
    for h in sorted(horizons):
        deterministic = 0
        mean = variance = 0.0
        largest_uncertain = 0
        for profile in profiles:
            if h > profile.cap_remaining:
                continue
            blocks = max(profile.actual_blocks,
                         -(-(profile.p + profile.actual_g + h) // profile.block_size))
            survival = profile.survival(h)
            if survival == 1.0:
                deterministic += blocks
            else:
                # An underflowed positive tail still has a nonzero support
                # range; retain that range in the Bernstein deviation term.
                mean += blocks * survival
                variance += blocks * blocks * survival * (1.0 - survival)
                largest_uncertain = max(largest_uncertain, blocks)
        shift = largest_uncertain * ell / 3.0
        upper = deterministic + mean + shift + math.sqrt(2.0 * variance * ell + shift * shift)
        certified = max(certified, math.ceil(upper))
        if certified >= worst_cap:
            return liability + worst_cap
    return liability + certified
