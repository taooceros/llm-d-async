#!/usr/bin/env python3
"""Exact useful-service KV area under the uninterrupted after-token allocation model.

A = sum_i sum_{g=1..d_i} ceil((p_i + g) / L) is schedule-invariant in this model.
Scheduling changes its time profile, not its value; idle residency and recomputation
are not included. This is not a measurement of physical HBM or mixed-step execution.

For C usable blocks, T >= ceil(A/C), hence mean useful-service batch <= D/ceil(A/C).
This necessary upper bound is not proved attainable and must not be reported as
achievable headroom. Engine counters require a separate allocation-phase audit
before comparison with this model.
"""
import argparse
import glob
import json
import random

DEFAULT_BLOCK_SIZE = 256

BOUND_KIND = 'necessary_upper_bound_not_proved_attainable'

BOUND_NOTE = (
    'This bound follows from an aggregate KV block-epoch area identity under an '
    'uninterrupted-service allocation model: total area A is policy-invariant, so any run needs at '
    'least ceil(A / usable_blocks) service epochs to emit its D decode tokens. No feasible '
    'schedule attaining it has been exhibited, and none is implied by the derivation. It is a '
    'necessary upper bound on mean batch only and must not be reported as achievable headroom, nor '
    'used as the denominator of a "fraction of attainable" figure.'
)


def _prefix_area(tokens, block_size):
    """Exact ``sum_{m=1..tokens} ceil(m / block_size)`` in O(1) integer arithmetic.

    Writing ``tokens = q * block_size + r`` with ``0 <= r < block_size``: the first q whole blocks
    contribute ``block_size`` terms each, of value 1, 2, ..., q respectively, and the r-token
    remainder sits in block q + 1. Hence ``block_size * q * (q + 1) / 2 + r * (q + 1)``, where the
    triangular product is always even and so divides exactly.
    """
    whole_blocks, remainder = divmod(tokens, block_size)
    return (block_size * whole_blocks * (whole_blocks + 1) // 2
            + remainder * (whole_blocks + 1))


def request_area(p, d, block_size=DEFAULT_BLOCK_SIZE):
    """Exact ``sum_{g=1..d} ceil((p + g) / block_size)`` for one request.

    Reindexing by total context ``m = p + g`` turns the sum into a difference of prefix sums over
    the contiguous range ``m = p+1 .. p+d``. As g runs 1..d the term is piecewise constant and
    steps up by one at each block boundary, so this is the closed-form specialisation of walking
    the ``d / block_size + 1`` whole blocks the range spans -- O(1) rather than O(d / block_size),
    and exact for every boundary case: p a multiple of block_size, d short of the next boundary,
    and d landing exactly on one.
    """
    if d <= 0:
        return 0
    return _prefix_area(p + d, block_size) - _prefix_area(p, block_size)


def kv_area(manifest, block_size=DEFAULT_BLOCK_SIZE):
    """Policy-invariant KV block-epoch area: sum_i sum_{g=1..d_i} ceil((p_i + g) / block_size)."""
    total = 0
    for entry in manifest:
        p, d = entry[0], entry[1]
        total += request_area(p, d, block_size)
    return total


def bounds(manifest, usable_blocks, block_size=DEFAULT_BLOCK_SIZE):
    """Returns dict with total_decode_tokens, area, min_steps, max_mean_batch."""
    if usable_blocks <= 0:
        raise ValueError('usable_blocks must be a positive measured block count')
    decode_tokens = sum(int(entry[1]) for entry in manifest)
    area = kv_area(manifest, block_size)
    min_steps = -(-area // usable_blocks)
    return {
        'n': len(manifest),
        'total_decode_tokens': decode_tokens,
        'area': area,
        'block_size': block_size,
        'usable_blocks': usable_blocks,
        'min_steps': min_steps,
        'max_mean_batch': decode_tokens / min_steps if min_steps else None,
        'bound_kind': BOUND_KIND,
        'note': BOUND_NOTE,
    }


def _naive_request_area(p, d, block_size):
    """Reference double loop; O(d). Only used by the self-test."""
    return sum(-((p + g) // -block_size) for g in range(1, d + 1))


def _selftest(trials=250, block_size=DEFAULT_BLOCK_SIZE, verbose=True):
    """Check the fast path against the naive double loop. Returns True on pass."""
    cases = []
    rng = random.Random(20260915)
    for _ in range(trials):
        cases.append((rng.randint(1, 3000), rng.randint(1, 3000), 'random'))
    for multiple in (1, 2, 5, 11):
        p = multiple * block_size
        cases.append((p, 1, 'p multiple of L, d=1'))
        cases.append((p, block_size, 'p multiple of L, d=L'))
        cases.append((p, block_size - 1, 'p multiple of L, d=L-1'))
        cases.append((p, block_size + 1, 'p multiple of L, d=L+1'))
    for p in (1, 255, 256, 257, 511, 1000, 3000):
        cases.append((p, 1, 'd=1'))
        gap = (-p) % block_size
        if gap:
            cases.append((p, gap - 1, 'd one short of a boundary'))
            cases.append((p, gap, 'd exactly on a boundary'))
            cases.append((p, gap + 1, 'd one past a boundary'))
        cases.append((p, gap + block_size, 'd exactly on a later boundary'))
    failures = []
    for p, d, label in cases:
        got = request_area(p, d, block_size)
        want = _naive_request_area(p, d, block_size)
        if got != want or not isinstance(got, int):
            failures.append((label, p, d, got, want))
    manifest = [[p, d, i] for i, (p, d, _) in enumerate(cases)]
    aggregate = kv_area(manifest, block_size)
    aggregate_want = sum(_naive_request_area(p, d, block_size) for p, d, _ in cases)
    if aggregate != aggregate_want:
        failures.append(('aggregate kv_area', None, None, aggregate, aggregate_want))
    if kv_area([], block_size) != 0:
        failures.append(('empty manifest', None, None, kv_area([], block_size), 0))
    if kv_area([[100, 7, 0, 'extra', 'trailing']], block_size) != request_area(100, 7, block_size):
        failures.append(('extra trailing fields', 100, 7, None, None))
    if verbose:
        print(f'_selftest: {len(cases)} (p, d) cases at L={block_size} '
              f'({trials} random, {len(cases) - trials} explicit edge cases)')
        for label, p, d, got, want in failures:
            print(f'  FAIL {label}: p={p} d={d} fast={got} naive={want}')
        print('_selftest: PASS' if not failures else f'_selftest: FAIL ({len(failures)})')
    return not failures


def _iter_runs(artifact):
    for case in artifact.get('measurements', {}).get('policy_cases', []) or []:
        case_name = (case.get('case', {}).get('policy', {}) or {}).get('name')
        for run in case.get('runs', []) or []:
            name = (run.get('policy', {}) or {}).get('name') or case_name or '<unnamed>'
            yield name, run


def _resolve(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(matched)
        else:
            print(f'{pattern}: no such artifact path or glob match; skipped')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--artifact', action='append', required=True,
                        help='artifact path or glob; repeatable')
    parser.add_argument('--usable-blocks', type=int, required=True,
                        help='measured usable KV blocks (never hardcoded as truth)')
    parser.add_argument('--block-size', type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument('--selftest', action='store_true', help='run the self-test and exit')
    args = parser.parse_args()
    if args.selftest:
        return 0 if _selftest() else 1

    for path in _resolve(args.artifact):
        print(f'\n== {path}')
        try:
            with open(path) as handle:
                artifact = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f'  unreadable artifact ({exc}); skipped')
            continue
        runs = list(_iter_runs(artifact))
        if not runs:
            print('  no measurements.policy_cases[*].runs[*] entries; skipped')
        for name, run in runs:
            manifest = run.get('manifest')
            if not manifest:
                print(f'  {name} repeat={run.get("repeat")}: no manifest recorded -- this '
                      'artifact predates per-run manifest recording; skipped')
                continue
            result = bounds(manifest, args.usable_blocks, args.block_size)
            actual = run.get('mean_scheduled_requests')
            actual_text = f'{actual:.2f}' if isinstance(actual, (int, float)) else 'n/a'
            print(f'  {name} repeat={run.get("repeat")}')
            print(f'    n={result["n"]}  D={result["total_decode_tokens"]:,} tok  '
                  f'A={result["area"]:,} block-epochs  (L={result["block_size"]}, '
                  f'C={result["usable_blocks"]})')
            print(f'    min_steps >= {result["min_steps"]:,}   '
                  f'max_mean_batch <= {result["max_mean_batch"]:.2f}   '
                  f'measured mean_scheduled_requests = {actual_text}')
            print(f'    bound_kind: {result["bound_kind"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
