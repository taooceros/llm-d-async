#!/usr/bin/env python3
"""Analysis CLI for offline planner and scheduling runs.

Evaluates and validates run artifacts against manifest/admission bounds,
trace-level KV occupancy, drain tail metrics, and qualified cross-arm comparisons.
"""

import argparse
import bisect
import glob
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys

# Import kv_area_bound cleanly from same directory or package
try:
    from benchmarks import kv_area_bound
except ImportError:
    try:
        import kv_area_bound
    except ImportError:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "kv_area_bound",
            os.path.join(os.path.dirname(__file__), "kv_area_bound.py")
        )
        kv_area_bound = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(kv_area_bound)


def resolve_artifacts(patterns):
    """Expand repeatable paths/globs, preserving deterministic sorted order."""
    paths = []
    seen = set()
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            for p in matched:
                norm = os.path.abspath(p)
                if norm not in seen:
                    seen.add(norm)
                    paths.append(p)
        else:
            if os.path.exists(pattern):
                norm = os.path.abspath(pattern)
                if norm not in seen:
                    seen.add(norm)
                    paths.append(pattern)
            else:
                sys.stderr.write(f"Warning: pattern or path did not match any file: {pattern}\n")
    return paths


def extract_clean_policy(policy_dict):
    """Return policy configuration omitting name for cross-arm matching."""
    if not isinstance(policy_dict, dict):
        return {}
    return {k: v for k, v in sorted(policy_dict.items()) if k != 'name'}


def extract_trace_flags(case_dict, run_dict):
    """Extract all trace-related boolean/config flags."""
    flags = {}
    if isinstance(case_dict, dict):
        for k, v in sorted(case_dict.items()):
            if 'trace' in k:
                flags[k] = v
    if isinstance(run_dict, dict):
        for k, v in sorted(run_dict.items()):
            if 'trace' in k and k not in flags:
                flags[k] = v
    return flags


def post_schedule_calendar(manifest, starts, block_size):
    """Ideal sparse calendar at pre-sampling p+age, not the planner's p+age+1."""
    if len(manifest) != len(starts):
        raise ValueError('Planner manifest and start-vector lengths differ')
    events = {}
    for (p, d, _), start in zip(manifest, starts):
        if any(type(x) is not int for x in (p, d, start, block_size)) or min(p, d, block_size) < 1 or start < 0:
            raise ValueError('Invalid advisory calendar input')
        first = (p + block_size - 1) // block_size
        last = (p + d + block_size - 2) // block_size
        events[start] = events.get(start, 0) + first
        for block in range(first, last):
            epoch = start + block * block_size - p + 1
            events[epoch] = events.get(epoch, 0) + 1
        end = start + d
        events[end] = events.get(end, 0) - last
    allocated, calendar = 0, []
    for epoch, delta in sorted(events.items()):
        allocated += delta
        calendar.append([epoch, allocated])
    return calendar


def analyze_run(run, case_meta, artifact_meta, artifact_path):
    """Validate and compute all required metrics for a single successful run."""
    case_spec = case_meta.get('case', {}) if isinstance(case_meta, dict) else {}
    policy_spec = (run.get('policy') or case_spec.get('policy') or {})
    policy_name = policy_spec.get('name') or '<unnamed>'
    clean_policy = extract_clean_policy(policy_spec)

    wall_s = run.get('wall_s')
    output_tok_s = run.get('output_tok_s')
    output_tokens = run.get('output_tokens')
    workload_sha256 = run.get('workload_sha256')
    source_script_hash = run.get('script_sha256') or artifact_meta.get('script_sha256')
    repeat = run.get('repeat', 0)

    # Planning metrics & surrogate_J
    planner = run.get('planner')
    planning_s = (
        run.get('planning_wall_s')
        or run.get('planning_s')
        or (planner.get('planning_s') if isinstance(planner, dict) else None)
        or policy_spec.get('planning_s')
    )
    surrogate_J = (
        run.get('surrogate_J')
        or (planner.get('surrogate_J') if isinstance(planner, dict) else None)
        or policy_spec.get('surrogate_J')
    )
    planner_stats = planner.get('stats') if isinstance(planner, dict) else None
    planner_module_hashes = run.get('planner_module_hashes')

    # Preemptions and batch
    preemptions = run.get('preemption_events', 0)
    mean_batch = run.get('mean_scheduled_requests')

    # Usable blocks C = total_kv_blocks - 1 (measured, not hardcoded)
    total_kv = run.get('total_kv_blocks')
    iterations = run.get('iterations') or []
    if total_kv is None and iterations and 'total_kv_blocks' in iterations[0]:
        total_kv = iterations[0]['total_kv_blocks']
    
    if total_kv is not None:
        usable_blocks = total_kv - 1
        usable_blocks_provenance = "measured total_kv_blocks - 1"
    elif case_meta.get('usable_kv_blocks') is not None:
        usable_blocks = case_meta['usable_kv_blocks']
        usable_blocks_provenance = "case usable_kv_blocks"
    else:
        usable_blocks = 566
        usable_blocks_provenance = "default fallback 566"

    # Block size: require observed if present else model assumption 256 marked
    obs_block_size = (
        run.get('kv_block_size')
        or case_meta.get('kv_block_size')
        or case_spec.get('kv_block_size')
    )
    if obs_block_size is not None:
        block_size = int(obs_block_size)
        block_size_provenance = "observed"
    else:
        block_size = 256
        block_size_provenance = "current experiment model assumption 256"

    # Manifest and admissions validation
    manifest = run.get('manifest')
    manifest_validation = {}
    area_manifest = None

    if manifest:
        sum_manifest_d = sum(int(entry[1]) for entry in manifest)
        canon_json = json.dumps(manifest, separators=(',', ':')).encode('utf-8')
        manifest_sha256 = hashlib.sha256(canon_json).hexdigest()
        
        manifest_validation = {
            'manifest_present': True,
            'manifest_length': len(manifest),
            'sum_manifest_d': sum_manifest_d,
            'output_tokens_equals_sum_manifest_d': (output_tokens == sum_manifest_d),
            'manifest_sha256': manifest_sha256,
            'manifest_sha256_matches_workload_sha256': (manifest_sha256 == workload_sha256),
            'provenance': 'manifest recorded in run; validated canonical JSON triples sha256',
        }

        req_ids = run.get('manifest_request_ids')
        if req_ids is not None:
            manifest_validation['manifest_request_ids_present'] = True
            manifest_validation['manifest_request_ids_length'] = len(req_ids)
            manifest_validation['manifest_request_ids_length_matches_manifest'] = (len(req_ids) == len(manifest))
            manifest_validation['manifest_request_ids_unique'] = (len(set(req_ids)) == len(req_ids))
        else:
            manifest_validation['manifest_request_ids_present'] = False

        area_manifest = manifest
        n = len(manifest)
    else:
        # Older artifact absent manifest: compute from complete unique admissions [p, d]
        admissions = run.get('admissions') or []
        unique_reqs = {}
        for adm in admissions:
            req_id = adm.get('request_id')
            p = adm.get('p')
            d = adm.get('d')
            if req_id is not None and p is not None and d is not None:
                unique_reqs[req_id] = (int(p), int(d))
            elif p is not None and d is not None:
                unique_reqs[len(unique_reqs)] = (int(p), int(d))

        sum_admissions_d = sum(d for p, d in unique_reqs.values())
        manifest_validation = {
            'manifest_present': False,
            'admissions_present': bool(admissions),
            'unique_admissions_count': len(unique_reqs),
            'sum_admissions_d': sum_admissions_d,
            'output_tokens_equals_sum_admissions_d': (output_tokens == sum_admissions_d),
            'provenance': 'complete unique admissions [p,d]; order invariant; no hash reconstruction',
            'no_hash_reconstruction': True,
        }
        area_manifest = [[p, d, i] for i, (p, d) in enumerate(unique_reqs.values())]
        n = len(unique_reqs) if unique_reqs else run.get('requests', 0)

    # Compute area bounds via kv_area_bound.bounds + unrounded C*D/A
    area_results = None
    if area_manifest and usable_blocks > 0:
        b_res = kv_area_bound.bounds(area_manifest, usable_blocks=usable_blocks, block_size=block_size)
        d_tokens = b_res['total_decode_tokens']
        a_area = b_res['area']
        unrounded_cd_over_a = (usable_blocks * d_tokens) / a_area if a_area > 0 else None
        area_results = {
            'bounds': b_res,
            'unrounded_cd_over_a': unrounded_cd_over_a,
            'usable_blocks': usable_blocks,
            'usable_blocks_provenance': usable_blocks_provenance,
            'block_size': block_size,
            'block_size_provenance': block_size_provenance,
        }

    # Trace metrics (never pretend missing trace is present)
    trace_present = bool(iterations)
    host_kv_step_weighted = None
    host_kv_time_weighted = None
    drain_metrics = None
    calendar_metrics = None

    if trace_present and total_kv is not None:
        raw_occ = [total_kv - row['free_kv_blocks_after_schedule'] for row in iterations]
        # Allocation baseline audit: total_kv_blocks (e.g. 567) includes 1 null block, usable capacity is total_kv - 1 (e.g. 566).
        # Capacity fraction: (total - free - 1) / (total - 1).
        occ_excluding_null = [max(0, b - 1) for b in raw_occ]
        
        # Step-weighted
        step_raw_mean = sum(raw_occ) / len(raw_occ)
        step_excl_null_mean = sum(occ_excluding_null) / len(occ_excluding_null)
        host_kv_step_weighted = {
            'mean_used_excluding_null': step_excl_null_mean,
            'capacity_fraction_of_usable': step_excl_null_mean / usable_blocks if usable_blocks else None,
            'raw_allocated_including_null': step_raw_mean,
            'raw_fraction_of_total': step_raw_mean / total_kv if total_kv else None,
            'denominator_usable_blocks': usable_blocks,
            'denominator_total_blocks': total_kv,
            'formula': '(total - free - 1) / (total - 1) for capacity fraction; raw total-free with denominator total includes null block',
            'type': 'step-weighted',
        }

        # Time-weighted: compute from row scheduled_at_s intervals bounded wall_s
        times = [row['scheduled_at_s'] for row in iterations]
        dts = []
        for i in range(len(times) - 1):
            t_curr = times[i]
            t_next = min(wall_s, times[i + 1]) if wall_s is not None else times[i + 1]
            dts.append(max(0.0, t_next - t_curr))
        if wall_s is not None and times:
            dts.append(max(0.0, wall_s - times[-1]))
        else:
            dts.append(0.0)

        total_dt = sum(dts)
        if total_dt > 0:
            time_raw_mean = sum(o * dt for o, dt in zip(raw_occ, dts)) / total_dt
            time_excl_null_mean = sum(o * dt for o, dt in zip(occ_excluding_null, dts)) / total_dt
            host_kv_time_weighted = {
                'mean_used_excluding_null': time_excl_null_mean,
                'capacity_fraction_of_usable': time_excl_null_mean / usable_blocks if usable_blocks else None,
                'raw_allocated_including_null': time_raw_mean,
                'raw_fraction_of_total': time_raw_mean / total_kv if total_kv else None,
                'denominator_usable_blocks': usable_blocks,
                'denominator_total_blocks': total_kv,
                'total_bounded_time_s': total_dt,
                'formula': '(total - free - 1) / (total - 1) for capacity fraction; raw total-free with denominator total includes null block',
                'type': 'time-weighted',
            }
        # Drain tail: LAST scheduled step >= half median POSITIVE executed batch through wall end
        batches = [row['scheduled_requests'] for row in iterations]
        positive_batches = [b for b in batches if b > 0]
        if positive_batches and wall_s is not None:
            med_batch = statistics.median(positive_batches)
            half_med = 0.5 * med_batch
            eligible = [i for i, b in enumerate(batches) if b >= half_med]
            if eligible:
                last_step_idx = max(eligible)
                drain_start_t = iterations[last_step_idx]['scheduled_at_s']
                drain_s = max(0.0, wall_s - drain_start_t)
                drain_share = drain_s / wall_s if wall_s > 0 else 0.0
                drain_metrics = {
                    'median_batch': med_batch,
                    'half_median_batch_threshold': half_med,
                    'positive_batches_count': len(positive_batches),
                    'total_batches_count': len(batches),
                    'last_qualifying_step_iteration': iterations[last_step_idx].get('iteration', last_step_idx),
                    'last_qualifying_step_scheduled_at_s': drain_start_t,
                    'drain_seconds': drain_s,
                    'drain_share': drain_share,
                    'definition': 'LAST scheduled step >= half median positive executed batch through wall end',
                }

        # Live calendar comparison if present in iterations
        live_conservative_residuals = None
        has_cal = any('calendar_signed_error_blocks' in row for row in iterations)
        if has_cal:
            signed_errs = [
                row['calendar_signed_error_blocks']
                for row in iterations
                if 'calendar_signed_error_blocks' in row
            ]
            if signed_errs:
                mean_err = sum(signed_errs) / len(signed_errs)
                max_abs_err = max(abs(e) for e in signed_errs)
                first_abs_gt_1 = next(
                    (row.get('iteration', idx)
                     for idx, row in enumerate(iterations)
                     if abs(row.get('calendar_signed_error_blocks', 0)) > 1),
                    None
                )
                safety_exceedances = sum(
                    1 for row in iterations if row.get('calendar_exceeds_safe_peak', False)
                )
                first_safety_step = next(
                    (row.get('iteration', idx)
                     for idx, row in enumerate(iterations)
                     if row.get('calendar_exceeds_safe_peak', False)),
                    None
                )
                live_conservative_residuals = {
                    'mean_signed_error_blocks': mean_err,
                    'max_abs_error_blocks': max_abs_err,
                    'first_abs_gt_1_step': first_abs_gt_1,
                    'safety_exceedances_count': safety_exceedances,
                    'first_safety_exceedance_step': first_safety_step,
                    'calendar_kind': 'LIVE rebased conservative calendar, not advisory offline timetable replay',
                    'convention': 'native post-schedule p+g convention; pending prefills endpoint upper bounds, not exact expected occupancy',
                    'comparison_doc': run.get('calendar_comparison'),
                }

        # Advisory sparse-calendar versus hardware replay alongside live residuals
        advisory_replay = None
        planner_data = run.get('planner')
        cal_events = None
        advisory_unavailable_reason = None
        if (run.get('execution_state') or {}).get('planner_order_only'):
            advisory_unavailable_reason = (
                'Order-only execution has no service-epoch alignment or release certificate; '
                'the recorded timetable is a planning diagnostic only')
        elif isinstance(planner_data, dict):
            # O's hidden actual targets are not its forecast: require the saved representative.
            planner_manifest = run.get('planner_manifest')
            if planner_manifest is None and policy_spec.get('track', run.get('track')) == 'C':
                planner_manifest = manifest
            if planner_manifest is not None and planner_data.get('starts') is not None:
                cal_events = post_schedule_calendar(planner_manifest, planner_data['starts'], block_size)
            else:
                advisory_unavailable_reason = 'Ordered planner input manifest was not recorded'

        if cal_events and trace_present and total_kv is not None:
            cal_epochs = [entry[0] for entry in cal_events]
            cal_blocks = [entry[1] for entry in cal_events]
            advisory_safe_peak = max(cal_blocks)

            adv_signed_errors = []
            actual_allocations = []
            advisory_allocations = []
            compared_steps = []

            for idx, row in enumerate(iterations):
                step = row.get('iteration', idx)
                free_b = row.get('free_kv_blocks_after_schedule')
                if free_b is None:
                    continue
                # Null-block baseline: pool total_kv has 1 null block; actual allocated is total_kv - free - 1
                actual_b = total_kv - free_b - 1

                # Query advisory calendar at epoch = step (explicit 1:1 mapping under ideal instantaneous-prefill assumption)
                if step < 0:
                    adv_b = 0
                else:
                    k = bisect.bisect_right(cal_epochs, step) - 1
                    adv_b = cal_blocks[k] if k >= 0 else 0

                err = actual_b - adv_b
                adv_signed_errors.append(err)
                actual_allocations.append(actual_b)
                advisory_allocations.append(adv_b)
                compared_steps.append(step)

            if adv_signed_errors:
                adv_max_abs_err = max(abs(e) for e in adv_signed_errors)
                adv_max_pos_err = max(adv_signed_errors)
                adv_min_err = min(adv_signed_errors)
                adv_mean_err = sum(adv_signed_errors) / len(adv_signed_errors)
                adv_first_gt_1 = next(
                    (compared_steps[idx]
                     for idx, e in enumerate(adv_signed_errors)
                     if abs(e) > 1),
                    None
                )
                actual_peak_b = max(actual_allocations) if actual_allocations else 0
                actual_above_adv_peak = (
                    actual_peak_b > advisory_safe_peak
                    if advisory_safe_peak is not None else False
                )
                above_peak_steps = [
                    compared_steps[idx]
                    for idx, act in enumerate(actual_allocations)
                    if advisory_safe_peak is not None and act > advisory_safe_peak
                ]
                live_violations_count = sum(
                    1 for row in iterations if row.get('calendar_exceeds_safe_peak', False)
                )

                advisory_replay = {
                    'compact_signed_errors': adv_signed_errors,
                    'compared_steps': compared_steps,
                    'max_abs_error_blocks': adv_max_abs_err,
                    'max_positive_error_blocks': adv_max_pos_err,
                    'min_signed_error_blocks': adv_min_err,
                    'mean_signed_error_blocks': adv_mean_err,
                    'first_abs_gt_1_step': adv_first_gt_1,
                    'advisory_safe_peak_blocks': advisory_safe_peak,
                    'actual_peak_blocks': actual_peak_b,
                    'actual_exceeded_advisory_safe_peak': actual_above_adv_peak,
                    'actual_above_advisory_safe_peak_count': len(above_peak_steps),
                    'actual_above_advisory_safe_peak_first_step': above_peak_steps[0] if above_peak_steps else None,
                    'live_bound_violations_count': live_violations_count,
                    'epoch_to_step_mapping': 'explicit 1:1 mapping (epoch t = engine step s) under ideal instantaneous-prefill assumption',
                    'interpretation_limits': (
                        'Advisory drift reflects initial chunked-prefill ramp-up delay (where actual occupancy lags instantaneous prefill) '
                        'and terminal tail duration difference; it is strictly distinct from live certificate violation.'
                    ),
                    'allocation_convention': (
                        'native post-schedule/pre-sampling ceil((p+age)/block_size), age=0..d-1; '
                        'null block excluded from pool baseline'
                    ),
                }

        if live_conservative_residuals or advisory_replay:
            calendar_metrics = {
                'live_conservative_residuals': live_conservative_residuals,
                'advisory_calendar_replay': advisory_replay,
                'advisory_unavailable_reason': advisory_unavailable_reason,
                'mean_signed_error_blocks': (
                    live_conservative_residuals['mean_signed_error_blocks']
                    if live_conservative_residuals else (advisory_replay['mean_signed_error_blocks'] if advisory_replay else None)
                ),
                'max_abs_error_blocks': (
                    live_conservative_residuals['max_abs_error_blocks']
                    if live_conservative_residuals else (advisory_replay['max_abs_error_blocks'] if advisory_replay else None)
                ),
                'first_abs_gt_1_step': (
                    live_conservative_residuals['first_abs_gt_1_step']
                    if live_conservative_residuals else (advisory_replay['first_abs_gt_1_step'] if advisory_replay else None)
                ),
                'safety_exceedances_count': (
                    live_conservative_residuals['safety_exceedances_count']
                    if live_conservative_residuals else None
                ),
                'first_safety_exceedance_step': (
                    live_conservative_residuals['first_safety_exceedance_step']
                    if live_conservative_residuals else None
                ),
                'calendar_kind': (
                    'LIVE rebased conservative calendar alongside advisory offline sparse-calendar replay'
                    if (live_conservative_residuals and advisory_replay)
                    else (live_conservative_residuals['calendar_kind'] if live_conservative_residuals else 'Advisory offline sparse-calendar replay')
                ),
                'convention': 'native post-schedule p+g convention; pending prefills endpoint upper bounds; null block excluded',
                'comparison_doc': run.get('calendar_comparison'),
            }
        elif advisory_unavailable_reason:
            calendar_metrics = {
                'live_conservative_residuals': None,
                'advisory_calendar_replay': None,
                'advisory_unavailable_reason': advisory_unavailable_reason,
                'safety_exceedances_count': None,
            }

    # Worker shapes: padding mean / wasted lanes only if worker rank0 authoritative rows present
    worker_shapes = run.get('worker_shapes') or []
    padding_metrics = None
    rank0_entry = next((w for w in worker_shapes if isinstance(w, dict) and w.get('rank') == 0), None)
    if rank0_entry and rank0_entry.get('rows'):
        rows = rank0_entry['rows']
        pads = [r['padded_requests'] for r in rows if 'padded_requests' in r]
        wasted = [
            r['padded_requests'] - r['scheduled_requests']
            for r in rows
            if 'padded_requests' in r and 'scheduled_requests' in r
        ]
        if pads and wasted:
            padding_metrics = {
                'authoritative_source': 'worker_shapes rank0',
                'mean_padded_requests': sum(pads) / len(pads),
                'mean_wasted_lanes': sum(wasted) / len(wasted),
                'num_rows': len(rows),
                'derived_walltime_coefficient_products': False,
                'note': 'lane counts reported; no derived walltime coefficient products',
            }

    trace_flags = extract_trace_flags(case_spec, run)

    # F3 eviction, guard, and decision coverage metrics when present
    f3_metrics = None
    eviction = run.get('eviction')
    if eviction and isinstance(eviction, dict):
        scorer = eviction.get('scorer') or {}
        events = eviction.get('events') or []
        guard_events = eviction.get('guard_events') or []
        audit_events = [e for e in events if isinstance(e, dict) and e.get('event') == 'restricted_survivor_service_audit']
        survivor_audit_holds = all(e.get('holds', False) for e in audit_events) if audit_events else None

        f3_metrics = {
            'mode': eviction.get('mode'),
            'schedule_calls': eviction.get('schedule_calls'),
            'eligible_steps': eviction.get('eligible_steps'),
            'selector_preemptions': eviction.get('selector_preemptions'),
            'eligible_preemptions': eviction.get('eligible_preemptions'),
            'traversal_isolation_claim': eviction.get('traversal_isolation_claim', False),
            'unique_evicted_requests': eviction.get('unique_evicted_requests'),
            'events_count': len(events),
            'guard_events_count': len(guard_events),
            'scorer_summary': {
                'decisions': scorer.get('decisions'),
                'fallbacks': scorer.get('fallbacks'),
                'budget_violations': scorer.get('budget_violations'),
                'online_s': scorer.get('online_s'),
                'startup_s': scorer.get('startup_s'),
            },
            'restricted_survivor_service_audit': {
                'audit_events_count': len(audit_events),
                'holds': survivor_audit_holds,
                'events': audit_events,
            } if audit_events else None,
            'seam_audit': eviction.get('seam'),
        }

    # Check ranking exclusion flags (integration smoke or profile overhead included in wall)
    integration_smoke = bool(run.get('integration_smoke') or case_spec.get('integration_smoke'))
    profile_overhead = bool(run.get('profile_overhead_included_in_wall') or case_spec.get('profile_overhead_included_in_wall'))
    excluded_from_ranking = integration_smoke or profile_overhead
    exclusion_reasons = []
    if integration_smoke:
        exclusion_reasons.append("integration_smoke=True")
    if profile_overhead:
        exclusion_reasons.append("profile_overhead_included_in_wall=True")
    exclusion_reason_str = "; ".join(exclusion_reasons) if exclusion_reasons else None

    return {
        'artifact_path': artifact_path,
        'policy_name': policy_name,
        'clean_policy': clean_policy,
        'repeat': repeat,
        'excluded_from_ranking': excluded_from_ranking,
        'exclusion_reason': exclusion_reason_str,
        'wall_s': wall_s,
        'output_tok_s': output_tok_s,
        'output_tokens': output_tokens,
        'n': n,
        'workload_sha256': workload_sha256,
        'source_script_hash': source_script_hash,
        'planning_s': planning_s,
        'surrogate_J': surrogate_J,
        'planner_stats': planner_stats,
        'planner_module_hashes': planner_module_hashes,
        'mean_batch': mean_batch,
        'preemptions': preemptions,
        'trace_present': trace_present,
        'trace_flags': trace_flags,
        'host_kv_step_weighted': host_kv_step_weighted,
        'host_kv_time_weighted': host_kv_time_weighted,
        'drain_metrics': drain_metrics,
        'calendar_metrics': calendar_metrics,
        'f3_metrics': f3_metrics,
        'padding_metrics': padding_metrics,
        'manifest_validation': manifest_validation,
        'area_results': area_results,
    }


def are_module_hashes_compatible(obs1, obs2):
    """Check if module hashes are provably equal.
    
    Per Phase 3 contract: treat missing dependency hash as unproven, not equal.
    If an arm uses planner/eviction modules, missing hashes cannot be asserted equal.
    If both arms do not use planner modules (pure baseline without planner dependencies),
    module hashes are not required dependencies.
    """
    pmh1 = obs1.get('planner_module_hashes')
    pmh2 = obs2.get('planner_module_hashes')
    
    uses_modules = (
        obs1.get('clean_policy', {}).get('admit') in ('planner', 'f3')
        or obs2.get('clean_policy', {}).get('admit') in ('planner', 'f3')
        or bool(obs1.get('planner_stats')) or bool(obs2.get('planner_stats'))
        or bool(obs1.get('f3_metrics')) or bool(obs2.get('f3_metrics'))
        or pmh1 is not None or pmh2 is not None
    )
    if uses_modules:
        if pmh1 is None or pmh2 is None:
            return False, "missing dependency hash treated as unproven, not equal"
        if pmh1 != pmh2:
            return False, f"different planner module hashes ({pmh1} vs {pmh2})"
        return True, None
    else:
        return True, None


def find_trace_overhead_pairs(observations):
    """Find trace overhead pairs matching strict criteria and document disqualified pairs."""
    qualified_pairs = []
    disqualified_candidates = []
    n_obs = len(observations)

    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            obs1 = observations[i]
            obs2 = observations[j]

            # Candidate pairing: identical policy config ignoring name
            if obs1['clean_policy'] != obs2['clean_policy']:
                continue
            if obs1['workload_sha256'] != obs2['workload_sha256']:
                continue
            if obs1['output_tokens'] != obs2['output_tokens']:
                continue

            flags1 = obs1['trace_flags']
            flags2 = obs2['trace_flags']
            flags_differ = (flags1 != flags2)

            h1 = obs1['source_script_hash']
            h2 = obs2['source_script_hash']
            same_script_hash = (h1 is not None and h1 == h2)

            same_planner_hashes, hash_incompat_reason = are_module_hashes_compatible(obs1, obs2)

            pair_id = [
                f"{obs1['policy_name']} (repeat {obs1['repeat']})",
                f"{obs2['policy_name']} (repeat {obs2['repeat']})"
            ]

            if flags_differ:
                if same_script_hash and same_planner_hashes:
                    wall1 = obs1['wall_s']
                    wall2 = obs2['wall_s']
                    makespan_diff_s = wall2 - wall1
                    pct_diff = ((wall2 - wall1) / wall1) * 100.0 if wall1 else None
                    qualified_pairs.append({
                        'pair': pair_id,
                        'source_script_hash': h1,
                        'planner_module_hashes': obs1.get('planner_module_hashes'),
                        'workload_sha256': obs1['workload_sha256'],
                        'output_tokens': obs1['output_tokens'],
                        'flags1': flags1,
                        'flags2': flags2,
                        'wall_s_first': wall1,
                        'wall_s_second': wall2,
                        'makespan_diff_s': makespan_diff_s,
                        'makespan_percent': pct_diff,
                        'note': 'no fixed floor; raw measured makespan percent reported',
                    })
                else:
                    reasons = []
                    if not same_script_hash:
                        reasons.append(f"different source script hashes ({h1} vs {h2})")
                    if not same_planner_hashes:
                        reasons.append(hash_incompat_reason or "planner module hashes unproven or differ")
                    disqualified_candidates.append({
                        'pair': pair_id,
                        'reason': "; ".join(reasons) + "; different source hashes must NOT claim overhead pair",
                        'flags1': flags1,
                        'flags2': flags2,
                    })

    return qualified_pairs, disqualified_candidates


def compute_repeat_noise_and_pair_gaps(group_observations):
    """Compute same-arm repeat noise floors and repeat-paired arm gaps."""
    # Group by (clean_policy_repr, trace_flags_repr, source_script_hash, planner_module_hashes_repr, output_tokens)
    arm_runs = {}
    for obs in group_observations:
        pol_key = json.dumps(obs['clean_policy'], sort_keys=True)
        flag_key = json.dumps(obs['trace_flags'], sort_keys=True)
        pmh = obs.get('planner_module_hashes')
        uses_mod = (
            obs.get('clean_policy', {}).get('admit') in ('planner', 'f3')
            or bool(obs.get('planner_stats'))
            or bool(obs.get('f3_metrics'))
            or pmh is not None
        )
        if uses_mod and pmh is None:
            mod_key = f"unproven_missing_dependency_hash_{id(obs)}"
        else:
            mod_key = json.dumps(pmh or {}, sort_keys=True)
        tok_key = obs.get('output_tokens')
        key = (pol_key, flag_key, obs['source_script_hash'], mod_key, tok_key)
        arm_runs.setdefault(key, []).append(obs)

    same_arm_noise_floors = []
    for (pol_key, flag_key, script_hash, mod_key, tok_key), runs in arm_runs.items():
        if len(runs) > 1:
            walls = [r['wall_s'] for r in runs if r['wall_s'] is not None]
            if len(walls) > 1:
                spread_s = max(walls) - min(walls)
                min_w = min(walls)
                spread_pct = (spread_s / min_w) * 100.0 if min_w > 0 else 0.0
                same_arm_noise_floors.append({
                    'policy_config': runs[0]['clean_policy'],
                    'trace_flags': runs[0]['trace_flags'],
                    'policy_name': runs[0]['policy_name'],
                    'source_script_hash': script_hash,
                    'planner_module_hashes': runs[0].get('planner_module_hashes'),
                    'output_tokens': tok_key,
                    'repeats_count': len(runs),
                    'wall_s_values': walls,
                    'max_min_spread_s': spread_s,
                    'spread_percent': spread_pct,
                    'label': 'same_arm_noise_floor_explicit_repeats',
                    'note': 'explicit repeated SAME ARM CONFIG at fixed manifest with same script hash, module hashes, and equal emitted tokens',
                })

    # Compute pair gap values per repeats between different arms (equal emitted tokens required)
    by_policy_name = {}
    for obs in group_observations:
        by_policy_name.setdefault(obs['policy_name'], {})[obs['repeat']] = obs

    repeat_pair_gaps = []
    names = sorted(by_policy_name.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            repeats_a = by_policy_name[name_a]
            repeats_b = by_policy_name[name_b]
            common_repeats = sorted(set(repeats_a.keys()) & set(repeats_b.keys()))
            if common_repeats:
                gaps = []
                for rep in common_repeats:
                    obs_a = repeats_a[rep]
                    obs_b = repeats_b[rep]
                    # Equal-token and module-hash compatibility qualification
                    if obs_a.get('output_tokens') != obs_b.get('output_tokens') or obs_a.get('output_tokens') is None:
                        continue
                    compat, _ = are_module_hashes_compatible(obs_a, obs_b)
                    if not compat:
                        continue
                    wa = obs_a['wall_s']
                    wb = obs_b['wall_s']
                    if wa is not None and wb is not None:
                        diff_s = wa - wb
                        pct = (diff_s / wb) * 100.0 if wb > 0 else 0.0
                        gaps.append({
                            'repeat': rep,
                            'output_tokens': obs_a['output_tokens'],
                            'source_script_hash_a': obs_a.get('source_script_hash'),
                            'source_script_hash_b': obs_b.get('source_script_hash'),
                            'planner_module_hashes_a': obs_a.get('planner_module_hashes'),
                            'planner_module_hashes_b': obs_b.get('planner_module_hashes'),
                            'wall_s_a': wa,
                            'wall_s_b': wb,
                            'gap_s': diff_s,
                            'gap_percent_relative_to_b': pct,
                        })
                if gaps:
                    repeat_pair_gaps.append({
                        'arm_a': name_a,
                        'arm_b': name_b,
                        'repeats': gaps,
                        'note': 'raw pair gap values per repeat for equal-emitted-token repeated arms',
                    })

    return same_arm_noise_floors, repeat_pair_gaps

def compare_arms_in_group(group_observations):
    """Generate qualified comparisons across arms emitting equal token counts.
    
    Excludes runs with integration_smoke=True or profile_overhead_included_in_wall=True
    from headline/ranking comparisons.
    """
    ranking_candidates = [obs for obs in group_observations if not obs.get('excluded_from_ranking')]
    if not ranking_candidates:
        return []

    # Find baseline if available (e.g. 'default' or arm with highest makespan)
    baseline_obs = next(
        (obs for obs in ranking_candidates if 'default' in obs['policy_name'].lower()),
        None
    )
    if baseline_obs is None:
        # Fallback to max makespan arm as reference
        baseline_obs = max(
            ranking_candidates,
            key=lambda obs: obs['wall_s'] if obs['wall_s'] is not None else -1
        )

    base_wall = baseline_obs['wall_s']
    base_tokens = baseline_obs.get('output_tokens')
    comparisons = []

    for obs in sorted(ranking_candidates, key=lambda o: o['wall_s'] if o['wall_s'] is not None else 1e9):
        # Enforce equal-token qualification across comparisons
        if obs.get('output_tokens') != base_tokens or base_tokens is None:
            continue
        wall = obs['wall_s']
        rel_diff_pct = ((wall - base_wall) / base_wall) * 100.0 if (wall is not None and base_wall) else None

        step_occ = obs['host_kv_step_weighted']['capacity_fraction_of_usable'] if obs['host_kv_step_weighted'] else None
        time_occ = obs['host_kv_time_weighted']['capacity_fraction_of_usable'] if obs['host_kv_time_weighted'] else None
        drain_s = obs['drain_metrics']['drain_seconds'] if obs['drain_metrics'] else None
        drain_sh = obs['drain_metrics']['drain_share'] if obs['drain_metrics'] else None

        comparisons.append({
            'arm': obs['policy_name'],
            'repeat': obs['repeat'],
            'output_tokens': obs['output_tokens'],
            'source_script_hash': obs.get('source_script_hash'),
            'planner_module_hashes': obs.get('planner_module_hashes'),
            'wall_s': wall,
            'vs_reference_arm': baseline_obs['policy_name'],
            'vs_reference_wall_percent': rel_diff_pct,
            'output_tok_s': obs['output_tok_s'],
            'mean_batch': obs['mean_batch'],
            'kv_capacity_occupancy_step_weighted_pct': step_occ * 100.0 if step_occ is not None else None,
            'kv_capacity_occupancy_time_weighted_pct': time_occ * 100.0 if time_occ is not None else None,
            'preemptions': obs['preemptions'],
            'drain_seconds': drain_s,
            'drain_share': drain_sh,
            'surrogate_J': obs['surrogate_J'],
            'planning_s': obs['planning_s'],
            'note': 'all proxies reported next to measured throughput/makespan; capacity fractions exclude null block; equal-token qualified',
        })

    return comparisons


def run_phase_boundary_check(quiet=False):
    """Arithmetic fixture, not a hardware trace or performance evaluation."""
    calendar = post_schedule_calendar([[255, 4, 0], [256, 3, 1]], [0, 0], 256)
    epochs = [event[0] for event in calendar]
    # Prefill's first sample has g=0: both prompts initially consume one block.
    # At age 1 only p=256 grows; at age 2 p=255 grows. Releases occur at d.
    expected = [2, 3, 4, 2, 0]
    observed = [calendar[bisect.bisect_right(epochs, age) - 1][1] for age in range(5)]
    assert observed == expected, (observed, expected)
    total = 11
    free = [total - 1 - blocks for blocks in expected]
    assert [total - available - 1 for available in free] == observed
    assert [total - available - blocks for available, blocks in zip(free, observed)] == [1] * 5
    if not quiet:
        print('Pre-sampling phase-boundary arithmetic: PASS', observed)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analysis CLI for offline planner and scheduling runs."
    )
    parser.add_argument(
        '--artifact',
        action='append',
        required=False,
        help="Artifact path or glob pattern (repeatable)."
    )
    parser.add_argument(
        '--output',
        required=False,
        help="Path to write the resulting analysis JSON."
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress human-readable summary on stdout."
    )
    parser.add_argument(
        '--test-phase-boundary',
        action='store_true',
        help="Run deterministic phase-boundary allocation correctness check."
    )
    args = parser.parse_args(argv)

    if args.test_phase_boundary:
        ok = run_phase_boundary_check(quiet=args.quiet)
        if not args.artifact:
            return 0 if ok else 1

    if not args.artifact:
        sys.stderr.write("Error: --artifact is required when not only running --test-phase-boundary.\n")
        return 1
    if not args.output:
        sys.stderr.write("Error: --output is required when analyzing artifacts.\n")
        return 1

    artifact_paths = resolve_artifacts(args.artifact)
    if not artifact_paths:
        sys.stderr.write("No valid artifact files found.\n")
        return 1

    all_observations = []
    failed_cases = []

    for path in artifact_paths:
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                document = json.load(handle)
        except Exception as exc:
            failed_cases.append({
                'artifact': path,
                'policy_name': None,
                'error': f"Unreadable artifact ({type(exc).__name__}: {exc})",
                'failed': True,
            })
            continue

        if not isinstance(document, dict):
            failed_cases.append({
                'artifact': path,
                'policy_name': None,
                'error': f"Root JSON is {type(document).__name__}, expected object",
                'failed': True,
            })
            continue

        artifact_meta = {
            'submission_id': document.get('submission_id'),
            'status': document.get('status'),
            'script_sha256': document.get('script_sha256'),
        }

        measurements = document.get('measurements')
        if not isinstance(measurements, dict):
            # Check if artifact itself represents a failed execution or non-measurement
            if document.get('status') == 'failed':
                failed_cases.append({
                    'artifact': path,
                    'policy_name': None,
                    'error': document.get('error') or "Artifact marked failed at top level",
                    'failed': True,
                })
            continue

        # Extract explicitly recorded failed cases in measurements
        for fc in measurements.get('failed_cases') or []:
            failed_cases.append({
                'artifact': path,
                'policy_name': fc.get('policy') or fc.get('policy_name'),
                'error': fc.get('error'),
                'failed': True,
            })

        for case_idx, case in enumerate(measurements.get('policy_cases') or []):
            case_failed = case.get('failed') is True
            case_error = case.get('error')
            policy_spec = (case.get('case', {}).get('policy') or case.get('policy') or {})
            case_pol_name = case.get('policy_name') or policy_spec.get('name')

            if case_failed or case_error:
                failed_cases.append({
                    'artifact': path,
                    'policy_name': case_pol_name or f"case_{case_idx}",
                    'error': case_error or "Case explicitly marked failed",
                    'failed': True,
                    'case_spec': case.get('case'),
                })
                continue

            runs = case.get('runs') or []
            if not runs:
                failed_cases.append({
                    'artifact': path,
                    'policy_name': case_pol_name or f"case_{case_idx}",
                    'error': "No runs present in policy case",
                    'failed': True,
                    'case_spec': case.get('case'),
                })
                continue

            for run in runs:
                if run.get('failed') or run.get('error') or run.get('status') == 'failed':
                    failed_cases.append({
                        'artifact': path,
                        'policy_name': (run.get('policy') or {}).get('name') or case_pol_name,
                        'error': run.get('error') or "Run explicitly marked failed",
                        'failed': True,
                    })
                    continue

                try:
                    obs = analyze_run(run, case, artifact_meta, path)
                    all_observations.append(obs)
                except Exception as exc:
                    failed_cases.append({
                        'artifact': path,
                        'policy_name': (run.get('policy') or {}).get('name') or case_pol_name,
                        'error': f"Exception during run analysis ({type(exc).__name__}: {exc})",
                        'failed': True,
                    })

    # Group by workload_sha256 and assert same emitted counts before arm comparisons
    workload_groups = {}
    for obs in all_observations:
        w_hash = obs['workload_sha256']
        workload_groups.setdefault(w_hash, []).append(obs)

    analyzed_groups = {}
    arm_comparisons_by_group = {}
    repeat_noise_and_gaps_by_group = {}

    for w_hash, group_obs in workload_groups.items():
        emitted_counts = {obs['output_tokens'] for obs in group_obs if obs['output_tokens'] is not None}
        same_emitted = (len(emitted_counts) == 1)

        same_arm_floors, pair_gaps = compute_repeat_noise_and_pair_gaps(group_obs)
        repeat_noise_and_gaps_by_group[w_hash] = {
            'same_arm_noise_floors': same_arm_floors,
            'repeat_pair_gaps': pair_gaps,
        }

        if same_emitted:
            comparisons = compare_arms_in_group(group_obs)
            arm_comparisons_by_group[w_hash] = comparisons
            analyzed_groups[w_hash] = {
                'workload_sha256': w_hash,
                'runs_count': len(group_obs),
                'assert_same_emitted_tokens': True,
                'emitted_output_tokens': next(iter(emitted_counts)) if emitted_counts else None,
                'status': 'valid_for_arm_comparisons',
            }
        else:
            arm_comparisons_by_group[w_hash] = []
            analyzed_groups[w_hash] = {
                'workload_sha256': w_hash,
                'runs_count': len(group_obs),
                'assert_same_emitted_tokens': False,
                'emitted_output_tokens_set': sorted(list(emitted_counts)),
                'status': 'disqualified_for_arm_comparisons',
                'disqualification_reason': (
                    f"Emitted token counts differ across arms ({sorted(list(emitted_counts))}). "
                    f"Per SCHEDULING_HEADROOM_GOAL.md, an arm that emits fewer tokens has not gone faster."
                ),
            }

    # Trace overhead pairing
    qualified_trace_pairs, disqualified_trace_candidates = find_trace_overhead_pairs(all_observations)

    final_payload = {
        'summary': {
            'artifacts_evaluated': len(artifact_paths),
            'successful_observations': len(all_observations),
            'failed_cases_count': len(failed_cases),
            'workload_groups_count': len(workload_groups),
            'qualified_trace_overhead_pairs_count': len(qualified_trace_pairs),
        },
        'failed_cases': failed_cases,
        'workload_groups': analyzed_groups,
        'arm_comparisons': arm_comparisons_by_group,
        'repeat_noise_and_pair_gaps': repeat_noise_and_gaps_by_group,
        'trace_overhead_pairs': {
            'qualified': qualified_trace_pairs,
            'disqualified_candidates': disqualified_trace_candidates,
        },
        'raw_observations': all_observations,
    }

    # Ensure output directory exists and write JSON
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(final_payload, handle, indent=2)

    if not args.quiet:
        print(f"\n=== Planner Analysis Completed ===")
        print(f"Artifacts processed: {len(artifact_paths)}")
        print(f"Successful runs analyzed: {len(all_observations)}")
        print(f"Failed cases preserved: {len(failed_cases)}")
        if failed_cases:
            print("\nExplicit Failed Cases:")
            for fc in failed_cases:
                print(f"  - [{fc.get('policy_name')}] {fc.get('artifact')}: {fc.get('error')}")

        print(f"\nWorkload Groups: {len(workload_groups)}")
        for w_hash, group_info in analyzed_groups.items():
            print(f"  Group {w_hash[:12]}... (n_runs={group_info['runs_count']}): "
                  f"status={group_info['status']}, tokens={group_info.get('emitted_output_tokens')}")

        if qualified_trace_pairs:
            print(f"\nQualified Trace Overhead Pairs ({len(qualified_trace_pairs)}):")
            for qp in qualified_trace_pairs:
                print(f"  - {qp['pair'][0]} vs {qp['pair'][1]}: "
                      f"makespan_diff = {qp['makespan_diff_s']:+.2f}s ({qp['makespan_percent']:+.2f}%) "
                      f"[flags: {qp['flags1']} vs {qp['flags2']}]")

        if disqualified_trace_candidates:
            print(f"\nDisqualified Trace Overhead Candidates ({len(disqualified_trace_candidates)}):")
            for dqc in disqualified_trace_candidates:
                print(f"  - {dqc['pair'][0]} vs {dqc['pair'][1]}: {dqc['reason']}")

        print(f"\nWrote full analysis JSON to: {args.output}\n")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
