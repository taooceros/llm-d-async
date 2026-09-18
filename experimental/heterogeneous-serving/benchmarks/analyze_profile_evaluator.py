#!/usr/bin/env python3
"""Whole-run residuals: fixed decode curve versus held-out trace regression.

Neither predictor is a hardware measurement or an executable timetable evaluator:
both consume observed trace features. Folds hold out an entire policy family,
including renamed repeats and C/O twins. No cross-manifest/source/mode pooling.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

try:
    from benchmarks import analyze_planner as analysis, kv_planner
except ModuleNotFoundError:
    import analyze_planner as analysis
    import kv_planner


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def collect(paths):
    groups, excluded, seen = defaultdict(list), [], set()
    for path in paths:
        try:
            document = json.loads(Path(path).read_text())
        except (OSError, ValueError) as error:
            excluded.append({'artifact': path, 'reason': str(error)})
            continue
        if not isinstance(document, dict):
            continue
        measurements = document.get('measurements')
        if not isinstance(measurements, dict):
            continue
        for case_index, case in enumerate(measurements.get('policy_cases') or []):
            if case.get('failed') or case.get('error'):
                excluded.append({'artifact': path, 'case': case_index,
                                 'reason': 'failed case', 'error': case.get('error')})
                continue
            spec = case.get('case') or {}
            for run in case.get('runs') or []:
                identity = f"{document.get('submission_id')}:{case_index}:{run.get('repeat', 0)}"
                reason = None
                manifest = run.get('manifest')
                modules = run.get('planner_module_hashes')
                script = document.get('script_sha256')
                policy = run.get('policy') or spec.get('policy') or {}
                iterations = run.get('iterations') or []
                if run.get('failed') or run.get('error'):
                    reason = 'failed run'
                elif run.get('integration_smoke') or spec.get('integration_smoke'):
                    reason = 'integration smoke'
                elif run.get('profile_overhead_included_in_wall'):
                    reason = 'profiling included in measured makespan'
                elif not manifest or not iterations:
                    reason = 'missing ordered manifest or trace'
                elif not script or not modules or not document.get('submission_id'):
                    reason = 'unproven source/dependency identity'
                elif identity in seen:
                    reason = 'duplicate physical run'
                else:
                    digest = hashlib.sha256(json.dumps(manifest, separators=(',', ':')).encode()).hexdigest()
                    if digest != run.get('workload_sha256'):
                        reason = 'manifest hash mismatch'
                    elif sum(row[1] for row in manifest) != run.get('output_tokens'):
                        reason = 'manifest useful-output mismatch'
                    elif sum(row.get('emitted_output_tokens', 0) for row in iterations) != run['output_tokens']:
                        reason = 'trace useful-output mismatch'
                if reason:
                    excluded.append({'artifact': path, 'run_id': identity,
                                     'policy': policy.get('name'), 'reason': reason})
                    continue
                seen.add(identity)
                active = [row for row in iterations if row['scheduled_requests'] > 0]
                if not active or any(row['scheduled_requests'] > 512 for row in active):
                    excluded.append({'artifact': path, 'run_id': identity,
                                     'reason': 'trace outside declared decode-curve batch domain'})
                    continue
                curve_s = sum(67500 + 2920 * row['scheduled_requests']
                              + 514 * kv_planner._PAD[row['scheduled_requests']]
                              for row in active) / 1e7
                multi_tokens = sum(row['multi_token_tokens'] for row in active)
                family = {key: value for key, value in policy.items()
                          if key not in ('name', 'track', 'sort_key', 'search_seed')}
                flags = {'trace_iterations': run.get('trace_iterations', True),
                         'trace_shapes': bool(spec.get('trace_shapes', False))}
                mode = 'synchronous' if (run.get('service_audit') or {}).get('synchronous') else 'native_async'
                key = canonical([run['workload_sha256'], script, modules, flags, mode])
                groups[key].append({'run_id': identity, 'artifact': path, 'policy': policy.get('name'),
                                    'family': canonical(family), 'workload_sha256': run['workload_sha256'],
                                    'script_sha256': script, 'module_hashes': modules, 'trace_flags': flags,
                                    'service_mode': mode, 'output_tokens': run['output_tokens'],
                                    'measured_wall_s': run['wall_s'], 'curve_on_trace_s': curve_s,
                                    'multi_token_tokens': multi_tokens,
                                    'saved_planner_surrogate_J': (run.get('planner') or {}).get('surrogate_J')})
    return groups, excluded


def residual_metrics(rows, predictor):
    errors = [row[predictor] - row['measured_wall_s'] for row in rows]
    if not errors:
        return None
    return {'runs': len(errors), 'mae_s': statistics.mean(abs(error) for error in errors),
            'rmse_s': (statistics.mean(error * error for error in errors)) ** .5,
            'max_abs_error_s': max(abs(error) for error in errors),
            'residual_convention': 'prediction minus measured makespan'}


def evaluate(groups):
    results = []
    for key, rows in sorted(groups.items()):
        assert len({row['output_tokens'] for row in rows}) == 1
        families = defaultdict(list)
        for row in rows:
            families[row['family']].append(row)
        folds = []
        for held_out, test in sorted(families.items()):
            train = {family: members for family, members in families.items() if family != held_out}
            train_ids = [row['run_id'] for members in train.values() for row in members]
            test_ids = [row['run_id'] for row in test]
            assert set(train_ids).isdisjoint(test_ids)
            fold = {'held_out_family': held_out, 'train_run_ids': train_ids, 'test_run_ids': test_ids}
            if len(train) < 3:
                fold['unavailable_reason'] = 'need at least three independent training families'
            else:
                # Equal family weights prevent renamed repeats or twins dominating the fit.
                x = np.array([[statistics.mean(row['curve_on_trace_s'] for row in members),
                               statistics.mean(row['multi_token_tokens'] for row in members) / 1e6]
                              for members in train.values()])
                y = np.array([statistics.mean(row['measured_wall_s'] for row in members)
                              for members in train.values()])
                scale = np.linalg.norm(x, axis=0)
                normalized = x / np.where(scale > 0, scale, 1)
                if np.linalg.matrix_rank(normalized) < 2:
                    fold['unavailable_reason'] = 'training trace features are rank deficient'
                else:
                    beta = np.linalg.lstsq(normalized, y, rcond=None)[0] / scale
                    fold['coefficients'] = {'curve_multiplier': float(beta[0]),
                                            'seconds_per_million_multi_token_tokens': float(beta[1])}
                    fold['condition_number'] = float(np.linalg.cond(normalized))
                    for row in test:
                        row['held_out_trace_prediction_s'] = float(
                            beta @ [row['curve_on_trace_s'], row['multi_token_tokens'] / 1e6])
                        row['held_out_trace_residual_s'] = row['held_out_trace_prediction_s'] - row['measured_wall_s']
            folds.append(fold)
        common = [row for row in rows if 'held_out_trace_prediction_s' in row]
        metrics = {'fixed_decode_curve': residual_metrics(common, 'curve_on_trace_s'),
                   'held_out_trace_regression': residual_metrics(common, 'held_out_trace_prediction_s')}
        winner = min(metrics, key=lambda name: metrics[name]['mae_s']) if common else None
        results.append({'qualification_key': json.loads(key), 'families': len(families),
                        'same_useful_output_tokens': rows[0]['output_tokens'], 'folds': folds,
                        'paired_held_out_metrics': metrics, 'lower_held_out_mae': winner,
                        'fixed_curve_all_runs': residual_metrics(rows, 'curve_on_trace_s'), 'runs': rows})
    return results


def profile_interpretation(run_path, summary_path):
    document = json.loads(Path(run_path).read_text())
    runs = [run for case in document['measurements']['policy_cases'] for run in case.get('runs', [])
            if run.get('profile_capture')]
    if len(runs) != 1:
        raise ValueError('Profile interpretation requires exactly one captured run')
    run = runs[0]
    summary = json.loads(Path(summary_path).read_text())['measurements']
    capture = run['profile_capture']
    start = capture['start_iteration']
    stop = start + capture['captured_iterations']
    rows = run['iterations']
    window = rows[start:stop]
    def gap(index):
        return rows[index]['scheduled_at_s'] - rows[index - 1]['completed_at_s'] if 0 < index < len(rows) else None
    attributed = summary['module_attributed_us']
    modules = summary['by_module_us']
    return {'run_artifact': run_path, 'summary_artifact': summary_path,
            'measured_profiled_wall_s': run['wall_s'], 'captured_engine_iterations': [start, stop],
            'host_window_scheduled_requests': sorted({row['scheduled_requests'] for row in window}),
            'host_window_mean_scheduled_tokens': statistics.mean(row['scheduled_tokens'] for row in window),
            'measured_start_hook_gap_s': gap(start), 'measured_stop_hook_gap_s': gap(stop),
            'trace_event_count': summary['event_count'], 'busiest_lane': summary['busiest_lane'],
            'busiest_lane_summed_event_duration_us': summary['busiest_total_us'],
            'recorded_op_groups': summary['op_groups'], 'module_attributed_event_duration_us': attributed,
            'listed_module_duration_us': sum(row['total_us'] for row in modules),
            'unlisted_module_duration_us': attributed - sum(row['total_us'] for row in modules),
            'recorded_modules': modules, 'recorded_tf_op_paths': summary['by_tfop_full_us'],
            'recorded_sources': summary['by_source_us'], 'recorded_hlo_categories': summary['by_hlo_category_us'],
            'limitations': ['Op-group percentages divide by the selected lane summed event durations, not run wall time.',
                            'Module metadata sums all duration events carrying tf_op; it is not the same denominator as the selected lane.',
                            'Summary lacks event time extent and executed-step markers: full-window device coverage is unproven; no device milliseconds per engine step.',
                            'Generic shard_map/fusion labels do not establish FFN, normalization or sampling absence or complete semantic stage shares.',
                            'Do not subtract profile hook gaps to invent an unprofiled makespan.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', action='append', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile-artifact')
    parser.add_argument('--profile-summary')
    parser.add_argument('--profile-output')
    args = parser.parse_args(argv)
    profile_args = (args.profile_artifact, args.profile_summary, args.profile_output)
    if any(profile_args) and not all(profile_args):
        parser.error('All three --profile-* paths are required together')
    groups, excluded = collect(analysis.resolve_artifacts(args.artifact))
    cohorts = evaluate(groups)
    payload = {'cohorts': cohorts, 'excluded': excluded,
               'interpretation': 'Predictions are not measurements. Trace regressions are explanatory, not executable planner evaluators. Association does not establish a dominant causal driver. Selection uses paired held-out errors only; no cross-source/manifest/mode pooling.'}
    Path(args.output).write_text(json.dumps(payload, indent=2) + '\n')
    if all(profile_args):
        Path(args.profile_output).write_text(json.dumps(profile_interpretation(
            args.profile_artifact, args.profile_summary), indent=2) + '\n')
    print(json.dumps({'qualified_cohorts': len(cohorts), 'qualified_runs': sum(len(c['runs']) for c in cohorts),
                      'excluded': len(excluded), 'output': args.output}))


if __name__ == '__main__':
    main()
