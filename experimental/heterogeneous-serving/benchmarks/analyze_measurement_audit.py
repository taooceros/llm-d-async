#!/usr/bin/env python3
"""Reconcile saved physical serving measurements; never estimate execution."""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def batch_distribution(rows):
    counts = {}
    for row in rows:
        size = row['executed_requests']
        counts[size] = counts.get(size, 0) + 1
    if not counts:
        return None
    total = sum(counts.values())
    return {'iterations': total, 'min': min(counts), 'max': max(counts),
            'mean': sum(size * count for size, count in counts.items()) / total,
            'histogram': dict(sorted(counts.items()))}


def summarize_run(run):
    n = run['finished_requests']
    iterations = run['iterations']
    traced = run['trace_enabled']
    if traced and sum(r.get('emitted_output_tokens', 0) for r in iterations) != run['output_tokens']:
        raise ValueError('Trace and completed output counts differ')
    lives = run.get('request_lifecycles')
    if traced and lives is not None:
        if len(lives) != n or sum(r['output_tokens'] for r in lives) != run['output_tokens']:
            raise ValueError('Request lifecycles do not reconcile')
        for life in lives:
            expected = ((life['last_output_s'] - life['first_output_s']) / (life['output_tokens'] - 1)
                        if life['output_tokens'] > 1 else None)
            if life['observed_tpot_s'] != expected:
                raise ValueError('Incorrect request TPOT denominator')
    workers = run.get('worker_shapes', [])
    if workers:
        canonical = workers[0]['rows']
        if any(worker['rows'] != canonical for worker in workers[1:]):
            raise ValueError('TP workers disagree on prepared global inputs')
    preemptions = (sum(len(r['preempted_request_ids']) for r in iterations)
                   if traced and all(r['preempted_request_ids'] is not None for r in iterations) else None)
    tpot = [r['observed_tpot_s'] for r in lives or [] if r['observed_tpot_s'] is not None]
    shape_summary = {}
    for label, decode in [('decode_only', True), ('prefill_or_mixed', False)]:
        selected = [r for r in iterations if r.get('decode_only') is decode]
        if selected:
            scheduled = sum(r['scheduled_tokens'] for r in selected)
            padded = sum(r['padded_input_tokens'] for r in selected)
            shape_summary[label] = {'iterations': len(selected), 'scheduled_tokens': scheduled,
                                    'prepared_tokens': padded, 'token_slot_density': scheduled / padded,
                                    'buckets': sorted({r['padded_input_tokens'] for r in selected})}
    cadence = [r for r in iterations if r.get('decode_completion_interval')]
    cadence_wall = sum(r['completion_interval_s'] for r in cadence)
    decode_rows = [r for r in iterations if r.get('decode_only') is True]
    queue_fields_available = traced and bool(iterations) and all('waiting_requests' in r for r in iterations)
    empty_suffix_start = None
    if queue_fields_available:
        last_waiting = max((index for index, row in enumerate(iterations) if row['waiting_requests'] > 0), default=-1)
        if last_waiting + 1 < len(iterations):
            empty_suffix_start = min(run['wall_s'], iterations[last_waiting + 1]['scheduled_at_s']) if last_waiting >= 0 else 0.0
    scheduled_total = sum(r['scheduled_tokens'] for r in iterations) if traced else None
    minimal_input_total = run['prompt_tokens'] + run['output_tokens'] - n
    backlog = {
        'first_output_s': min(r['first_output_s'] for r in lives) if lives else None,
        'first_request_finished_s': min(r['last_output_s'] for r in lives) if lives else None,
        'queue_empty_suffix_start_s': empty_suffix_start,
        'queue_empty_suffix_s': run['wall_s'] - empty_suffix_start if empty_suffix_start is not None else None,
        'queue_definition': 'Waiting requests observed after scheduling. Final empty-queue suffix may include prefill and is not a pure decode tail.',
        'decode_batches': batch_distribution(decode_rows),
        'decode_batches_with_waiting_requests': batch_distribution(r for r in decode_rows if r['waiting_requests'] > 0) if queue_fields_available else None,
        'decode_batches_without_waiting_requests': batch_distribution(r for r in decode_rows if r['waiting_requests'] == 0) if queue_fields_available else None,
        'minimum_free_kv_blocks_after_schedule': min(r['free_kv_blocks_after_schedule'] for r in iterations) if traced and iterations else None,
        'scheduled_input_delta': scheduled_total - minimal_input_total if traced else None,
        'scheduled_input_delta_definition': 'Scheduled input tokens minus prompt+output-request count; net extra scheduled work, not an exact recomputation or discarded-output counter.',
        'post_all_first_outputs_window': run.get('post_all_first_outputs_window'),
    }
    return {'repeat': run['repeat'], 'phase': run['phase'], 'trace_enabled': traced,
            'backlog_profile': backlog,
            'wall_s': run['wall_s'], 'output_tok_s': run['output_tok_s'], 'prompt_sha256': run['prompt_sha256'],
            'scheduled_input_tokens': scheduled_total,
            'execution_rows_sha256': hashlib.sha256(json.dumps(workers[0]['rows'], sort_keys=True, separators=(',', ':')).encode()).hexdigest() if workers else None,
            'finished_requests': n, 'prompt_tokens': run['prompt_tokens'], 'output_tokens': run['output_tokens'],
            'preemptions': preemptions, 'free_kv_blocks_after': run['free_kv_blocks_after_completion'],
            'post_completion_quiescence_s': run['post_completion_quiescence_s'],
            'request_tpot_sample_count': len(tpot), 'request_tpot_median_s': statistics.median(tpot) if tpot else None,
            'request_tpot_p95_s': percentile(tpot, .95), 'shapes': shape_summary,
            'decode_completion_cadence': {'intervals': len(cadence), 'wall_s': cadence_wall,
                'output_tokens': sum(r['emitted_output_tokens'] for r in cadence),
                'definition': 'engine output completion intervals whose adjacent native batches are decode-only; not isolated device execution'} if cadence else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=pathlib.Path)
    args = parser.parse_args()
    rows, fleets = [], []
    entries = []
    for path in sorted(args.directory.glob('*.json')):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get('status') != 'SUCCEEDED':
            continue
        grouped = data.get('measurements', {})
        if 'runs' in grouped:
            entries.append((str(path), data, grouped))
            continue
        for index, unit in enumerate(grouped.get('fleet_cases', [])):
            if 'runs' in unit:
                entries.append((f'{path}#case{index}', data, unit))
    for label, data, measured in entries:
        if 'engine_names' in measured:
            fleet_runs = []
            classes = measured['case'].get('classes', [measured['case']])
            expected_requests = sum(c['n'] for c in classes)
            expected_outputs = sum(c['n'] * c['d'] for c in classes)
            expected_prompts = sum(c['n'] * c['p'] for c in classes)
            for run in measured['runs']:
                indices = [index for shard in run['assignment_indices'] for index in shard]
                if sorted(indices) != list(range(len(indices))):
                    raise ValueError('Fleet workload is not an exact partition')
                engines = [summarize_run(r) for r in run['engines']]
                if sum(r['finished_requests'] for r in engines) != len(indices):
                    raise ValueError('Fleet completion count differs from manifest')
                if sum(r['output_tokens'] for r in engines) != run['output_tokens']:
                    raise ValueError('Fleet output token counts differ')
                if len(indices) != expected_requests or run['output_tokens'] != expected_outputs or sum(r['prompt_tokens'] for r in engines) != expected_prompts:
                    raise ValueError('Fleet completed workload differs from its requested classes')
                overlap = run['joint_completion_wall_s'] < sum(r['wall_s'] for r in engines) if len(engines) > 1 else None
                if overlap is False:
                    raise ValueError('The multi-engine generation windows did not overlap')
                fleet_runs.append({k: v for k, v in run.items() if k != 'engines'} | {'engines': engines, 'overlapping_generation_windows': overlap})
            timed = [r for r in fleet_runs if not r['warmup'] and r['trace_enabled']]
            control = next(r for r in fleet_runs if not r['trace_enabled'])
            fleets.append({'artifact': label, 'submission_id': data['submission_id'],
                           'case': measured['case'], 'total_chips': measured['total_chips'],
                           'engine_count': len(measured['engine_names']), 'measured_repeats': len(timed),
                           'median_joint_wall_s': statistics.median(r['joint_completion_wall_s'] for r in timed),
                           'median_output_tok_s': statistics.median(r['output_tok_s'] for r in timed),
                           'min_output_tok_s': min(r['output_tok_s'] for r in timed),
                           'max_output_tok_s': max(r['output_tok_s'] for r in timed),
                           'untraced_control_output_tok_s': control['output_tok_s'],
                           'timing_semantics': measured['timing_semantics'], 'runs': fleet_runs})
            continue
        argv = data['remote_arguments']
        tp = int(argv[argv.index('--tp') + 1])
        case = measured['case']
        runs = [summarize_run(r) for r in measured['runs']]
        generation = [r for r in runs if r['phase'] == 'full_generation']
        prefills = [r for r in runs if r['phase'] == 'prefill_plus_first_token']
        controls = [r for r in runs if r['phase'] == 'full_generation_untraced_control']
        median = statistics.median(r['output_tok_s'] for r in generation)
        expected = case['n'] * case['d']
        if any(r['finished_requests'] != case['n'] or r['output_tokens'] != expected for r in generation + controls):
            raise ValueError('Completed generation differs from declared workload')
        if any(r['finished_requests'] != case['n'] or r['output_tokens'] != case['n'] for r in prefills):
            raise ValueError('Prefill-plus-first-token completion mismatch')
        if len({r['prompt_sha256'] for r in measured['runs']}) != len(measured['runs']):
            raise ValueError('Measurement phases reused a prompt manifest')
        rows.append({'artifact': label, 'submission_id': data['submission_id'], 'source_sha256': data['script_sha256'],
                     'tp': tp, 'n': case['n'], 'p': case['p'], 'd': case['d'], 'cohort_size': case.get('cohort_size', case['n']),
                     'measured_repeats': len(generation), 'median_output_tok_s': median,
                     'median_output_tok_s_per_chip': median / tp,
                     'median_generation_wall_s': statistics.median(r['wall_s'] for r in generation),
                     'median_prefill_plus_first_token_wall_s': statistics.median(r['wall_s'] for r in prefills),
                     'untraced_control_output_tok_s': controls[0]['output_tok_s'],
                     'control_rate_relative_to_traced_percent': 100 * (controls[0]['output_tok_s'] / median - 1),
                     'runs': runs})
    comparisons = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1:]:
            if any(left[k] != right[k] for k in ('n', 'p', 'd')):
                continue
            if left['tp'] != right['tp'] and left['cohort_size'] != right['cohort_size']:
                continue
            if [r['prompt_sha256'] for r in left['runs']] != [r['prompt_sha256'] for r in right['runs']]:
                raise ValueError('Matched operating points used different prompt manifests')
            signature_pairs = [(a['execution_rows_sha256'], b['execution_rows_sha256']) for a, b in zip(left['runs'], right['runs']) if a['trace_enabled'] and b['trace_enabled']]
            execution_match = all(a == b for a, b in signature_pairs) if all(a is not None and b is not None for a, b in signature_pairs) else None
            comparisons.append({'left': left['artifact'], 'right': right['artifact'],
                                'observed_execution_rows_match': execution_match,
                                'right_over_left_output_rate': right['median_output_tok_s'] / left['median_output_tok_s'],
                                'right_over_left_per_chip_rate': right['median_output_tok_s_per_chip'] / left['median_output_tok_s_per_chip'],
                                'kind': 'matched_submitted_workload' if left['tp'] != right['tp'] else 'cohort_submission'})
    fleet_comparisons = []
    for left_index, left in enumerate(fleets):
        for right in fleets[left_index + 1:]:
            left_hashes = [r['workload_sha256'] for r in left['runs']]
            if left_hashes != [r['workload_sha256'] for r in right['runs']]:
                continue
            if left['total_chips'] != right['total_chips']:
                raise ValueError('Fleet comparison does not use an equal chip budget')
            left_runs = [r for r in left['runs'] if not r['warmup'] and r['trace_enabled']]
            right_runs = [r for r in right['runs'] if not r['warmup'] and r['trace_enabled']]
            fleet_comparisons.append({'left': left['artifact'], 'right': right['artifact'],
                'right_over_left_joint_output_rate': statistics.median(r['output_tok_s'] for r in right_runs) / statistics.median(r['output_tok_s'] for r in left_runs),
                'identical_canonical_workload': True, 'total_chips': left['total_chips']})
    summary = {'timing_policy': 'completed physical vLLM outputs; no subtractive decode timing or isolated-rate fleet projection',
               'cells': rows, 'fleets': fleets, 'matched_comparisons': comparisons, 'fleet_comparisons': fleet_comparisons}
    (args.directory / 'corrected_results.json').write_text(json.dumps(summary, indent=2) + '\n')
    if rows:
        fields = [key for key in rows[0] if key != 'runs']
        with (args.directory / 'corrected_results.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    if fleets:
        fields = [key for key in fleets[0] if key != 'runs']
        with (args.directory / 'fleet_results.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(dict(row, case=json.dumps(row['case'], sort_keys=True)) for row in fleets)
    artifacts = [str(args.directory / 'corrected_results.json')]
    if rows:
        artifacts.append(str(args.directory / 'corrected_results.csv'))
    if fleets:
        artifacts.append(str(args.directory / 'fleet_results.csv'))
    print(json.dumps({'reconciled_cells': len(rows), 'reconciled_fleets': len(fleets),
                      'artifacts': artifacts}))


if __name__ == '__main__':
    main()
