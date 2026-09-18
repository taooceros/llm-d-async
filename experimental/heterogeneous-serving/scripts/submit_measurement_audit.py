#!/usr/bin/env python3
"""Submit physical measurement-audit actions through RCM's Ray supervisor."""
import argparse
import hashlib
import json
import pathlib
import shlex
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omp_rcm import StageRunner, stage
from omp_rcm.ray_job import RayJobSupervisor

parser = argparse.ArgumentParser()
parser.add_argument('--ray-url', default='http://127.0.0.1:8265')
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--timeout', type=float, default=1800)
modes = parser.add_mutually_exclusive_group()
modes.add_argument('--cases', type=pathlib.Path, help='JSON case list; --output is then a result directory')
modes.add_argument('--plan', type=pathlib.Path, help='Ordered JSON action plan; saves each step and reconciles the result directory')
parser.add_argument('remote_args', nargs=argparse.REMAINDER)
args = parser.parse_args()
remote_args = args.remote_args[1:] if args.remote_args[:1] == ['--'] else args.remote_args
if args.plan and remote_args:
    parser.error('--plan supplies its own remote arguments')
if not args.plan and not remote_args:
    parser.error('Specify audit arguments after --')
SCRIPT = ROOT / 'benchmarks/run_measurement_audit.py'
if args.plan:
    steps = json.loads(args.plan.read_text())
    if not isinstance(steps, list) or not steps:
        parser.error('--plan must contain a nonempty list of action steps')
    outputs = [step['output'] for step in steps]
    if len(outputs) != len(set(outputs)) or any(pathlib.Path(name).name != name for name in outputs):
        parser.error('Plan outputs must be unique filenames within --output')
else:
    cases = json.loads(args.cases.read_text()) if args.cases else [None]
    steps = []
    for case in cases:
        requested = list(remote_args)
        destination = args.output
        if case is not None:
            requested += ['--case', json.dumps(case)]
            destination /= f"tp{case['tp']}_p{case['p']}_d{case['d']}_n{case['n']}_c{case.get('cohort_size', case['n'])}.json"
        steps.append({'remote_args': requested, 'output': str(destination)})


def resolve_step(step):
    requested = list(step['remote_args'])
    if 'topology_source' in step:
        initialization = json.loads((args.output / step['topology_source']).read_text())['measurements']
        proof = [{'ip': worker['node_ip'], 'chip_coords': [device['coords'] for device in worker['local_devices']]}
                 for worker in initialization['devices']]
        xs = sorted({min(chip[0] for chip in host['chip_coords']) for host in proof})
        selected_x = xs[step['submesh_index']]
        hosts = [host for host in proof if min(chip[0] for chip in host['chip_coords']) == selected_x]
        requested += ['--topology-map', json.dumps(proof), '--host-ips', ','.join(host['ip'] for host in hosts)]
    return requested

calculated_expected_s = max(60.0, float(sum((step.get('timeout') or args.timeout or 1800.0) for step in steps))) if steps else float(args.timeout or 1800.0)

@stage(name='measurement_audit', order_idx=1, expected_s=calculated_expected_s)
def run_remote(state, timing):
    supervisor = state.get('supervisor')
    if supervisor is None:
        supervisor = RayJobSupervisor(ray_url=args.ray_url, timing_engine=timing)
        state.set('supervisor', supervisor)
    for index in range(state.get('case_index', 0), len(steps)):
        step = steps[index]
        requested_args = resolve_step(step)
        if not args.plan:
            requested_args = list(state.get_param('remote_args', remote_args))
            if args.cases:
                requested_args += step['remote_args'][len(remote_args):]
        destination = args.output / step['output'] if args.plan else pathlib.Path(step['output'])
        print('[OMP_EVENT: AUDIT_PLAN_STEP ' + json.dumps({'index': index, 'total': len(steps), 'output': str(destination)}) + ']', flush=True)
        submission_id = state.get('submission_id')
        previous = supervisor.get_job(submission_id) if submission_id else None
        if previous is not None and previous.status in ('FAILED', 'STOPPED'):
            submission_id = None
        if submission_id is None:
            source = SCRIPT.read_bytes()
            source_sha = hashlib.sha256(source).hexdigest()
            snapshots = destination.parent / 'source_snapshots'
            snapshots.mkdir(parents=True, exist_ok=True)
            (snapshots / (source_sha + '.py')).write_bytes(source)
            paths = [pathlib.Path(__file__), ROOT / 'admission_control/tpu/measurement_worker.py',
                     ROOT / 'benchmarks/analyze_measurement_audit.py']
            paths += [ROOT / 'benchmarks' / (name + '.py') for name in
                      ('kv_planner', 'kv_future_profile', 'kv_execution', 'kv_eviction', 'kv_admission')]
            for path in paths:
                content = path.read_bytes()
                (snapshots / (hashlib.sha256(content).hexdigest() + '.py')).write_bytes(content)
            if args.plan:
                (snapshots / (hashlib.sha256(args.plan.read_bytes()).hexdigest() + '.json')).write_bytes(args.plan.read_bytes())
            submission_id = supervisor.submit_job(
                entrypoint=shlex.join(['python3', '-u', 'benchmarks/run_measurement_audit.py', *requested_args]),
                runtime_env={
                    'working_dir': str(ROOT),
                    'excludes': ['.venv*', '.git', '.jj', 'formalization', 'skaffold', '__pycache__', '*.pyc', 'docs', 'results', 'experiments', 'reports', '**/results/**', '**/reports/**'],
                    'env_vars': {'PYTHONUNBUFFERED': '1'},
                },
                metadata={'purpose': 'physical-measurement-claim-audit', 'script_sha256': source_sha},
                name='physical-measurement-audit',
            )
            state.set('submission_id', submission_id)
        result = supervisor.supervise_until_completion(
            submission_id=submission_id, timeout_s=step.get('timeout', args.timeout),
            stage_name='measurement_audit', state_store=state,
        )
        if not result['measurements']:
            raise RuntimeError('Successful job produced no structured measurement evidence')
        evidence = {
            'submission_id': submission_id, 'status': result['status'],
            'script_sha256': supervisor.get_job(submission_id).metadata['script_sha256'],
            'remote_arguments': requested_args, 'measurements': result['measurements'],
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(evidence, indent=2) + '\n')
        temporary.replace(destination)
        state.set('case_index', index + 1)
        state.set('submission_id', None)
        print('[OMP_EVENT: AUDIT_RESULT_SAVED path=' + str(destination) + ']', flush=True)


@stage(name='reconcile_measurements', order_idx=2, expected_s=60)
def reconcile(state, timing):
    import runpy
    original_argv = sys.argv
    try:
        sys.argv = [str(ROOT / 'benchmarks/analyze_measurement_audit.py'), str(args.output)]
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    runner = StageRunner(silence_timeout_s=5400)
    runner.register(run_remote)
    if args.plan:
        runner.register(reconcile)
    if not runner.run():
        raise SystemExit(1)
