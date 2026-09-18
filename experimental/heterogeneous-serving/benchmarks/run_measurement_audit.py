#!/usr/bin/env python3
"""Physical serving remeasurement with explicit topology and measured boundaries.

Submit through the RCM RayJobSupervisor. Actors and reservations persist between
invocations so inspection and measurement retries do not reload model weights.
The release action removes only this audit's actor and placement group.
"""
import argparse
import bisect
import collections
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import random
import statistics
import sys
import time

import ray
from ray.util.placement_group import placement_group, placement_group_table, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pkg_util.tpu_topology import TPUTopologyDiscovery
import kv_planner
import kv_future_profile
import kv_execution
import kv_eviction
import kv_admission

for _policy_module in (kv_planner, kv_future_profile, kv_execution, kv_eviction, kv_admission):
    ray.cloudpickle.register_pickle_by_value(_policy_module)
POLICY_MODULE_SHA256 = {module.__name__: hashlib.sha256(
    pathlib.Path(module.__file__).read_bytes()).hexdigest()
    for module in (kv_planner, kv_future_profile, kv_execution, kv_eviction, kv_admission)}

NAMESPACE = 'omp_measurement_audit'
MODEL = '/models/gemma-4-31b'


def emit(event, **fields):
    print('[OMP_EVENT: ' + event + ' ' + json.dumps(fields, default=str) + ']', flush=True)


@ray.remote(num_cpus=0)
class ProgressRelay:
    """Out-of-band progress channel from the engine actor to the driver.

    The AuditWorker is detached, owned by an earlier job, and single-threaded: Ray does not
    forward its stdout to a later job's driver, and any method call queues behind the
    in-flight measurement. So neither printing nor a drain method on that actor can report
    liveness while a run is in progress. This separate actor is free to answer immediately.
    """

    def __init__(self):
        self._records = collections.deque()

    def push(self, record):
        self._records.append(record)

    def drain(self):
        drained = list(self._records)
        self._records.clear()
        return drained


def emit_progress(relay, event, **fields):
    """Emit locally and push to the driver's relay.

    `push` is fire-and-forget: the measurement never blocks on the relay, and a dead or slow
    relay degrades reporting without touching the run.
    """
    emit(event, **fields)
    if relay is not None:
        relay.push.remote({'event': event, 'at_wall_s': time.time(), **fields})


def device_inventory(worker):
    import jax
    return {
        'host': __import__('socket').gethostname(),
        'node_ip': ray.util.get_node_ip_address(),
        'worker_rank': worker.rank,
        'physical_tpu_worker_id': worker.physical_tpu_worker_id,
        'global_devices': [{'id': d.id, 'kind': d.device_kind, 'process_index': d.process_index,
                            'coords': list(d.coords) if hasattr(d, 'coords') else None} for d in jax.devices()],
        'local_devices': [{'id': d.id, 'coords': list(d.coords), 'memory': d.memory_stats()} for d in jax.local_devices()],
        'selected_device_ids': [d.id for d in worker.devices],
    }


@ray.remote(num_cpus=0)
class AuditWorker:
    def __init__(self, tp, ips, bounds, pg):
        self.tp, self.ips, self.bounds, self.pg = tp, ips, bounds, pg
        self.llm = None
        self.manifest = None
        os.environ.update({
            'TPU_MULTIHOST_BACKEND': 'ray',
            'TPU_TOPOLOGY': '2x4' if tp == 8 else '4x4',
            'TPU_HOST_BOUNDS': bounds,
            'TPU_PROCESS_BOUNDS': bounds,
            'TPU_CHIPS_PER_HOST_BOUNDS': '2,2,1',
            'TPU_CHIPS_PER_PROCESS_BOUNDS': '2,2,1',
            'TPU_PROCESS_ADDRESSES': ','.join(ip + ':8471' for ip in ips),
            'TPU_PROCESS_PORT': '8471',
            'JAX_PLATFORMS': 'tpu,cpu',
            'JAX_COMPILATION_CACHE_DIR': '/models/jax_cache',
            'VLLM_ENABLE_V1_MULTIPROCESSING': '0',
        })

    def initialize(self):
        if self.llm is not None:
            return self.manifest
        from vllm import LLM
        from vllm.engine.arg_utils import EngineArgs
        config_path = pathlib.Path(MODEL) / 'config.json'
        config_bytes = config_path.read_bytes()
        model_config = json.loads(config_bytes)
        weights = [{'name': p.name, 'bytes': p.stat().st_size}
                   for p in sorted(pathlib.Path(MODEL).glob('*.safetensors'))]
        if not weights:
            raise RuntimeError('Real model weight files are missing')
        original = EngineArgs.create_engine_config
        pg = self.pg

        def pinned_config(engine_args, *args, **kwargs):
            config = original(engine_args, *args, **kwargs)
            config.parallel_config.placement_group = pg
            config.parallel_config.worker_cls = 'admission_control.tpu.measurement_worker.MeasurementTPUWorker'
            return config

        EngineArgs.create_engine_config = pinned_config
        start = time.perf_counter()
        try:
            self.llm = LLM(
                model=MODEL, tensor_parallel_size=self.tp,
                distributed_executor_backend='ray', dtype='bfloat16',
                max_model_len=8192, max_num_seqs=512,
                max_num_batched_tokens=2048,
                gpu_memory_utilization=0.85, enable_prefix_caching=False,
                skip_tokenizer_init=True, enforce_eager=False,
                limit_mm_per_prompt={'image': 0, 'audio': 0, 'video': 0},
            )
        finally:
            EngineArgs.create_engine_config = original
        engine = self.llm.llm_engine
        config = engine.vllm_config
        if config.parallel_config.placement_group.id != self.pg.id:
            raise RuntimeError('Executor did not retain the explicit placement group')
        if config.cache_config.enable_prefix_caching:
            raise RuntimeError('Prefix caching unexpectedly enabled')
        if config.load_config.load_format == 'dummy':
            raise RuntimeError('Dummy weights are not a measurement backend')
        devices = engine.collective_rpc(device_inventory)
        if len(devices) != len(self.ips) or any(len(x['global_devices']) != self.tp for x in devices):
            raise RuntimeError('Actual JAX device topology differs from requested topology: ' + str(devices))
        core = engine.engine_core.engine_core
        scheduler = core.scheduler
        pool = scheduler.kv_cache_manager.block_pool
        versions = {}
        for name in ('vllm', 'tpu-inference', 'jax', 'jaxlib', 'ray', 'transformers'):
            versions[name] = importlib.metadata.version(name)
        self.manifest = {
            'backend': 'real_vllm_serving', 'input_kind': 'synthetic_token_ids',
            'model': MODEL, 'model_config': model_config,
            'model_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
            'weight_files': weights, 'versions': versions,
            'tp': self.tp, 'host_ips': self.ips, 'host_bounds': self.bounds,
            'placement_group_id': self.pg.id.hex(), 'devices': devices,
            'initialization_s': time.perf_counter() - start,
            'scheduler_class': type(scheduler).__module__ + '.' + type(scheduler).__name__,
            'scheduler_config': dataclasses.asdict(config.scheduler_config),
            'cache_config': dataclasses.asdict(config.cache_config),
            'batch_queue_size': core.batch_queue_size,
            'block_pool_total': pool.num_gpu_blocks,
            'block_pool_free': pool.get_num_free_blocks(),
        }
        return self.manifest

    def apply(self, fn, *args):
        if self.llm is None:
            raise RuntimeError('Engine has not initialized')
        return fn(self.llm, *args)

def run_with_progress_relay(actor, fn, *args, poll_interval_s=20.0):
    """Call into the detached engine actor while relaying its progress to this driver.

    Ray forwards a worker's stdout only to the driver whose job owns it, and the AuditWorker
    is detached and owned by the earlier initialize job, so a blocking `ray.get` on it is
    silent no matter how healthy the run is. The worker is also single-threaded, so a drain
    method on it would queue behind the measurement. A separate relay actor sidesteps both:
    it answers immediately and its records are re-emitted here, where RCM can see them.
    """
    relay = ProgressRelay.remote()
    pending = actor.apply.remote(fn, *args, relay)
    emit('AUDIT_RELAY_START', call=getattr(fn, '__name__', str(fn)), poll_interval_s=poll_interval_s)
    polls = 0
    relayed = 0
    try:
        while True:
            ready, _ = ray.wait([pending], timeout=poll_interval_s)
            for record in ray.get(relay.drain.remote()):
                relayed += 1
                emit(record.pop('event', 'AUDIT_RELAYED'), **record)
            polls += 1
            if ready:
                result = ray.get(pending)
                emit('AUDIT_RELAY_DONE', call=getattr(fn, '__name__', str(fn)),
                     polls=polls, relayed_events=relayed)
                return result
            emit('AUDIT_RELAY_ALIVE', call=getattr(fn, '__name__', str(fn)),
                 polls=polls, relayed_events=relayed)
    finally:
        ray.kill(relay, no_restart=True)


def unpatch_worker_hooks(worker):
    """Drop a _prepare_inputs hook orphaned by a killed job."""
    runner = worker.model_runner
    original = getattr(runner, '_audit_prepare_original', None)
    if original is None:
        return {'rank': worker.rank, 'restored': False}
    runner._prepare_inputs = original
    del runner._audit_prepare_original
    if hasattr(runner, '_audit_shape_rows'):
        del runner._audit_shape_rows
    return {'rank': worker.rank, 'restored': True}


def recover_quiescence(llm):
    """Return a reused detached actor to a clean state.

    measure_policy patches scheduler.schedule / update_from_output as *instance*
    attributes shadowing the class methods. If the job is killed, its `finally`
    never runs and those closures survive on the actor, bound to a relay that no
    longer exists -- the next run then hangs emitting into the void. Deleting the
    instance attributes re-exposes the class methods.
    """
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    # Presence of an instance attribute proves nothing: measure_policy's `finally` assigns the
    # ORIGINAL bound methods back onto the instance, so a clean run also leaves them in vars().
    # Distinguish by identity against the class function.
    hook_state = {}
    for nm in ('schedule', 'update_from_output'):
        inst = vars(scheduler).get(nm)
        if inst is None:
            hook_state[nm] = {'present': False}
            continue
        cls_fn = getattr(type(scheduler), nm, None)
        underlying = getattr(inst, '__func__', None)
        hook_state[nm] = {
            'present': True,
            'qualname': getattr(inst, '__qualname__', None) or getattr(underlying, '__qualname__', None),
            'module': getattr(inst, '__module__', None),
            'is_original_bound_method': underlying is not None and underlying is cls_fn,
            'is_closure': getattr(inst, '__closure__', None) is not None,
        }
    stale = [nm for nm, st in hook_state.items()
             if st.get('present') and not st.get('is_original_bound_method')]
    for nm in stale:
        del vars(scheduler)[nm]
    worker_unpatch = engine.collective_rpc(unpatch_worker_hooks)
    before = {'client_requests': engine.get_num_unfinished_requests(),
              'scheduler_requests': scheduler.get_num_unfinished_requests(),
              'queued_batches': len(core.batch_queue or ())}
    start = time.perf_counter()   # drain only; excludes the unpatch and RPC above
    while engine.has_unfinished_requests() or core.batch_queue or scheduler.has_requests():
        if not core.batch_queue and not scheduler.has_requests():
            raise RuntimeError('Client request state has no corresponding engine work')
        engine.step()
    return {'purpose': 'untimed recovery of interrupted audit work', 'before': before,
            'scheduler_hook_state': hook_state,
            'stale_hooks_removed': stale,
            'worker_hook_unpatch': worker_unpatch,
            'drain_wall_s': time.perf_counter() - start,
            'free_kv_blocks': scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
            'queued_batches_after': len(core.batch_queue or ())}

def capture_worker_shapes(worker, enabled):
    runner = worker.model_runner
    edges = (256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
    if enabled:
        if hasattr(runner, '_audit_prepare_original'):
            raise RuntimeError('A previous shape capture was not closed')
        original = runner._prepare_inputs
        rows = []
        def prepare(scheduled):
            count = runner.input_batch.num_reqs
            computed = runner.input_batch.num_computed_tokens_cpu[:count]
            histogram, lower = [], 0
            for upper in edges:
                histogram.append(int(((computed >= lower) & (computed < upper)).sum()))
                lower = upper
            histogram.append(int((computed >= lower).sum()))
            row = {
                'scheduled_tokens': int(scheduled.total_num_scheduled_tokens),
                'scheduled_requests': len(scheduled.num_scheduled_tokens),
                'executed_requests': int(count),
                'decode_only': bool(runner.input_batch.request_distribution[0] == count),
                'context_position_min': int(computed.min()),
                'context_position_max': int(computed.max()),
                'context_sum': int(computed.sum()),
                'context_sum_sliding': int(computed.clip(None, 1024).sum()),
                'context_histogram': histogram,
            }
            inputs = original(scheduled)
            row.update(padded_input_tokens=int(inputs[0].shape[0]), padded_requests=int(inputs[7]))
            rows.append(row)
            return inputs
        runner._audit_prepare_original = original
        runner._audit_shape_rows = rows
        runner._prepare_inputs = prepare
        return {'rank': worker.rank, 'capturing': True}
    runner._prepare_inputs = runner._audit_prepare_original
    rows = runner._audit_shape_rows
    del runner._audit_prepare_original, runner._audit_shape_rows
    return {'rank': worker.rank, 'node_ip': __import__('socket').gethostbyname(__import__('socket').gethostname()),
            'num_tokens_paddings': list(runner.num_tokens_paddings),
            'num_reqs_paddings': list(runner.num_reqs_paddings),
            'context_histogram_edges': list(edges),
            'continue_decode': bool(runner.enable_continue_decode), 'rows': rows}


def measure(llm, case, repeats, workload=None, completion=None):
    from vllm import SamplingParams
    n, p, d = case['n'], case['p'], case['d']
    cohort_size = case.get('cohort_size', n)
    dimensions = [(len(r['prompt_token_ids']), r['d']) for r in workload] if workload is not None else [(p, d)]
    if min(n, p, d, cohort_size, repeats) <= 0 or any(min(a, b) <= 0 or a + b > llm.llm_engine.vllm_config.model_config.max_model_len for a, b in dimensions):
        raise ValueError('Invalid measurement dimensions; no implicit clipping is permitted')
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    pool = scheduler.kv_cache_manager.block_pool
    model_config = json.loads((pathlib.Path(MODEL) / 'config.json').read_text())
    vocab = model_config.get('text_config', model_config)['vocab_size']
    seed_fields = {'p': p, 'd': d, 'seed': case.get('seed', 42)}
    case_seed = int(hashlib.sha256(json.dumps(seed_fields, sort_keys=True).encode()).hexdigest()[:8], 16)
    records = []

    def one_run(repeat, output_len, phase, warmup=False, trace=True):
        if engine.has_unfinished_requests() or scheduler.get_num_unfinished_requests() or core.batch_queue:
            raise RuntimeError('Previous workload has not drained')
        phase_offset = int(phase == 'prefill_plus_first_token')
        if workload is None:
            rng = random.Random(case_seed + repeat * 1009 + output_len * 17 + phase_offset)
            prompts = [{'prompt_token_ids': [rng.randrange(100, vocab) for _ in range(p)]} for _ in range(n)]
            output_lengths = [output_len] * n
        else:
            prompts = [{'prompt_token_ids': r['prompt_token_ids']} for r in workload]
            output_lengths = [r['d'] for r in workload]
        prompt_hash = hashlib.sha256(json.dumps(prompts, separators=(',', ':')).encode()).hexdigest()
        sampling = [SamplingParams(max_tokens=length, temperature=0.0, ignore_eos=True, detokenize=False) for length in output_lengths] if workload is not None else SamplingParams(max_tokens=output_len, temperature=0.0, ignore_eos=True, detokenize=False)
        expected_outputs = {}
        from collections import deque
        iterations, pending = [], deque()
        worker_shapes = []
        if trace:
            engine.collective_rpc(capture_worker_shapes, args=(True,))
        original_schedule = scheduler.schedule
        original_update = scheduler.update_from_output
        seen_outputs = set()
        lifecycles = {}
        emitted_tokens = 0
        decode_window = None
        had_preemption = False
        run_start = None

        def schedule(*args, **kwargs):
            t0 = time.perf_counter()
            result = original_schedule(*args, **kwargs)
            for request in result.scheduled_new_reqs:
                expected_outputs[request.req_id] = scheduler.requests[request.req_id].max_tokens
            preempted = getattr(result, 'preempted_req_ids', None)
            row = {
                'iteration': len(iterations), 'scheduled_at_s': t0 - run_start,
                'scheduled_tokens': result.total_num_scheduled_tokens,
                'scheduled_requests': len(result.num_scheduled_tokens),
                'running_requests': len(scheduler.running),
                'waiting_requests': len(scheduler.waiting),
                'free_kv_blocks_after_schedule': pool.get_num_free_blocks(),
                'total_kv_blocks': pool.num_gpu_blocks,
                'preempted_request_ids': list(preempted) if preempted is not None else None,
                'all_scheduled_requests_previously_emitted': all(rid in seen_outputs for rid in result.num_scheduled_tokens),
            }
            iterations.append(row)
            pending.append((row, t0, result.num_scheduled_tokens))
            return result

        def update(scheduled, model_output, *args, **kwargs):
            nonlocal emitted_tokens, decode_window, had_preemption
            result = original_update(scheduled, model_output, *args, **kwargs)
            now = time.perf_counter()
            row, t0, expected_tokens = pending.popleft()
            if scheduled.num_scheduled_tokens != expected_tokens:
                raise RuntimeError('Scheduled and completed iteration identities differ')
            new_tokens = 0
            finished = []
            for batch in result.values():
                for output in batch.outputs:
                    new_tokens += len(output.new_token_ids)
                    if output.new_token_ids:
                        seen_outputs.add(output.request_id)
                        life = lifecycles.get(output.request_id)
                        if life is None:
                            life = {'request_id': output.request_id, 'first_output_s': now - run_start,
                                    'last_output_s': now - run_start, 'output_tokens': 0}
                            lifecycles[output.request_id] = life
                        life['last_output_s'] = now - run_start
                        life['output_tokens'] += len(output.new_token_ids)
                    if output.finish_reason is not None:
                        finished.append(output.request_id)
                        if output.request_id in lifecycles:
                            lifecycles[output.request_id]['finish_reason'] = str(output.finish_reason)
            emitted_tokens += new_tokens
            had_preemption |= bool(row['preempted_request_ids'])
            row.update(completed_at_s=now - run_start,
                       scheduled_to_output_s=now - t0,
                       emitted_output_tokens=new_tokens,
                       finished_request_ids=finished)
            if output_len > 1 and len(seen_outputs) == n and decode_window is None:
                decode_window = {'start_s': now - run_start, 'outputs_before_window': emitted_tokens}
            return result

        if trace:
            scheduler.schedule, scheduler.update_from_output = schedule, update
        cohorts = []
        try:
            run_start = time.perf_counter()
            run_start_wall = time.time()
            actual_prompt_tokens = actual_output_tokens = finished_requests = 0
            for offset in range(0, n, cohort_size):
                chunk = prompts[offset:offset + cohort_size]
                start = time.perf_counter()
                chunk_sampling = sampling[offset:offset + cohort_size] if isinstance(sampling, list) else sampling
                outputs = llm.generate(chunk, chunk_sampling, use_tqdm=False)
                elapsed = time.perf_counter() - start
                lengths = [len(out.outputs[0].token_ids) for out in outputs]
                prompt_lengths = [len(out.prompt_token_ids) for out in outputs]
                wanted = output_lengths[offset:offset + cohort_size]
                if len(outputs) != len(chunk) or lengths != wanted or any(out.prompt_token_ids != prompt['prompt_token_ids'] for out, prompt in zip(outputs, chunk)):
                    raise RuntimeError('Actual completed workload differs from requested dimensions')
                actual_prompt_tokens += sum(prompt_lengths)
                actual_output_tokens += sum(lengths)
                finished_requests += len(outputs)
                cohorts.append({'requests': len(chunk), 'wall_s': elapsed,
                                'prompt_tokens': sum(prompt_lengths), 'output_tokens': sum(lengths)})
            wall = time.perf_counter() - run_start
            if completion is not None:
                completion[0].put({'index': completion[1], 'completed': True})
            drain_start = time.perf_counter()
            while core.batch_queue or scheduler.has_requests():
                engine.step()
            quiescence_drain_s = time.perf_counter() - drain_start
        finally:
            scheduler.schedule, scheduler.update_from_output = original_schedule, original_update
            if trace:
                worker_shapes = engine.collective_rpc(capture_worker_shapes, args=(False,))
        if engine.has_unfinished_requests() or scheduler.get_num_unfinished_requests():
            raise RuntimeError('Measurement returned with unfinished requests')
        if trace and emitted_tokens != actual_output_tokens:
            raise RuntimeError(f'Trace output count {emitted_tokens} != completed output count {actual_output_tokens}')
        if trace and (len(lifecycles) != n or any(life['output_tokens'] != expected_outputs[request_id] for request_id, life in lifecycles.items())):
            raise RuntimeError('Per-request output events do not reconcile with the completed workload')
        if trace:
            executed = [r for r in iterations if r['scheduled_tokens'] > 0]
            for worker in worker_shapes:
                if len(worker['rows']) != len(executed):
                    raise RuntimeError('Worker model executions do not match scheduled iterations')
                for scheduled_row, worker_row in zip(executed, worker['rows']):
                    if scheduled_row['scheduled_tokens'] != worker_row['scheduled_tokens']:
                        raise RuntimeError('Worker and scheduler disagree on executed tokens')
            for scheduled_row, worker_row in zip(executed, worker_shapes[0]['rows']):
                scheduled_row.update(worker_row)
            previous = None
            for completed in iterations:
                if 'completed_at_s' not in completed:
                    continue
                if previous is not None:
                    completed['completion_interval_s'] = completed['completed_at_s'] - previous['completed_at_s']
                    completed['decode_completion_interval'] = bool(completed.get('decode_only') and previous.get('decode_only'))
                previous = completed
        for life in lifecycles.values():
            life['observed_tpot_s'] = ((life['last_output_s'] - life['first_output_s']) / (life['output_tokens'] - 1) if life['output_tokens'] > 1 else None)
        if decode_window is not None:
            decode_window.update(wall_s=wall - decode_window['start_s'],
                                 output_tokens=actual_output_tokens - decode_window['outputs_before_window'],
                                 preemption_free=(not had_preemption if iterations and all(r['preempted_request_ids'] is not None for r in iterations) else None),
                                 definition='last request first output observed through final completion; not all decode work')
        row = {'repeat': repeat, 'phase': phase, 'warmup': warmup, 'trace_enabled': trace,
               'prompt_sha256': prompt_hash, 'wall_s': wall,
               'run_start_wall_s': run_start_wall,
               'post_completion_quiescence_s': quiescence_drain_s,
               'prompt_tokens': actual_prompt_tokens, 'output_tokens': actual_output_tokens,
               'finished_requests': finished_requests,
               'output_tok_s': actual_output_tokens / wall,
               'prompt_tok_s': actual_prompt_tokens / wall if output_len == 1 else None,
               'cohorts': cohorts, 'iterations': iterations, 'worker_shapes': worker_shapes,
               'request_lifecycles': list(lifecycles.values()),
               'request_timing_semantics': 'engine-core output events; TPOT=(last-first)/(actual_outputs-1)',
               'post_all_first_outputs_window': decode_window,
               'free_kv_blocks_after_completion': pool.get_num_free_blocks(),
               'total_kv_blocks': pool.num_gpu_blocks}
        if not warmup:
            records.append(row)
        emit('AUDIT_ITERATION', case=case, repeat=repeat, phase=phase, warmup=warmup,
             wall_s=wall, outputs=actual_output_tokens, output_tok_s=row['output_tok_s'])
        return row

    if workload is not None:
        return one_run(case['repeat'], d, 'full_generation', warmup=case.get('warmup', False), trace=case.get('trace', True))

    one_run(-2, 1, 'prefill_plus_first_token', warmup=True)
    one_run(-1, d, 'full_generation', warmup=True)
    for repeat in range(repeats):
        one_run(repeat, 1, 'prefill_plus_first_token')
        one_run(repeat, d, 'full_generation')
    one_run(repeats, d, 'full_generation_untraced_control', trace=False)
    return {'case': case, 'repeats': repeats, 'runs': records,
            'timing_semantics': 'wall clock through completed vLLM outputs; iteration intervals may overlap under async execution',
            'full_generation_median_output_tok_s': statistics.median(r['output_tok_s'] for r in records if r['phase'] == 'full_generation')}


POLICY_ORDERS = ('arrival', 'short_first', 'long_first', 'alternating',
                 'short_decode_first', 'long_decode_first', 'short_decode_then_short_prompt', 'planned')
POLICY_ADMIT = ('greedy', 'memory_safe', 'watermark', 'trajectory', 'planner')


# Read on the submitter and transfer bytes explicitly; actor paths have a different lifetime.
SHAREGPT_PATH = pathlib.Path(__file__).resolve().parents[1] / 'data/workloads/sharegpt_sampled_500.json'
SHAREGPT_LARGE_PATH = pathlib.Path(__file__).resolve().parents[1] / 'data/workloads/sharegpt_large.json'
_TOKENIZED_SHAREGPT = {}


def cache_prompt_catalogue(llm, catalogue, content=None, digest=None):
    """Persist verified conversational tokens with the hot engine, not a Ray job directory."""
    if not hasattr(llm, '_audit_prompt_catalogues'):
        llm._audit_prompt_catalogues = {}
    cached = llm._audit_prompt_catalogues.get(catalogue)
    if content is None:
        return None if cached is None else {'sha256': cached['sha256'],
                                           'prompts': len(cached['prompts'])}
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError('Conversational catalogue transport checksum mismatch')
    if cached is None or cached['sha256'] != digest:
        from transformers import AutoTokenizer
        entries = json.loads(content)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        encoded = tokenizer([entry['prompt'] for entry in entries],
                            add_special_tokens=True)['input_ids']
        cached = {'sha256': digest, 'prompts': [
            {'prompt_id': entry['id'], 'prompt_token_ids': list(ids)}
            for entry, ids in zip(entries, encoded)]}
        llm._audit_prompt_catalogues[catalogue] = cached
    return {'sha256': cached['sha256'], 'prompts': len(cached['prompts'])}


def ensure_prompt_catalogue(actor, catalogue):
    path = {'small': SHAREGPT_PATH, 'large': SHAREGPT_LARGE_PATH}[catalogue]
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    cached = ray.get(actor.apply.remote(cache_prompt_catalogue, catalogue))
    if cached is None or cached['sha256'] != digest:
        cached = ray.get(actor.apply.remote(cache_prompt_catalogue, catalogue, content, digest))
    emit('AUDIT_CATALOGUE_READY', catalogue=catalogue, **cached)


def bind_prompt_catalogues(llm):
    _TOKENIZED_SHAREGPT.clear()
    _TOKENIZED_SHAREGPT.update({key: value['prompts'] for key, value in
        getattr(llm, '_audit_prompt_catalogues', {}).items()})


def load_sharegpt_prompts(catalogue='small', min_prompt_tokens=0):
    """Return engine-owned conversational tokens, memoized per catalogue.

    `small` is the canonical 500-prompt sample. `large` is 27,887 first-human turns drawn
    from ShareGPT_V3_unfiltered_cleaned_split, retained because 500 distinct prompts against
    a batch of 200+ forces a 72 % duplicate rate and leaves too few requests for a backlog
    deep enough to schedule against.

    `min_prompt_tokens` filters AFTER tokenization, so the threshold is in real tokens rather
    than a character proxy. The filtered catalogue size is what callers must record.
    """
    if catalogue not in ('small', 'large'):
        raise ValueError(f'Unknown prompt catalogue {catalogue!r}')
    if catalogue not in _TOKENIZED_SHAREGPT:
        raise RuntimeError('Prompt catalogue was not transferred and bound to the hot engine')
    prompts = _TOKENIZED_SHAREGPT[catalogue]
    if min_prompt_tokens > 0:
        prompts = [p for p in prompts if len(p['prompt_token_ids']) >= min_prompt_tokens]
        if not prompts:
            raise ValueError(f'No prompt in catalogue {catalogue!r} reaches {min_prompt_tokens} tokens')
    return prompts


def build_policy_workload(classes, vocab, seed):
    rng = random.Random(seed)
    workload = []
    for class_index, spec in enumerate(classes):
        for _ in range(spec['n']):
            workload.append({'index': len(workload), 'class_index': class_index,
                             'p': spec['p'], 'd': spec['d'],
                             'prompt_token_ids': [rng.randrange(100, vocab) for _ in range(spec['p'])]})
    rng.shuffle(workload)
    for position, request in enumerate(workload):
        request['arrival_index'] = position
    return workload


def build_trace_workload(classes, seed, max_model_len):
    """Real conversational prompts at their natural tokenized lengths.

    Memory pressure is created by the backlog size and the decode horizon of each class,
    never by fabricating prompt tokens. Prompts are drawn without replacement so every
    class holds a disjoint slice of the canonical trace.
    """
    catalogue = sorted(load_sharegpt_prompts(), key=lambda p: p['prompt_id'])
    rng = random.Random(seed)
    workload, taken = [], set()
    for class_index, spec in enumerate(classes):
        eligible = [p for p in catalogue if p['prompt_id'] not in taken
                    and 0 < len(p['prompt_token_ids']) + spec['d'] <= max_model_len]
        if len(eligible) < spec['n']:
            raise ValueError(f'Trace class {class_index} needs {spec["n"]} unused prompts within the '
                             f'context limit at d={spec["d"]}; {len(eligible)} qualify')
        for prompt in rng.sample(eligible, spec['n']):
            taken.add(prompt['prompt_id'])
            workload.append({'index': len(workload), 'class_index': class_index,
                             'p': len(prompt['prompt_token_ids']), 'd': spec['d'],
                             'prompt_id': prompt['prompt_id'],
                             'prompt_token_ids': prompt['prompt_token_ids']})
    rng.shuffle(workload)
    for position, request in enumerate(workload):
        request['arrival_index'] = position
    return workload


def build_real_prompt(prompts, ordered, target_len, salt):
    """A prompt of `target_len` real ShareGPT tokens.

    A contiguous prefix of one real prompt when one is long enough, otherwise consecutive
    WHOLE real prompts concatenated at prompt boundaries. Never a stride through a flattened
    token pool -- that produces scrambled synthetic text that is not a conversational workload.

    Returns (token_ids, source_prompt_indices, construction_mode) so the manifest can record
    exactly how every prompt was assembled.
    """
    if target_len < 1:
        raise ValueError('Prompt target length must be positive')
    if len(prompts[ordered[0]]['prompt_token_ids']) >= target_len:
        for index in ordered:
            tokens = prompts[index]['prompt_token_ids']
            if len(tokens) >= target_len:
                return list(tokens[:target_len]), [index], 'prefix_of_one_prompt'
    used, tokens = [], []
    cursor = salt % len(prompts)
    while len(tokens) < target_len:
        source = prompts[cursor % len(prompts)]['prompt_token_ids']
        tokens.extend(source)
        used.append(cursor % len(prompts))
        cursor += 1
    return tokens[:target_len], used, 'concatenated_whole_prompts'


def draw_decode_lengths(count, sigma, median, d_max, rng):
    """Capped (winsorized) integer log-normal decode horizons.

    Over-limit draws are clamped to d_max rather than discarded, so the realised
    distribution carries a point atom at the cap. Summaries must be computed after this
    rounding and clipping, never from the analytic log-normal.
    """
    mu = math.log(median)
    lengths = []
    for _ in range(count):
        value = int(round(math.exp(rng.gauss(mu, sigma))))
        lengths.append(min(max(value, 1), d_max))
    return lengths


def build_drawn_workload(spec, seed, max_model_len):
    """Backlog with decode horizons drawn from a capped log-normal.

    By default prompts keep their natural tokenized lengths. Backlogs larger than the
    catalogue draw prompts with replacement, which is sound only because prefix caching is
    disabled; the duplicate rate is recorded so the report can state it.

    `prompt_median` switches to *synthesised prompt lengths built from real ShareGPT text*:
    each request's length is drawn from a capped log-normal and the token ids are assembled
    by `build_real_prompt`, which uses a contiguous prefix of one real prompt or consecutive
    whole prompts concatenated at prompt boundaries. This is needed because the catalogue
    tops out near 1000 tokens, so a long-prefill regime is unreachable by filtering alone.
    Every request records how its prompt was constructed and from which sources.

    `rho` couples prompt length to decode length through a Gaussian copula. This is not
    'the same workload plus a predictor': correlating p with d changes joint KV demand and
    co-residency, so a correlated run is a different scheduling problem.
    """
    count = spec['n']
    sigma = float(spec['sigma'])
    median = int(spec.get('median', 256))
    d_max = int(spec.get('d_max', 4096))
    rho = float(spec.get('rho', 0.0))
    prompt_median = spec.get('prompt_median')
    prompt_sigma = float(spec.get('prompt_sigma', sigma))
    prompt_max = int(spec.get('prompt_max', max_model_len // 2))
    if count < 1 or sigma <= 0.0 or median < 1 or d_max < 1:
        raise ValueError('Invalid drawn-workload specification')
    if not -1.0 < rho < 1.0:
        raise ValueError('Copula rank correlation must lie in (-1, 1)')
    if prompt_median is not None and (int(prompt_median) < 1 or prompt_sigma <= 0.0):
        raise ValueError('Invalid synthesised prompt-length specification')
    source = spec.get('catalogue', 'small')
    min_prompt_tokens = int(spec.get('min_prompt_tokens', 0))
    catalogue = sorted(load_sharegpt_prompts(source, min_prompt_tokens),
                       key=lambda p: p['prompt_id'])
    rng = random.Random(seed)
    with_replacement = count > len(catalogue)
    if with_replacement:
        prompts = [catalogue[rng.randrange(len(catalogue))] for _ in range(count)]
    else:
        prompts = rng.sample(catalogue, count)
    if rho == 0.0:
        horizons = draw_decode_lengths(count, sigma, median, d_max, rng)
    else:
        mu = math.log(median)
        horizons = []
        ranked = sorted(range(count), key=lambda i: len(prompts[i]['prompt_token_ids']))
        normal_by_index = {}
        for rank, index in enumerate(ranked):
            quantile = (rank + 0.5) / count
            normal_by_index[index] = math.sqrt(2.0) * _erfinv(2.0 * quantile - 1.0)
        for index in range(count):
            coupled = rho * normal_by_index[index] + math.sqrt(1.0 - rho * rho) * rng.gauss(0.0, 1.0)
            value = int(round(math.exp(mu + sigma * coupled)))
            horizons.append(min(max(value, 1), d_max))
    workload, clipped_by_context = [], 0
    ordered = sorted(range(len(catalogue)),
                     key=lambda i: -len(catalogue[i]['prompt_token_ids']))
    for index, prompt in enumerate(prompts):
        if prompt_median is None:
            token_ids = prompt['prompt_token_ids']
            prompt_id, sources, construction = prompt['prompt_id'], None, 'natural_prompt'
        else:
            target = int(round(math.exp(math.log(int(prompt_median))
                                        + prompt_sigma * rng.gauss(0.0, 1.0))))
            target = min(max(target, 1), prompt_max)
            token_ids, sources, construction = build_real_prompt(catalogue, ordered, target, seed + index)
            prompt_id = f'{construction}:{target}:{sources[0]}'
        prompt_length = len(token_ids)
        horizon = horizons[index]
        if prompt_length + horizon > max_model_len:
            horizon = max_model_len - prompt_length
            clipped_by_context += 1
        if horizon < 1:
            raise ValueError(f'Prompt {prompt_id} leaves no decode budget within the context limit')
        request = {'index': index, 'class_index': 0, 'p': prompt_length, 'd': horizon,
                   'prompt_id': prompt_id, 'prompt_token_ids': token_ids}
        if sources is not None:
            request['prompt_construction'] = construction
            request['source_prompt_indices'] = sources
        workload.append(request)
    rng.shuffle(workload)
    for position, request in enumerate(workload):
        request['arrival_index'] = position
    distinct = len({r['prompt_id'] for r in workload})
    summary = {'sigma': sigma, 'median_target': median, 'd_max': d_max, 'rho': rho,
               'catalogue': source, 'catalogue_size': len(catalogue),
               'min_prompt_tokens': min_prompt_tokens,
               'prompt_median_target': prompt_median, 'prompt_sigma': prompt_sigma,
               'with_replacement': with_replacement,
               'duplicate_rate': 1.0 - distinct / count,
               'clipped_by_context': clipped_by_context}
    return workload, summary


def _erfinv(x):
    """Inverse error function: Winitzki's initial guess, Newton-refined to machine precision."""
    if not -1.0 < x < 1.0:
        raise ValueError('erfinv domain is (-1, 1)')
    a = 0.147
    ln_term = math.log(1.0 - x * x)
    first = 2.0 / (math.pi * a) + ln_term / 2.0
    guess = math.copysign(math.sqrt(max(math.sqrt(first * first - ln_term / a) - first, 0.0)), x)
    for _ in range(8):
        error = math.erf(guess) - x
        if abs(error) <= 1e-15:
            break
        guess -= error / (2.0 / math.sqrt(math.pi) * math.exp(-guess * guess))
    return guess


def summarize_horizons(workload):
    """Realised decode-horizon statistics, computed after rounding and clipping."""
    horizons = sorted(r['d'] for r in workload)
    count = len(horizons)
    mean = sum(horizons) / count
    variance = sum((h - mean) ** 2 for h in horizons) / count
    ceiling = horizons[-1]

    def quantile(fraction):
        return horizons[min(count - 1, int(fraction * count))]

    return {'count': count, 'median': quantile(0.5), 'mean': mean,
            'cv': math.sqrt(variance) / mean, 'p90': quantile(0.9), 'p99': quantile(0.99),
            'max': ceiling, 'atom_at_max': sum(1 for h in horizons if h == ceiling) / count,
            'total_decode_tokens': sum(horizons)}


def order_workload(workload, order):
    """Dispatch order.

    Decode-keyed orders sort on `d_sort`, never on `d` directly. Track C sets `d_sort` to
    the true horizon; Track O sets it to an estimate and is never handed the true value,
    so the information separation is structural rather than a matter of discipline.
    """
    if order in ('short_decode_first', 'long_decode_first', 'short_decode_then_short_prompt'):
        missing = [r['index'] for r in workload if 'd_sort' not in r]
        if missing:
            raise ValueError(f'Order {order} requires a d_sort key on every request; {len(missing)} lack one')
        if order == 'short_decode_first':
            return sorted(workload, key=lambda r: (r['d_sort'], r['arrival_index']))
        if order == 'long_decode_first':
            return sorted(workload, key=lambda r: (-r['d_sort'], r['arrival_index']))
        return sorted(workload, key=lambda r: (r['d_sort'], r['p'], r['arrival_index']))
    if order == 'arrival':
        return list(workload)
    if order == 'short_first':
        return sorted(workload, key=lambda r: (r['p'], r['arrival_index']))
    if order == 'long_first':
        return sorted(workload, key=lambda r: (-r['p'], r['arrival_index']))
    ranked = sorted(workload, key=lambda r: (r['p'], r['arrival_index']))
    short, long_side = ranked[:len(ranked) // 2], ranked[len(ranked) // 2:][::-1]
    interleaved = []
    for position in range(max(len(short), len(long_side))):
        if position < len(long_side):
            interleaved.append(long_side[position])
        if position < len(short):
            interleaved.append(short[position])
    return interleaved


ORACLE_REQUEST_FIELDS = ('d',)
POLICY_TRACKS = ('C', 'O')
POLICY_SORT_KEYS = (None, 'true_d', 'prompt', 'online_estimate', 'attained_service')


def policy_view(request, track):
    """The request as a policy is allowed to see it.

    Track O gets a copy with the oracle fields removed, so a Track O policy cannot read
    the true decode length even by accident; `d_sort` must then come from an estimator.
    The engine still needs `d` to set `max_tokens`, and reads it from the manifest by
    request index — never through the policy's view.
    """
    if track == 'C':
        return dict(request)
    if track != 'O':
        raise ValueError(f'Unknown policy track {track!r}')
    return {key: value for key, value in request.items() if key not in ORACLE_REQUEST_FIELDS}


class OnlineLengthEstimator:
    """N-D: predict decode length from prompt length using completed requests only.

    Fitted on (prompt tokens, *observed* output tokens) pairs reported by the engine for
    requests that have already finished in this run. The manifest is never read: this
    object holds no true decode length for any request, pending or complete, so there is
    nothing here for a Track O policy to leak. Scoring lives in `PredictionLedger`, which
    the harness owns and the policy never sees.

    When d is drawn independently of p the fit is flat by construction and the estimator
    cannot rank anything — that is a finding about online scheduling, not a defect.
    """

    def __init__(self, neighbours=32):
        if neighbours < 1:
            raise ValueError('Estimator neighbourhood must be positive')
        self.neighbours = neighbours
        self._prompts = []
        self._lengths = []

    def observe(self, prompt_tokens, observed_output_tokens):
        """Learn from a completion the engine actually reported."""
        position = bisect.bisect_left(self._prompts, prompt_tokens)
        self._prompts.insert(position, prompt_tokens)
        self._lengths.insert(position, observed_output_tokens)

    @property
    def sample_count(self):
        return len(self._prompts)

    def estimate(self, prompt_tokens):
        """Mean output length over the k completed requests with the closest prompt length.

        Returns None while nothing has completed, which the caller must treat as 'no
        ranking signal' rather than substituting a length-correlated stand-in.
        """
        count = len(self._prompts)
        if count == 0:
            return None
        position = bisect.bisect_left(self._prompts, prompt_tokens)
        low, high = position - 1, position
        picked = []
        wanted = min(self.neighbours, count)
        while len(picked) < wanted:
            if low >= 0 and (high >= count or
                             prompt_tokens - self._prompts[low] <= self._prompts[high] - prompt_tokens):
                picked.append(self._lengths[low])
                low -= 1
            elif high < count:
                picked.append(self._lengths[high])
                high += 1
            else:
                break
        return sum(picked) / len(picked)


class PredictionLedger:
    """Harness-side scoring of a Track O estimator against the manifest.

    Deliberately separate from the estimator: the harness may read the true decode length
    to report estimation error, the policy may not. Nothing in this object is reachable
    from the Dispatcher or the estimator.
    """

    def __init__(self):
        self._rows = []

    def record(self, request_id, prompt_tokens, predicted, actual):
        self._rows.append({'request_id': request_id, 'p': prompt_tokens,
                           'predicted_d': predicted, 'actual_d': actual})

    def summary(self):
        scored = [row for row in self._rows if row['predicted_d'] is not None]
        if not scored:
            return {'scored_requests': 0, 'unscored_requests': len(self._rows)}
        errors = [row['predicted_d'] - row['actual_d'] for row in scored]
        absolute = [abs(error) for error in errors]
        actual = [row['actual_d'] for row in scored]
        predicted = [row['predicted_d'] for row in scored]
        return {'scored_requests': len(scored),
                'unscored_requests': len(self._rows) - len(scored),
                'mean_error': statistics.fmean(errors),
                'mean_absolute_error': statistics.fmean(absolute),
                'median_absolute_error': statistics.median(absolute),
                'mean_actual_d': statistics.fmean(actual),
                'prediction_spread': statistics.pstdev(predicted) if len(predicted) > 1 else 0.0,
                'spearman_predicted_vs_actual': _spearman(predicted, actual)}


def _spearman(left, right):
    """Rank correlation; 0.0 when either side is constant, which a flat estimator will be."""
    count = len(left)
    if count < 2:
        return None

    def ranks(values):
        order = sorted(range(count), key=lambda i: values[i])
        out = [0.0] * count
        index = 0
        while index < count:
            stop = index
            while stop + 1 < count and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            shared = (index + stop) / 2.0
            for position in range(index, stop + 1):
                out[order[position]] = shared
            index = stop + 1
        return out

    left_ranks, right_ranks = ranks(left), ranks(right)
    left_mean = sum(left_ranks) / count
    right_mean = sum(right_ranks) / count
    numerator = sum((left_ranks[i] - left_mean) * (right_ranks[i] - right_mean) for i in range(count))
    left_var = sum((value - left_mean) ** 2 for value in left_ranks)
    right_var = sum((value - right_mean) ** 2 for value in right_ranks)
    if left_var == 0.0 or right_var == 0.0:
        return 0.0
    return numerator / math.sqrt(left_var * right_var)


class Dispatcher:
    """Undispatched backlog in policy order.

    Static sort keys are applied once. `online_estimate` is re-sorted as completions teach
    the estimator, because an online policy has no ranking signal until something finishes;
    re-sorting is throttled so it cannot dominate the dispatch loop.
    """

    def __init__(self, views, order, sort_key, estimator=None, resort_every=32, service_fn=None):
        if sort_key not in POLICY_SORT_KEYS:
            raise ValueError(f'Unknown sort key {sort_key!r}')
        decode_keyed = order in ('short_decode_first', 'long_decode_first',
                                 'short_decode_then_short_prompt')
        if decode_keyed and sort_key is None:
            raise ValueError(f'Order {order} needs an explicit sort_key')
        if sort_key is not None and not decode_keyed:
            raise ValueError(f'sort_key {sort_key!r} is meaningless for order {order}')
        if sort_key == 'online_estimate' and estimator is None:
            raise ValueError('An online sort key needs an estimator')
        self.order = order
        self.sort_key = sort_key
        self.estimator = estimator
        self.service_fn = service_fn
        self.resort_every = resort_every
        self._views = list(views)
        self._completions_since_sort = 0
        self.resorts = 0
        self.ranking_signal_available = sort_key in ('true_d', 'prompt')
        if sort_key == 'attained_service' and service_fn is None:
            raise ValueError('attained_service needs an observed-generated-tokens accessor')
        self._sort()

    def _sort(self):
        if self.sort_key == 'true_d':
            for view in self._views:
                if 'd' not in view:
                    raise ValueError('sort_key true_d requires the oracle field; Track O cannot use it')
                view['d_sort'] = view['d']
        elif self.sort_key == 'prompt':
            for view in self._views:
                view['d_sort'] = view['p']
        elif self.sort_key == 'attained_service':
            # Observed generated tokens. Over the *waiting* queue this is zero for every
            # request, so a stable sort leaves arrival order untouched: the key is degenerate
            # as a dispatch rule and is included to demonstrate that, not to compete.
            for view in self._views:
                view['d_sort'] = self.service_fn(view['index']) if self.service_fn else 0
            self.ranking_signal_available = len({v['d_sort'] for v in self._views}) > 1
        elif self.sort_key == 'online_estimate':
            informed = False
            for view in self._views:
                estimate = self.estimator.estimate(view['p'])
                if estimate is None:
                    view['d_sort'] = 0.0
                else:
                    view['d_sort'] = estimate
                    informed = True
            self.ranking_signal_available = informed
        self._views = order_workload(self._views, self.order)
        self.resorts += 1
        self._completions_since_sort = 0

    def note_completion(self):
        """A completion taught the estimator; re-rank what is left of the backlog.

        The first completion is where the estimator goes from no signal to some signal, so it
        always forces a re-sort; afterwards re-sorting is throttled so it cannot dominate the
        dispatch loop.
        """
        if self.sort_key != 'online_estimate':
            return
        self._completions_since_sort += 1
        first_signal = self.estimator.sample_count == 1
        if self._views and (first_signal or self._completions_since_sort >= self.resort_every):
            self._sort()

    def peek(self):
        return self._views[0] if self._views else None

    def pop(self):
        return self._views.pop(0)

    def __len__(self):
        return len(self._views)


def measure_policy(llm, case, repeats, relay=None):
    """Run one fixed backlog under an explicit intra-device admission policy.

    The engine keeps its own scheduler; the policy controls dispatch order and how many
    requests are allowed to be resident. Preemption stays under vLLM control and is
    recorded, so memory pressure and eviction are measured rather than assumed.
    """
    from vllm import SamplingParams
    bind_prompt_catalogues(llm)
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    pool = scheduler.kv_cache_manager.block_pool
    block_size = engine.vllm_config.cache_config.block_size
    max_model_len = engine.vllm_config.model_config.max_model_len
    policy = case['policy']
    order = policy.get('order', 'arrival')
    cap = policy.get('cap')
    admit = policy.get('admit', 'greedy')
    horizon = policy.get('horizon', 'true_d')
    track = policy.get('track', 'C')
    sort_key = policy.get('sort_key')
    window = policy.get('window', 64)
    margin = policy.get('margin', 0)
    source = case.get('source', 'synthetic')
    classes = case.get('classes')
    draw = case.get('draw')
    planned = admit == 'planner'
    planner_order_only = order == 'planned'
    needs_plan = planned or planner_order_only
    if planner_order_only and (admit != 'greedy' or sort_key is not None
                               or 'certificate' in policy
                               or policy.get('service_discipline', 'native') != 'native'):
        raise ValueError('Planned order uses native greedy admission, no sort key or release certificate')
    eviction_mode = policy.get('eviction')
    native_admission = policy.get('native_admission')
    if native_admission not in (None, 'baseline', 'shadow', 'adaptive'):
        raise ValueError('Unknown native admission mode')
    if native_admission and (needs_plan or eviction_mode or admit != 'greedy'
            or order != 'arrival' or sort_key is not None
            or cap not in ((128, 256) if native_admission == 'baseline' else (256,))
            or policy.get('service_discipline', 'native') != 'native'
            or policy.get('throttle_prefills') or source != 'drawn'):
        raise ValueError('Native admission requires arrival cap256, drawn prompts and native execution')
    if eviction_mode is not None and (needs_plan or source != 'drawn'):
        raise ValueError('Eviction arms require a declared law and a non-planner admission policy')
    certificate = policy.get('certificate', 'ready_epoch_certificate')
    synchronous = planned or eviction_mode is not None or policy.get('service_discipline') == 'sync_ready'
    force_throttle = bool(policy.get('throttle_prefills', False))
    if needs_plan:
        if source != 'drawn' or not case.get('trace_iterations', True):
            raise ValueError('Planner arms require a declared workload law and per-step tracing')
        cap = (64 if planner_order_only else 128) if cap is None else cap
    if order not in POLICY_ORDERS or admit not in POLICY_ADMIT or horizon not in ('true_d', 'max_d'):
        raise ValueError('Unknown policy specification: ' + json.dumps(policy, sort_keys=True))
    if track not in POLICY_TRACKS:
        raise ValueError(f'Unknown policy track {track!r}')
    if source not in ('synthetic', 'sharegpt', 'drawn'):
        raise ValueError('Unknown workload source: ' + str(source))
    if cap is not None and cap < 1:
        raise ValueError('Policy residency cap must be a positive request count')
    if window < 1 or margin < 0:
        raise ValueError('Lookahead window must be positive and the block margin non-negative')
    if sort_key == 'online_estimate' and admit == 'greedy' and cap is None:
        raise ValueError('An online estimator cannot inform dispatch under unthrottled greedy admission: '
                         'the whole backlog is submitted before anything completes, so every estimate is '
                         'unset and the arm silently degenerates to arrival order. Throttle it -- either '
                         'admit=watermark, or a residency cap, which the dispatch loop enforces BEFORE '
                         'the admission gate -- and give its Track C twin the same throttle.')
    if track == 'O' and (sort_key == 'true_d' or admit == 'trajectory' or
                         (admit == 'memory_safe' and horizon == 'true_d')):
        raise ValueError('Track O may not use the true decode length; '
                         'sort_key=true_d, admit=trajectory and memory_safe/true_d are Track C only')
    if source == 'drawn':
        if classes is not None or not isinstance(draw, dict):
            raise ValueError('A drawn workload is specified by `draw`, not by `classes`')
    else:
        if draw is not None:
            raise ValueError('`draw` applies only to source=drawn')
        if repeats < 1 or any(spec['n'] < 1 or spec['d'] < 1 for spec in classes):
            raise ValueError('Invalid policy workload dimensions')
        if source == 'sharegpt' and any('p' in spec for spec in classes):
            raise ValueError('A trace workload carries its own prompt lengths; do not specify p')
        if source == 'synthetic':
            if any(spec['p'] < 1 for spec in classes):
                raise ValueError('Invalid policy workload dimensions')
            if any(spec['p'] + spec['d'] > max_model_len for spec in classes):
                raise ValueError('Policy workload exceeds the engine context limit; no implicit clipping is permitted')
    if repeats < 1:
        raise ValueError('Invalid repeat count')
    model_config = json.loads((pathlib.Path(MODEL) / 'config.json').read_text())
    vocab = model_config.get('text_config', model_config)['vocab_size']
    usable_blocks = pool.num_gpu_blocks - 1
    records = []

    def one_run(repeat, warmup):
        startup_start = time.perf_counter()
        if engine.has_unfinished_requests() or scheduler.get_num_unfinished_requests() or core.batch_queue:
            raise RuntimeError('Previous workload has not drained')
        # Repeats redraw the manifest by default, which samples workload variability but means
        # two repeats are different workloads: comparing raw makespans across repeats mixes
        # timing noise with token-count differences. Policy comparisons must therefore be paired
        # within a repeat. Set manifest_per_repeat=False to hold the manifest fixed and isolate
        # hardware timing noise.
        run_seed = case.get('seed', 42) + (repeat * 1009 if case.get('manifest_per_repeat', True) else 0)
        if source == 'drawn':
            workload, draw_summary = build_drawn_workload(draw, run_seed, max_model_len)
        elif source == 'sharegpt':
            workload, draw_summary = build_trace_workload(classes, run_seed, max_model_len), None
        else:
            workload, draw_summary = build_policy_workload(classes, vocab, run_seed), None
        if 'saved_manifest' in case:
            actual_manifest = [[r['p'], r['d'], r['arrival_index']] for r in workload]
            if actual_manifest != case['saved_manifest']:
                raise ValueError('Reconstructed ShareGPT requests differ from the saved ordered manifest')
            if ['r%d' % r['index'] for r in workload] != case['saved_request_ids']:
                raise ValueError('Saved request identity/order mismatch')
        if case.get('shadow_request_limit'):
            if native_admission != 'shadow':
                raise ValueError('Only a non-intervening shadow run may truncate the saved backlog')
            workload = workload[:int(case['shadow_request_limit'])]
        manifest = {'r%d' % request['index']: request for request in workload}
        worst_case_d = max(request['d'] for request in workload)
        future_model = gate = planned_result = discipline = None
        eviction_controller = eviction_guard = eviction_scorer = None
        admission_controller = None
        planning_wall_s = 0.0
        public_requests = {request['index']: {key: request[key]
                           for key in ('index', 'p', 'arrival_index')} for request in workload}
        indices = {'r%d' % index: index for index in public_requests}
        if needs_plan or eviction_mode is not None or native_admission:
            planning_start = time.perf_counter()
            views = [dict(public_requests[request['index']]) for request in workload]
            if track == 'C':
                for view, request in zip(views, workload):
                    view['d'] = request['d']
            future_model = kv_future_profile.FutureModel(
                track, views, {**draw, 'n': len(workload)}, max_model_len, block_size)
            planning_wall_s = time.perf_counter() - planning_start
        if eviction_mode is not None:
            eviction_scorer = kv_eviction.F3Scorer(future_model, startup_elapsed_s=planning_wall_s)
            planning_wall_s = time.perf_counter() - planning_start
        if needs_plan:
            representative = future_model.planner_manifest(policy.get('planning_quantile', 0.5))
            budget_s = policy.get('planning_budget_s', 55 if len(workload) <= 1763 else 110)
            planned_result = kv_planner.plan(
                representative, usable_blocks, block_size, cap,
                search=policy.get('search', True),
                budget_s=max(0.0, budget_s - (time.perf_counter() - planning_start)),
                seed=policy.get('search_seed', 42))
            if planned_result is None:
                raise RuntimeError('No individually feasible planner manifest')
            json.dumps(planned_result)
            planning_wall_s = time.perf_counter() - planning_start
            if planned:
                gate = kv_execution.PlannerGate(future_model, kv_future_profile.certified_peak,
                    usable_blocks, certificate, len(workload), policy.get('risk', 0.01))

        def reserve_blocks(request):
            if future_model is not None:
                return future_model.future_profile(request['index']).terminal_blocks
            decode_horizon = request['d'] if horizon == 'true_d' else worst_case_d
            return math.ceil((request['p'] + decode_horizon) / block_size)

        estimator = OnlineLengthEstimator() if sort_key == 'online_estimate' else None
        ledger = PredictionLedger() if estimator is not None else None
        iterations, pending = [], collections.deque()
        trace_iterations = bool(case.get('trace_iterations', True))
        schedule_calls = 0
        # vLLM's InputProcessor.assign_request_id appends a random suffix, so the engine's
        # internal id is not the id we submit. add_request returns the internal id; the
        # scheduler and the schedule/update hooks speak internal, while RequestOutput from
        # engine.step() speaks external. Keep both directions explicitly: looking a resident
        # up by the wrong one returns None and silently corrupts every live gate.
        internal_by_external, external_by_internal = {}, {}
        lifecycles, admitted_reserve, expected_outputs = {}, {}, {}

        def observed_internal_generated(internal):
            life = lifecycles.get(internal)
            return life['output_tokens'] if life is not None else 0

        def observed_generated(index):
            """Tokens this request has been observed to emit; zero before it is admitted.

            Must be bound after the id maps and lifecycle table exist: Dispatcher sorts during
            construction, so this is called immediately.
            """
            internal = internal_by_external.get('r%d' % index)
            return observed_internal_generated(internal)

        if needs_plan:
            dispatcher = kv_execution.PlannedDispatcher(
                [policy_view(request, track) for request in workload],
                planned_result, representative, gate)
        else:
            dispatcher = Dispatcher([policy_view(request, track) for request in workload],
                                    order, sort_key, estimator, service_fn=observed_generated)
        original_schedule = scheduler.schedule
        base_schedule = original_schedule
        original_update = scheduler.update_from_output
        emitted_tokens = preemption_events = 0
        run_start = None

        def schedule(*args, **kwargs):
            nonlocal schedule_calls
            schedule_calls += 1
            t0 = time.perf_counter()
            if discipline is not None:
                discipline.before_schedule()
            if force_throttle:
                if args:
                    args = (True, *args[1:])
                else:
                    kwargs['throttle_prefills'] = True
            result = original_schedule(*args, **kwargs)
            if admission_controller is not None:
                admission_controller.issued(result)
            if discipline is not None:
                discipline.after_schedule(result)
            for request in result.scheduled_new_reqs:
                expected_outputs[request.req_id] = scheduler.requests[request.req_id].max_tokens
            if not trace_iterations:
                pending.append((None, t0, result.num_scheduled_tokens))
                return result
            preempted = getattr(result, 'preempted_req_ids', None)
            per_request = result.num_scheduled_tokens
            # Heuristic, NOT a phase split: a final prefill/recompute chunk of exactly one token
            # is indistinguishable here from a decode step. The authoritative decode count is
            # runner.input_batch.request_distribution[0], captured via capture_worker_shapes.
            single_tok = sum(1 for n in per_request.values() if n == 1)
            row = {'iteration': len(iterations), 'scheduled_at_s': t0 - run_start,
                   'scheduled_tokens': result.total_num_scheduled_tokens,
                   'scheduled_requests': len(per_request),
                   'single_token_requests': single_tok,
                   'multi_token_requests': len(per_request) - single_tok,
                   'multi_token_tokens': sum(n for n in per_request.values() if n > 1),
                   'running_requests': len(scheduler.running),
                   'waiting_requests': len(scheduler.waiting),
                   'admitted_not_finished': len(admitted_reserve),
                   'undispatched_requests': len(dispatcher),
                   'free_kv_blocks_after_schedule': pool.get_num_free_blocks(),
                   'total_kv_blocks': pool.num_gpu_blocks,
                   'preempted_request_ids': list(preempted) if preempted is not None else None}
            if gate is not None:
                row['live_calendar_next_bound_blocks'] = (
                    gate.next_allocation_bound + gate.pending_added)
                row['live_calendar_peak_bound_blocks'] = gate.base_peak + gate.pending_added
                row['calendar_signed_error_blocks'] = (pool.num_gpu_blocks - 1
                    - pool.get_num_free_blocks() - row['live_calendar_next_bound_blocks'])
                row['calendar_exceeds_safe_peak'] = (pool.num_gpu_blocks - 1
                    - pool.get_num_free_blocks() > row['live_calendar_peak_bound_blocks'])
            iterations.append(row)
            pending.append((row, t0, result.num_scheduled_tokens))
            return result

        def update(scheduled, model_output, *args, **kwargs):
            nonlocal emitted_tokens, preemption_events
            result = original_update(scheduled, model_output, *args, **kwargs)
            now = time.perf_counter()
            row, t0, expected_tokens = pending.popleft()
            if scheduled.num_scheduled_tokens != expected_tokens:
                raise RuntimeError('Scheduled and completed iteration identities differ')
            new_tokens, finished = 0, []
            for batch in result.values():
                for output in batch.outputs:
                    new_tokens += len(output.new_token_ids)
                    if output.new_token_ids:
                        life = lifecycles.setdefault(output.request_id, {
                            'request_id': output.request_id, 'first_output_s': now - run_start,
                            'last_output_s': now - run_start, 'output_tokens': 0})
                        life['last_output_s'] = now - run_start
                        life['output_tokens'] += len(output.new_token_ids)
                    if output.finish_reason is not None:
                        if future_model is not None:
                            external = external_by_internal[output.request_id]
                            future_model.observe_completion(indices[external],
                                lifecycles[output.request_id]['output_tokens'])
                        finished.append(output.request_id)
                        # Teach the estimator here, where the observation is actually made.
                        # Driving it from the step loop instead depends on the lifecycle map
                        # being populated under the same identity the step output carries,
                        # which is not guaranteed for every completion.
                        if estimator is not None:
                            external = external_by_internal.get(output.request_id)
                            observed = lifecycles.get(output.request_id, {}).get('output_tokens')
                            if external is None or not observed:
                                raise RuntimeError(
                                    f'Finished {output.request_id} has no mapped identity or no observed '
                                    f'output (external={external}, observed={observed}); the estimator '
                                    'may not fall back to the manifest')
                            estimator.observe(manifest[external]['p'], observed)
                            dispatcher.note_completion()
            if admission_controller is not None:
                admission_controller.retired(scheduled)
            emitted_tokens += new_tokens
            preemption_events += len(getattr(scheduled, 'preempted_req_ids', None) or ())
            if row is not None:
                row.update(completed_at_s=now - run_start, scheduled_to_output_s=now - t0,
                           emitted_output_tokens=new_tokens, finished_request_ids=finished)
            return result

        scheduler.schedule, scheduler.update_from_output = schedule, update
        admission_log = []
        deferrals = collections.Counter()
        computed_field = None

        def live_computed_tokens(request_id):
            """Tokens the engine has actually computed for a resident request.

            Read from the running scheduler, not inferred from elapsed steps: chunked
            prefill, token-budget skips and preemption all break the one-token-per-step
            assumption. The resolved field name is recorded in the run row.
            """
            nonlocal computed_field
            request = scheduler.requests.get(internal_by_external.get(request_id, request_id))
            if request is None:
                return None
            if computed_field is None:
                for name in ('num_computed_tokens', 'num_tokens_computed', 'computed_tokens'):
                    if hasattr(request, name):
                        computed_field = name
                        break
                else:
                    raise RuntimeError('Cannot read computed-token state from this vLLM build; '
                                       f'request exposes {sorted(vars(request))}')
            return getattr(request, computed_field)

        def resident_context(request_id, entry):
            """Context this request must hold, and what the engine has computed of it.

            `num_computed_tokens` is reset by preemption while the emitted tokens survive, so
            the context a preempted request will re-acquire is its prompt plus everything it has
            already generated — not just its prompt. Generated count comes from engine-reported
            output events (`lifecycles`), which preemption does not roll back.
            """
            internal_id = internal_by_external.get(request_id, request_id)
            observed_generated = lifecycles.get(internal_id, {}).get('output_tokens', 0)
            required = entry['p'] + observed_generated
            computed = live_computed_tokens(request_id)
            if computed is None:
                computed = 0
            return required, computed, observed_generated

        def projected_blocks_at_horizon(candidate):
            """P-A: block demand W steps out, from live state, crediting no unobserved release.

            Every admitted request is charged the whole context it must hold — prompt plus
            observed generated tokens — even when the engine reports zero computed tokens,
            which is the state of a freshly admitted or freshly preempted request. Growth on
            top of that is bounded by one token per step over the window. Nothing is released
            until the engine reports the request finished.
            """
            total = 0
            for request_id in admitted_reserve:
                entry = manifest[request_id]
                required, computed, generated = resident_context(request_id, entry)
                context = max(required, computed)
                remaining = max(entry['d'] - generated, 0)
                total += math.ceil((context + min(window, remaining)) / block_size)
            candidate_entry = manifest['r%d' % candidate['index']]
            total += math.ceil((candidate_entry['p'] + min(window, candidate_entry['d'])) / block_size)
            return total

        def pending_prefill_blocks():
            """Blocks owed to admitted requests the engine has not finished (re)computing.

            `add_request` does not allocate; the scheduler allocates during `schedule()`. So the
            free-block count is stale for every admission decided between two engine steps, and a
            gate reading it raw would admit the whole backlog in one inner loop and silently
            degenerate to greedy. A preempted request owes its prompt *and* its regenerated
            context, so the shortfall is measured against prompt-plus-generated, not prompt.
            """
            total = 0
            for request_id in admitted_reserve:
                entry = manifest[request_id]
                required, computed, _ = resident_context(request_id, entry)
                shortfall = math.ceil(required / block_size) - math.ceil(computed / block_size)
                if shortfall > 0:
                    total += shortfall
            return total

        def admission_allows(candidate):
            """Gate the next dispatch. Returns (allowed, reason)."""
            if gate is not None:
                return gate.allows(candidate['index']), 'planner_certificate'
            if admit == 'greedy':
                return True, None
            if admit == 'memory_safe':
                needed = reserve_blocks(manifest['r%d' % candidate['index']])
                if sum(admitted_reserve.values()) + needed > usable_blocks:
                    return False, 'reservation'
                return True, None
            needed = math.ceil(candidate['p'] / block_size)
            effective_free = pool.get_num_free_blocks() - pending_prefill_blocks()
            if admit == 'watermark':
                if effective_free < needed + margin:
                    return False, 'watermark'
                return True, None
            if effective_free < needed + margin:
                return False, 'live_check'
            if projected_blocks_at_horizon(candidate) > usable_blocks - margin:
                return False, 'projection'
            return True, None

        emit_progress(relay, 'AUDIT_POLICY_PHASE', policy=policy.get('name', order), repeat=repeat,
             phase='workload_built', requests=len(workload),
             decode_tokens=sum(r['d'] for r in workload), prompt_tokens=sum(r['p'] for r in workload))
        # Authoritative per-step shapes from the runner: real padding buckets, the true decode
        # count (request_distribution[0]), and context sums. Phase cannot be recovered from
        # scheduler aggregates because chunked prefill fuses prefill and decode into one pass.
        worker_shapes = []
        profile_capture = None
        profile_steps = int(case.get('profile_steps', 0))
        trace_shapes = bool(case.get('trace_shapes'))
        if trace_shapes:
            engine.collective_rpc(capture_worker_shapes, args=(True,))
        try:
            if native_admission:
                scheduler.schedule = base_schedule
                admission_controller = kv_admission.AdmissionController(
                    llm, future_model,
                    lambda request: indices[external_by_internal[request.request_id]],
                    observed_internal_generated, shadow=native_admission == 'shadow',
                    baseline=native_admission == 'baseline',
                    online_limit_s=policy.get('online_cpu_limit_s', 3.408),
                    startup_elapsed_s=planning_wall_s)
                admission_controller.install()
                original_schedule = scheduler.schedule
                scheduler.schedule = schedule
            if synchronous:
                discipline = kv_execution.ServiceDiscipline(llm, synchronous=True, ready_first=True,
                    strict=planned and track == 'C' and certificate == 'ready_epoch_certificate')
            if eviction_mode is not None:
                install_start = time.perf_counter()
                def index_of(request):
                    return indices[external_by_internal[request.request_id]]
                def eviction_snapshot(request, rank, protected):
                    index = index_of(request)
                    return {'index': index, 'p': request.num_prompt_tokens,
                            'g': observed_internal_generated(request.request_id),
                            'blocks': discipline.allocated_count(request.request_id),
                            'rank': rank, 'protected': protected}
                endpoints = {index: math.ceil((request['p'] + min(
                    draw['d_max'], max_model_len - request['p'])) / block_size)
                    for index, request in public_requests.items()}
                eviction_guard = kv_eviction.GuardAdapter(scheduler, index_of=index_of,
                    endpoint_blocks=endpoints, capacity=usable_blocks,
                    other_blocks=lambda _: discipline.unattributed_blocks())
                # The independent native seam must be underneath the common trace wrapper.
                scheduler.schedule = base_schedule
                eviction_controller = kv_eviction.EvictionController(scheduler, mode=eviction_mode,
                    guard=eviction_guard, scorer=eviction_scorer, snapshot=eviction_snapshot,
                    block_deficit=lambda _, request, tokens: (
                        discipline.new_blocks_needed(request, tokens) - pool.get_num_free_blocks()),
                    fresh_backlog=lambda: bool(len(dispatcher)),
                    reorder_audit=discipline.homogeneous_audit)
                eviction_controller.install()
                original_schedule = scheduler.schedule
                scheduler.schedule = schedule
                planning_wall_s += time.perf_counter() - install_start
                if planning_wall_s > 2.0:
                    raise RuntimeError('Eviction setup exceeded its two-second startup budget')
            execution_state = {'engine_step_method': core.step_fn.__name__,
                               'defer_block_free': scheduler.defer_block_free,
                               'planner_order_only': planner_order_only,
                               'certificate': certificate if planned else None}
            prompt_payload_sha256 = hashlib.sha256(json.dumps(
                [[r['index'], r['arrival_index'], r['prompt_id'], r['prompt_token_ids']]
                 for r in workload], separators=(',', ':')).encode()).hexdigest() if source == 'drawn' else None
            run_start = time.perf_counter()
            startup_wall_s = run_start - startup_start
            run_start_wall = time.time()
            dispatched = 0
            first_pass = True
            while (len(dispatcher) or admitted_reserve
                   or eviction_guard is not None and eviction_guard.has_pending()):
                if eviction_guard is not None:
                    # Advance completed packs before deciding whether external arrivals are legal.
                    eviction_guard.before_schedule()
                    if not len(dispatcher) and not admitted_reserve and not eviction_guard.has_pending():
                        break
                if gate is not None:
                    snapshots, other_blocks = discipline.snapshot(admitted_reserve,
                        internal_by_external, indices, public_requests, observed_generated)
                    gate.refresh(snapshots, other_blocks)
                while len(dispatcher):
                    if eviction_guard is not None and not eviction_guard.allow_arrivals:
                        deferrals['eviction_guard'] += 1
                        break
                    candidate = dispatcher.peek()
                    if cap is not None and len(admitted_reserve) >= cap:
                        deferrals['cap'] += 1
                        break
                    allowed, reason = admission_allows(candidate)
                    if not allowed:
                        deferrals[reason] += 1
                        break
                    entry = manifest['r%d' % candidate['index']]
                    needed = reserve_blocks(entry)
                    if needed > usable_blocks:
                        raise RuntimeError('A single request cannot fit the KV pool; workload is infeasible')
                    dispatcher.pop()
                    request_id = 'r%d' % candidate['index']
                    admitted_reserve[request_id] = needed
                    prediction = estimator.estimate(candidate['p']) if estimator is not None else None
                    if ledger is not None:
                        ledger.record(request_id, candidate['p'], prediction, entry['d'])
                    internal_id = engine.add_request(
                        request_id, {'prompt_token_ids': entry['prompt_token_ids']},
                        SamplingParams(max_tokens=entry['d'], temperature=0.0,
                                       ignore_eos=True, detokenize=False))
                    if internal_id is None:
                        raise RuntimeError('add_request did not return an internal request id; '
                                           'the external/internal mapping cannot be established')
                    internal_by_external[request_id] = internal_id
                    external_by_internal[internal_id] = request_id
                    if gate is not None:
                        gate.admitted(candidate['index'])
                    dispatched += 1
                    admission_log.append({'request_id': request_id, 'at_s': time.perf_counter() - run_start,
                                          'p': entry['p'], 'd': entry['d'],
                                          'predicted_d': prediction, 'rank_key': candidate.get('d_sort'),
                                          'resident_after': len(admitted_reserve)})
                    if needs_plan:
                        admission_log[-1]['planned_rank'] = dispatcher.ranks[candidate['index']]
                        admission_log[-1]['preferred_service_epoch'] = planned_result['starts'][
                            candidate['arrival_index']]
                if first_pass:
                    emit_progress(relay, 'AUDIT_POLICY_PHASE', policy=policy.get('name', order), repeat=repeat,
                         phase='first_dispatch_done', dispatched=dispatched,
                         undispatched=len(dispatcher),
                         elapsed_s=time.perf_counter() - run_start)
                if not engine.has_unfinished_requests():
                    raise RuntimeError('Policy admitted no work but the backlog is not drained')
                if (profile_steps and profile_capture is None
                        and schedule_calls >= case.get('profile_start_iteration', 400)):
                    logdir = case.get('profile_logdir', '/tmp/omp_jax_trace_policy')
                    profile_capture = {'logdir': logdir, 'start_iteration': schedule_calls,
                        'steps_requested': profile_steps,
                        'start_rpc': engine.collective_rpc(_trace_steps, args=(logdir, profile_steps))}
                for output in engine.step():
                    if output.finished:
                        admitted_reserve.pop(output.request_id, None)
                if (profile_capture is not None and 'stop_rpc' not in profile_capture
                        and schedule_calls - profile_capture['start_iteration'] >= profile_steps):
                    profile_capture['stop_rpc'] = engine.collective_rpc(_trace_stop)
                    profile_capture['captured_iterations'] = (
                        schedule_calls - profile_capture['start_iteration'])
                if first_pass:
                    emit_progress(relay, 'AUDIT_POLICY_PHASE', policy=policy.get('name', order), repeat=repeat,
                         phase='first_step_done', elapsed_s=time.perf_counter() - run_start,
                         iterations=schedule_calls)
                    first_pass = False
                if schedule_calls % 25 == 0:
                    emit_progress(relay, 'AUDIT_POLICY_PROGRESS', policy=policy.get('name', order), repeat=repeat,
                         iteration=schedule_calls, resident=len(admitted_reserve),
                         undispatched=len(dispatcher), emitted_tokens=emitted_tokens,
                         preemptions=preemption_events, free_kv_blocks=pool.get_num_free_blocks(),
                         elapsed_s=time.perf_counter() - run_start)
            while core.batch_queue or scheduler.has_requests():
                engine.step()
            wall = time.perf_counter() - run_start
        finally:
            try:
                if profile_capture is not None and 'stop_rpc' not in profile_capture:
                    profile_capture['stop_rpc'] = engine.collective_rpc(_trace_stop)
                    profile_capture['captured_iterations'] = (
                        schedule_calls - profile_capture['start_iteration'])
            finally:
                try:
                    if eviction_controller is not None:
                        eviction_controller.close()
                    if admission_controller is not None:
                        admission_controller.close()
                finally:
                    scheduler.schedule, scheduler.update_from_output = base_schedule, original_update
                    if discipline is not None:
                        discipline.restore()
                    if trace_shapes:
                        worker_shapes = [s for s in engine.collective_rpc(capture_worker_shapes, args=(False,))
                                         if s is not None]
        if engine.has_unfinished_requests() or scheduler.get_num_unfinished_requests():
            raise RuntimeError('Policy run returned with unfinished requests')
        expected_tokens = sum(request['d'] for request in workload)
        if dispatched != len(workload) or emitted_tokens != expected_tokens:
            raise RuntimeError(f'Policy delivered {emitted_tokens} tokens over {dispatched} requests; '
                               f'expected {expected_tokens} over {len(workload)}')
        if sorted(lifecycles) != sorted(expected_outputs) or len(expected_outputs) != len(workload):
            raise RuntimeError(f'Engine observed {len(expected_outputs)} admitted and {len(lifecycles)} emitting '
                               f'requests for a {len(workload)} request backlog')
        if any(life['output_tokens'] != expected_outputs[request_id] for request_id, life in lifecycles.items()):
            raise RuntimeError('Per-request output events do not reconcile with the completed workload')
        previous = None
        for completed in iterations:
            if 'completed_at_s' not in completed:
                continue
            if previous is not None:
                completed['completion_interval_s'] = completed['completed_at_s'] - previous['completed_at_s']
            previous = completed
        executed = [r for r in iterations if r['scheduled_tokens'] > 0]
        residency = [r['scheduled_requests'] for r in executed]
        row = {'repeat': repeat, 'warmup': warmup, 'policy': policy, 'wall_s': wall,
               'startup_wall_s': startup_wall_s,
               'startup_plus_serving_s': startup_wall_s + wall,
               'native_admission': admission_controller.summary() if admission_controller else None,
               'prompt_payload_sha256': prompt_payload_sha256,
               'run_start_wall_s': run_start_wall,
               'output_tokens': emitted_tokens, 'requests': len(workload),
               'output_tok_s': emitted_tokens / wall,
               'preemption_events': preemption_events,
               'engine_iterations': schedule_calls,
               'executed_iterations': len(executed) if trace_iterations else None,
               'mean_scheduled_requests': statistics.mean(residency) if residency else None,
               'median_scheduled_requests': statistics.median(residency) if residency else None,
               'max_scheduled_requests': max(residency) if residency else None,
               'mean_free_kv_blocks': statistics.mean(r['free_kv_blocks_after_schedule'] for r in executed) if executed else None,
               'min_free_kv_blocks': min((r['free_kv_blocks_after_schedule'] for r in executed), default=None),
               'total_kv_blocks': pool.num_gpu_blocks,
               'free_kv_blocks_after_completion': pool.get_num_free_blocks(),
               'workload_sha256': hashlib.sha256(json.dumps(
                   [[r['p'], r['d'], r['arrival_index']] for r in workload], separators=(',', ':')).encode()).hexdigest(),
               'track': track, 'sort_key': sort_key,
               'admission_window_steps': window if admit == 'trajectory' else None,
               'admission_margin_blocks': margin if admit in ('watermark', 'trajectory') else None,
               'computed_token_field': computed_field,
               'dispatch_deferrals': dict(deferrals),
               'dispatcher_resorts': dispatcher.resorts,
               'ranking_signal_available': dispatcher.ranking_signal_available,
               'draw_summary': draw_summary,
               'realised_horizons': summarize_horizons(workload),
               'manifest': [[r['p'], r['d'], r['arrival_index']] for r in workload],
               'manifest_request_ids': ['r%d' % r['index'] for r in workload],
               'trace_iterations': trace_iterations,
               'planner': planned_result, 'planning_wall_s': planning_wall_s,
               'planner_manifest': representative if needs_plan else None,
               'max_model_len': max_model_len,
               'planner_module_hashes': POLICY_MODULE_SHA256,
               'eviction': eviction_controller.summary() if eviction_controller is not None else None,
               'service_audit': discipline.audit if discipline is not None else None,
               'execution_state': execution_state,
               'gate_stats': gate.stats if gate is not None else None,
               'prompt_catalogue_sha256': (llm._audit_prompt_catalogues[
                   (draw or {}).get('catalogue', 'small')]['sha256']
                   if source in ('drawn', 'sharegpt') else None),
               'future_profile_model': ({'public_law': dict(future_model.law),
                   'cohort_indices': list(future_model.cohort_indices),
                   'completed_cohort_sample': future_model.calibration_sample(),
                   'planning_quantile': policy.get('planning_quantile', 0.5),
                   'law_refitted_during_run': False} if future_model is not None else None),
               'kv_block_size': block_size,
               'calendar_comparison': ('live residual calendar; native post-schedule/pre-sampling '
                   'p+g allocation; pending prefills use endpoint bounds; null block excluded'
                   if planned else None),
               'estimation_error': ledger.summary() if ledger is not None else None,
               'worker_shapes': worker_shapes,
               'profile_capture': profile_capture,
               'profile_overhead_included_in_wall': profile_capture is not None,
               'admissions': admission_log, 'iterations': iterations,
               'request_lifecycles': list(lifecycles.values()),
               'timing_semantics': 'policy dispatch start through drained backlog; makespan of a fixed request set'}
        if not warmup:
            records.append(row)
        emit_progress(relay, 'AUDIT_POLICY_RUN', policy=policy.get('name', order), repeat=repeat, warmup=warmup,
             wall_s=wall, output_tok_s=row['output_tok_s'], preemptions=preemption_events,
             mean_resident=row['mean_scheduled_requests'])
        return row

    # A full warmup costs as much as a measured run. The engine is warm after the first arm
    # in a plan, so only the first case needs it; later cases set warmup=false. Shapes are
    # unchanged across arms (same manifest, same padding buckets), so nothing recompiles.
    if case.get('warmup', True):
        one_run(-1, True)
    for repeat in range(repeats):
        one_run(repeat, False)
    return {'case': case, 'repeats': repeats, 'runs': records,
            'usable_kv_blocks': usable_blocks, 'kv_block_size': block_size,
            'median_makespan_s': statistics.median(r['wall_s'] for r in records),
            'median_output_tok_s': statistics.median(r['output_tok_s'] for r in records),
            'total_preemption_events': sum(r['preemption_events'] for r in records)}


def inspect_runtime(llm):
    def inspect_worker(worker):
        if worker.rank != 0:
            return None
        import inspect
        runner = worker.model_runner
        source_path = pathlib.Path(inspect.getsourcefile(type(runner)))
        return {'runner_class': type(runner).__module__ + '.' + type(runner).__name__,
                'runner_source_path': str(source_path), 'runner_source': source_path.read_text(),
                'runner_fields': sorted(vars(runner))}
    return {'worker_inspection': llm.llm_engine.collective_rpc(inspect_worker)}

def probe_scheduler(llm):
    """Facts about the *installed* vLLM scheduler, read from the running process.

    A local source checkout is not evidence: the deployed wheel can be a divergent
    commit. Every scheduler-level mechanism must be justified against what this returns.
    """
    import inspect
    import vllm
    engine = llm.llm_engine
    scheduler = engine.engine_core.engine_core.scheduler
    cls = type(scheduler)
    # Victim selection may live in a base class: the installed build runs AsyncScheduler,
    # a subclass. Searching only the leaf returns an empty list that looks like absence.
    mro = [c for c in cls.__mro__ if c.__module__.startswith('vllm')]
    victim_hits, mro_sources = [], {}
    for c in mro:
        try:
            csrc = inspect.getsource(c)
        except Exception:
            continue
        mro_sources[c.__module__ + '.' + c.__name__] = inspect.getsourcefile(c)
        for i, line in enumerate(csrc.splitlines(), 1):
            s = line.strip()
            if ('self.running.pop(' in s or 'preempted_req' in s or '_preempt' in s
                    or 'PREEMPT' in s):
                victim_hits.append({'class': c.__name__, 'line': i, 'text': s[:160]})
    src = inspect.getsource(cls)
    schedule_src = inspect.getsource(cls.schedule) if hasattr(cls, 'schedule') else ''
    throttle_src = ''
    for c in mro:
        try:
            csrc = inspect.getsource(c)
        except Exception:
            continue
        if 'throttle_prefills' in csrc:
            throttle_src = '\n'.join(s.strip() for s in csrc.splitlines()
                                     if 'throttle_prefills' in s)[:1200]
            break
    request = next(iter(scheduler.requests.values()), None)
    try:
        from vllm.v1.core.sched.request_queue import SchedulingPolicy
        policies = [p.value for p in SchedulingPolicy]
    except Exception as exc:
        policies = f'unavailable: {exc}'
    cache_config = engine.vllm_config.cache_config
    core = engine.engine_core.engine_core
    method_sources = {}
    coordinator = scheduler.kv_cache_manager.coordinator
    method_sources['coordinator.get_blocks'] = inspect.getsource(type(coordinator).get_blocks)
    method_sources['cache.get_blocks'] = inspect.getsource(type(scheduler.kv_cache_manager).get_blocks)
    for owner in (type(core), *mro, type(scheduler.kv_cache_manager)):
        for name in ('schedule', 'step', 'step_with_batch_queue', '_update_after_schedule',
                     '_preempt_request', 'update_from_output', 'allocate_slots', 'free'):
            if name in owner.__dict__:
                method_sources[owner.__name__ + '.' + name] = inspect.getsource(owner.__dict__[name])
    return {
        'vllm_version': getattr(vllm, '__version__', None),
        'scheduler_class': cls.__module__ + '.' + cls.__name__,
        'scheduler_source_path': inspect.getsourcefile(cls),
        'scheduler_methods': sorted(m for m in dir(cls) if not m.startswith('__')),
        'has_update_after_schedule': hasattr(cls, '_update_after_schedule'),
        'schedule_signature': str(inspect.signature(cls.schedule)),
        'schedule_calls_update_after_schedule': '_update_after_schedule' in schedule_src,
        'preemption_victim_lines': victim_hits,
        'mro_sources': mro_sources,
        'throttle_prefills_src': throttle_src,
        'scheduler_mro': [c.__module__ + '.' + c.__name__ for c in mro],
        'scheduling_policies': policies,
        'active_policy': str(getattr(scheduler, 'policy', None)),
        'request_has_priority': hasattr(request, 'priority') if request is not None else None,
        'request_fields': sorted(vars(request)) if request is not None else None,
        'waiting_queue_class': type(scheduler.waiting).__name__,
        'running_is_list': isinstance(scheduler.running, list),
        'cache_config_has_kv_offloading': hasattr(cache_config, 'kv_offloading_size'),
        'max_num_seqs': engine.vllm_config.scheduler_config.max_num_seqs,
        'max_num_batched_tokens': engine.vllm_config.scheduler_config.max_num_batched_tokens,
        'method_sources': method_sources,
        'coordinator_class': type(coordinator).__name__,
        'coordinator_fields': sorted(vars(coordinator)),
        'manager_fields': [sorted(vars(manager)) for manager in
                           getattr(coordinator, 'single_type_managers', ())],
        'manager_classes': [type(manager).__name__ for manager in
                            getattr(coordinator, 'single_type_managers', ())],
        'core_step_fn': str(getattr(core, 'step_fn', None)),
        'max_concurrent_batches': getattr(core, 'max_concurrent_batches', None),
        'batch_queue_capacity': getattr(core.batch_queue, 'maxlen', None),
        'cache_config': str(cache_config),
    }

def _profiler_surface(worker):
    """What profiling machinery the *installed* worker/runner actually exposes."""
    import inspect as _i
    runner = worker.model_runner
    out = {'rank': worker.rank,
           'worker_class': type(worker).__module__ + '.' + type(worker).__name__,
           'worker_mro': [c.__module__ + '.' + c.__name__ for c in type(worker).__mro__][:6],
           'runner_class': type(runner).__module__ + '.' + type(runner).__name__,
           'worker_has_profile': hasattr(worker, 'profile'),
           'runner_profiler_attrs': sorted(a for a in dir(runner) if 'profil' in a.lower()),
           'runner_profiler_values': {}}
    for a in ('phased_profiling_dir', 'phase_based_profiler'):
        if hasattr(runner, a):
            out['runner_profiler_values'][a] = repr(getattr(runner, a))[:200]
    if hasattr(worker, 'profile'):
        try:
            out['profile_signature'] = str(_i.signature(worker.profile))
            out['profile_source'] = _i.getsource(worker.profile)[:1500]
        except Exception as exc:
            out['profile_introspect_error'] = repr(exc)
    try:
        import jax
        out['jax_version'] = jax.__version__
        out['has_jax_profiler'] = hasattr(jax, 'profiler')
        out['trace_annotation_enabled'] = bool(jax.profiler.TraceAnnotation.is_enabled()) \
            if hasattr(jax.profiler, 'TraceAnnotation') and hasattr(jax.profiler.TraceAnnotation, 'is_enabled') else None
    except Exception as exc:
        out['jax_error'] = repr(exc)
    return out


def probe_profiler(llm):
    return {'surface': llm.llm_engine.collective_rpc(_profiler_surface)}


def _trace_steps(worker, logdir, steps):
    """Capture a JAX profiler trace over already-warmed steps. No weight reload."""
    if worker.rank != 0:
        return None
    import jax, os, time as _t
    os.makedirs(logdir, exist_ok=True)
    jax.profiler.start_trace(logdir)
    t0 = _t.perf_counter()
    return {'started': True, 'logdir': logdir, 't0': t0, 'steps_requested': steps}


def _trace_stop(worker):
    if worker.rank != 0:
        return None
    import jax, os, time as _t
    jax.profiler.stop_trace()
    return {'stopped': True, 'stopped_at': _t.perf_counter()}


def profile_decode(llm, steps=80, n=96, d=256, warm_steps=40,
                   logdir='/tmp/omp_jax_trace_decode'):
    """Capture a real JAX op trace over warmed decode steps on the hot actor.

    Real weights, real ShareGPT prompts, no reload. The trace is taken only after
    the engine is already in steady-state decode so prefill and compilation are
    excluded from the captured window.
    """
    from vllm import SamplingParams
    bind_prompt_catalogues(llm)
    import os, glob as _g
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    prompts = load_sharegpt_prompts()
    for i in range(n):
        engine.add_request(f'prof{i}',
                           {'prompt_token_ids': prompts[i % len(prompts)]['prompt_token_ids']},
                           SamplingParams(max_tokens=d, temperature=0.0,
                                          ignore_eos=True, detokenize=False))
    # burn through prefill so the captured window is decode-dominated
    warmed, decode_only_seen = 0, 0
    while warmed < warm_steps and scheduler.has_requests():
        engine.step(); warmed += 1
    runner_states = engine.collective_rpc(_profiler_surface)
    started = engine.collective_rpc(_trace_steps, args=(logdir, steps))
    t0 = time.perf_counter()
    captured = 0
    while captured < steps and scheduler.has_requests():
        engine.step(); captured += 1
    wall = time.perf_counter() - t0
    stopped = engine.collective_rpc(_trace_stop)
    while scheduler.has_requests() or engine.has_unfinished_requests():
        engine.step()
    files = sorted(_g.glob(os.path.join(logdir, '**', '*'), recursive=True))
    return {'warm_steps': warmed, 'captured_steps': captured, 'capture_wall_s': wall,
            'mean_step_ms': 1000*wall/max(1, captured),
            'logdir': logdir, 'trace_files': files[:40], 'trace_file_count': len(files),
            'trace_bytes': sum(os.path.getsize(f) for f in files if os.path.isfile(f)),
            'start_rpc': started, 'stop_rpc': stopped, 'profiler_surface': runner_states}


def summarize_trace(llm, logdir='/tmp/omp_jax_trace_decode', top=45):
    """Aggregate a captured Chrome trace by op name, on the actor.

    The raw trace is ~170 MB; only the aggregate crosses the wire. Durations are
    summed per event name over the device lanes, then expressed as a share of the
    busiest lane's total so the numbers mean 'fraction of device time'.
    """
    import gzip, glob as _g, os, collections as _c
    paths = _g.glob(os.path.join(logdir, '**', '*.trace.json.gz'), recursive=True)
    if not paths:
        return {'error': 'no trace.json.gz found', 'logdir': logdir}
    path = max(paths, key=os.path.getsize)
    with gzip.open(path, 'rt') as fh:
        blob = json.load(fh)
    events = blob.get('traceEvents', blob if isinstance(blob, list) else [])
    pid_names, tid_names = {}, {}
    for e in events:
        if e.get('ph') == 'M' and e.get('name') == 'process_name':
            pid_names[e.get('pid')] = e.get('args', {}).get('name', '')
        if e.get('ph') == 'M' and e.get('name') == 'thread_name':
            tid_names[(e.get('pid'), e.get('tid'))] = e.get('args', {}).get('name', '')
    lane = _c.Counter()
    by_lane_op = _c.defaultdict(_c.Counter)
    for e in events:
        if e.get('ph') != 'X' or 'dur' not in e:
            continue
        key = (pid_names.get(e.get('pid'), str(e.get('pid'))),
               tid_names.get((e.get('pid'), e.get('tid')), str(e.get('tid'))))
        lane[key] += e['dur']
        by_lane_op[key][e.get('name', '?')] += e['dur']
    if not lane:
        return {'error': 'no duration events', 'trace_path': path,
                'event_count': len(events)}
    # Device op lane, not the host lane: the busiest lane is always the Ray worker's CPU.
    def lane_rank(item):
        (proc, thread), us = item
        dev = 'TPU' in proc or 'device' in proc
        ops_lane = 'Ops' in thread or 'ops' in thread
        return (dev and ops_lane, dev, us)
    busiest, busy_us = max(lane.items(), key=lane_rank)
    ops = by_lane_op[busiest]
    total = sum(ops.values()) or 1
    groups = _c.Counter()
    for nm, dur in ops.items():
        if 'RPAd' in nm or 'RPA' in nm:
            groups['attention_sliding' if '-sw_' in nm else 'attention_full'] += dur
        elif 'fusion' in nm or 'matmul' in nm or 'dot' in nm or 'convolution' in nm:
            groups['gemm_fusion'] += dur
        elif 'all-reduce' in nm or 'all-gather' in nm or 'collective' in nm or 'reduce-scatter' in nm:
            groups['collective'] += dur
        elif 'copy' in nm or 'dynamic-update' in nm or 'dynamic-slice' in nm:
            groups['memory_movement'] += dur
        else:
            groups['other'] += dur
    # DoD 13 needs FFN and LM-head+sampling attributed, not just attention. Return the
    # non-attention ops verbatim so classification is auditable rather than asserted.
    non_attn = _c.Counter({n: d for n, d in ops.items() if 'RPA' not in n})
    # Op names are generic XLA fusions; semantic identity lives in per-event args
    # (hlo metadata / source info). Sample them so attribution is evidence-based.
    want = {n for n, _ in non_attn.most_common(12)}
    arg_samples, meta_keys = {}, _c.Counter()
    for e in events:
        if e.get('ph') != 'X' or e.get('name') not in want:
            continue
        a = e.get('args') or {}
        meta_keys.update(a.keys())
        if e['name'] not in arg_samples and a:
            arg_samples[e['name']] = {k: str(v)[:220] for k, v in a.items()}
    # aggregate device time by any long/source name we can find
    # tf_op carries the module path (jit(run_model)/<Module>/...), which is the authoritative
    # attribution for DoD 13. source gives file:line as a cross-check.
    by_module, by_source, by_category, flops = _c.Counter(), _c.Counter(), _c.Counter(), _c.Counter()
    by_tfop_full, by_stack = _c.Counter(), _c.Counter()
    attributed = 0.0
    for e in events:
        if e.get('ph') != 'X' or 'dur' not in e:
            continue
        a = e.get('args') or {}
        tf = a.get('tf_op')
        if tf:
            parts = [q for q in str(tf).split('/') if q and not q.startswith('jit(')]
            by_module[parts[0] if parts else str(tf)[:80]] += e['dur']
            by_tfop_full[str(tf)[:180]] += e['dur']
            attributed += e['dur']
            try: flops[parts[0] if parts else '?'] += float(a.get('model_flops') or 0)
            except Exception: pass
        src = a.get('source')
        if src: by_source[str(src)[:120]] += e['dur']
        st = a.get('source_stack')
        if st: by_stack[str(st).replace(chr(10), ' | ')[:260]] += e['dur']
        cat = a.get('hlo_category')
        if cat: by_category[str(cat)[:60]] += e['dur']
    by_meta = by_module
    return {'trace_path': path, 'trace_bytes': os.path.getsize(path),
            'non_attention_top': [{'name': n[:150], 'total_us': d, 'pct': round(100*d/total, 3)}
                                  for n, d in non_attn.most_common(60)],
            'non_attention_total_us': sum(non_attn.values()),
            'arg_keys_seen': dict(meta_keys.most_common(12)),
            'arg_samples': arg_samples,
            'by_module_us': [{'module': t, 'total_us': v} for t, v in by_module.most_common(30)],
            'by_source_us': [{'source': t, 'total_us': v} for t, v in by_source.most_common(60)],
            'by_tfop_full_us': [{'tf_op': t, 'total_us': v} for t, v in by_tfop_full.most_common(40)],
            'by_source_stack_us': [{'stack': t, 'total_us': v} for t, v in by_stack.most_common(30)],
            'by_hlo_category_us': [{'category': t, 'total_us': v} for t, v in by_category.most_common(15)],
            'module_attributed_us': attributed,
            'op_groups': [{'group': g, 'total_us': d, 'pct': round(100*d/total, 2)}
                          for g, d in groups.most_common()],
            'distinct_ops': len(ops),
            'attention_op_counts': {
                'full': sum(1 for n in ops if ('RPAd' in n or 'RPA' in n) and '-sw_' not in n),
                'sliding': sum(1 for n in ops if ('RPAd' in n or 'RPA' in n) and '-sw_' in n)},
            'event_count': len(events),
            'lanes': [{'lane': f'{k[0]} / {k[1]}', 'total_us': v}
                      for k, v in lane.most_common(8)],
            'busiest_lane': f'{busiest[0]} / {busiest[1]}',
            'busiest_total_us': busy_us,
            'top_ops': [{'name': n[:130], 'total_us': d, 'pct': round(100*d/total, 2)}
                        for n, d in ops.most_common(top)]}


def calibrate_prefill(llm, lengths=(256, 512, 1024, 1536, 2048, 3072, 4096), repeats=3):
    """DoD 12: marginal prefill cost from PREFILL-ONLY steps.

    One request at a time, `max_tokens=1`, nothing else resident, so every step is pure
    prefill (the final step also emits the single token). Chunked prefill splits a prompt of
    L tokens over ceil(L / max_num_batched_tokens) steps; we time the whole prefill and also
    each step, then regress total time on L to get us/prefill-token with no decode residents
    to confound it.
    """
    from vllm import SamplingParams
    bind_prompt_catalogues(llm)
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    prompts = load_sharegpt_prompts()
    # Real ShareGPT text only: a contiguous prefix of one real prompt when one is long
    # enough, otherwise consecutive WHOLE real prompts concatenated at prompt boundaries.
    # Never a stride through a flattened token pool -- that is scrambled synthetic text.
    ordered = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]['prompt_token_ids']))
    def build(L, salt):
        longest = prompts[ordered[0]]['prompt_token_ids']
        if len(longest) >= L:
            for i in ordered:
                toks = prompts[i]['prompt_token_ids']
                if len(toks) >= L:
                    return list(toks[:L]), [i], 'prefix_of_one_prompt'
        used, toks = [], []
        i = salt % len(prompts)
        while len(toks) < L:
            src = prompts[i % len(prompts)]['prompt_token_ids']
            toks.extend(src); used.append(i % len(prompts)); i += 1
        return toks[:L], used, 'concatenated_whole_prompts'
    budget = engine.vllm_config.scheduler_config.max_num_batched_tokens
    rows = []
    rid = 0
    for L in lengths:
        for rep in range(repeats):
            rid += 1
            toks, src_ids, mode = build(L, rid)
            steps = []
            engine.add_request(f'pf{rid}', {'prompt_token_ids': toks},
                               SamplingParams(max_tokens=1, temperature=0.0,
                                              ignore_eos=True, detokenize=False))
            t_all = time.perf_counter()
            while scheduler.has_requests() or engine.has_unfinished_requests():
                t0 = time.perf_counter()
                out = engine.step()
                steps.append(time.perf_counter() - t0)
            total = time.perf_counter() - t_all
            rows.append({'prompt_tokens': L, 'repeat': rep, 'total_s': total,
                         'n_steps': len(steps), 'step_s': steps,
                         'source_prompt_indices': src_ids, 'construction': mode,
                         'expected_chunks': math.ceil(L / budget)})
    by_len = {}
    for r in rows:
        by_len.setdefault(r['prompt_tokens'], []).append(r['total_s'])
    medians = {L: statistics.median(v) for L, v in by_len.items()}
    xs = sorted(medians)
    n = len(xs)
    sx = sum(xs); sy = sum(medians[x] for x in xs)
    sxx = sum(x * x for x in xs); sxy = sum(x * medians[x] for x in xs)
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom if denom else float('nan')
    intercept = (sy - slope * sx) / n if denom else float('nan')
    return {'max_num_batched_tokens': budget,
            'per_length_median_s': medians,
            'fixed_overhead_s': intercept,
            'us_per_prefill_token': slope * 1e6,
            'prefill_tokens_per_s': (1.0 / slope) if slope else None,
            'mfu_pct_at_31B': 100 * 2 * 31e9 * (1.0 / slope) / (197e12 * 16) if slope else None,
            'rows': rows}


def clock_probe(llm):
    return {'wall_s': time.time(), 'perf_s': time.perf_counter()}


def sync_clocks(actors, probes=9):
    samples = []
    for actor in actors:
        best = None
        for _ in range(probes):
            before = time.time()
            reading = ray.get(actor.apply.remote(clock_probe))
            after = time.time()
            round_trip = after - before
            candidate = {'round_trip_s': round_trip, 'offset_s': reading['wall_s'] - (before + after) / 2,
                         'offset_uncertainty_s': round_trip / 2}
            if best is None or candidate['round_trip_s'] < best['round_trip_s']:
                best = candidate
        best['probes'] = probes
        best['definition'] = ('engine wall clock minus coordinator wall clock, from the minimum round trip probe; '
                              'coordinator timeline t = run_start_wall_s - offset_s + engine relative timestamp')
        samples.append(best)
    return samples


def fleet_shard(llm, case, workload, queue, index):
    try:
        return measure(llm, case, 1, workload, (queue, index))
    except Exception as error:
        queue.put({'index': index, 'error': str(error)})
        raise


def measure_fleet(names, case, repeats):
    from ray.util.queue import Queue
    from ray.util.placement_group import get_placement_group
    actors = [ray.get_actor(name) for name in names]
    manifests = ray.get([actor.initialize.remote() for actor in actors])
    context_limits = ray.get([actor.apply.remote(lambda llm: llm.llm_engine.vllm_config.model_config.max_model_len) for actor in actors])
    ips = [ip for manifest in manifests for ip in manifest['host_ips']]
    if len(set(ips)) != len(ips) or len({m['model_config_sha256'] for m in manifests}) != 1:
        raise ValueError('Fleet engines must have disjoint hardware and identical model weights configuration')
    config = manifests[0]['model_config']
    vocab = config.get('text_config', config)['vocab_size']
    classes = case.get('classes', [case])
    assignment = case.get('assignment', 'round_robin')
    if assignment not in ('round_robin', 'by_class') or repeats < 1:
        raise ValueError('Invalid fleet measurement specification')
    if any(min(c['n'], c['p'], c['d']) < 1 or c['p'] + c['d'] > min(context_limits) for c in classes):
        raise ValueError('Invalid fleet workload dimensions')
    records = []
    clock_sync = sync_clocks(actors)
    for repeat in range(-1, repeats + 1):
        rng = random.Random(case.get('seed', 42) + repeat * 1009)
        workload = []
        for class_index, spec in enumerate(classes):
            for _ in range(spec['n']):
                workload.append({'index': len(workload), 'class_index': class_index, 'd': spec['d'],
                                 'prompt_token_ids': [rng.randrange(100, vocab) for _ in range(spec['p'])]})
        shards = [[] for _ in actors]
        for request in workload:
            target = request['class_index'] if assignment == 'by_class' else request['index']
            shards[target % len(actors)].append(request)
        if any(not shard for shard in shards):
            raise ValueError('Every allocated engine must receive work')
        queue = Queue(actor_options={'num_cpus': 0, 'scheduling_strategy': PlacementGroupSchedulingStrategy(
            placement_group=get_placement_group(names[0] + '_pg'), placement_group_bundle_index=0)})
        try:
            local_cases = [{'n': len(shard), 'p': max(len(r['prompt_token_ids']) for r in shard),
                            'd': max(r['d'] for r in shard), 'repeat': repeat,
                            'warmup': repeat < 0, 'trace': repeat != repeats} for shard in shards]
            start = time.perf_counter()
            futures = [actor.apply.remote(fleet_shard, local_case, shard, queue, index)
                       for index, (actor, local_case, shard) in enumerate(zip(actors, local_cases, shards))]
            launch_span = time.perf_counter() - start
            completed = set()
            while len(completed) < len(actors):
                event = queue.get(timeout=1800)
                if 'error' in event:
                    raise RuntimeError('Fleet participant failed: ' + event['error'])
                completed.add(event['index'])
            wall = time.perf_counter() - start
            rows = ray.get(futures)
        finally:
            queue.shutdown()
        tokens = sum(row['output_tokens'] for row in rows)
        record = {'repeat': repeat, 'warmup': repeat < 0, 'trace_enabled': repeat != repeats,
                  'joint_completion_wall_s': wall, 'launch_span_s': launch_span,
                  'output_tokens': tokens, 'output_tok_s': tokens / wall,
                  'workload_sha256': hashlib.sha256(json.dumps(workload, separators=(',', ':')).encode()).hexdigest(),
                  'assignment_indices': [[r['index'] for r in shard] for shard in shards], 'engines': rows}
        records.append(record)
        emit('AUDIT_FLEET_ITERATION', repeat=repeat, warmup=repeat < 0, wall_s=wall, output_tok_s=tokens / wall)
    return {'case': case, 'engine_names': names, 'total_chips': sum(m['tp'] for m in manifests),
            'engine_manifests': manifests, 'engine_max_model_lens': context_limits, 'runs': records,
            'clock_sync': clock_sync,
            'clock_semantics': ('per engine iteration timestamps are perf_counter offsets from that engine run_start; '
                                'map to the coordinator timeline with run_start_wall_s - offset_s before intersecting windows; '
                                'offset_uncertainty_s bounds the alignment error'),
            'timing_semantics': 'driver dispatch through completion notifications from all engines; excludes subsequent trace retrieval; includes submission and notification transport'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', type=int, choices=(8, 16), required=True)
    parser.add_argument('--action', choices=('initialize', 'measure', 'policy', 'release', 'recover', 'inspect', 'probe', 'admission-audit', 'f3-audit', 'probe-profiler', 'profile-decode', 'summarize-trace', 'calibrate-prefill', 'fleet', 'sources'), required=True)
    parser.add_argument('--case', type=json.loads)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--run-id', default='20260911')
    parser.add_argument('--recreate', action='store_true', help='Discard this audit actor before initialization')
    parser.add_argument('--topology-map', type=json.loads, help='Observed full-slice per-host JAX chip coordinates')
    parser.add_argument('--host-ips', help='Explicit comma-separated hosts selected from the observed topology map')
    parser.add_argument('--peer-run-id', help='Second same-size engine for the equal-chip fleet control')
    args = parser.parse_args()
    ray.init(address='auto', namespace=NAMESPACE)
    if args.action == 'sources':
        wanted = set(args.case['sha256'])
        found = {}
        cache = pathlib.Path(__file__).resolve().parents[2]
        for path in cache.glob('_ray_pkg_*/benchmarks/run_measurement_audit.py'):
            source = path.read_bytes()
            digest = hashlib.sha256(source).hexdigest()
            if digest in wanted:
                found[digest] = source.decode()
        print('TOPOLOGY_RESULT_JSON=' + json.dumps({'sources': found, 'missing': sorted(wanted - found.keys())}), flush=True)
        return
    name = f'measurement_audit_{args.run_id}_tp{args.tp}'
    pg_name = name + '_pg'
    if args.action == 'release':
        try:
            ray.kill(ray.get_actor(name), no_restart=True)
        except ValueError:
            pass
        from ray.util.placement_group import get_placement_group
        try:
            owned_pg = get_placement_group(pg_name)
        except ValueError:
            owned_pg = None
        if owned_pg is not None:
            remove_placement_group(owned_pg)
        deadline = time.monotonic() + 60
        while True:
            try:
                ray.get_actor(name)
                actor_absent = False
            except ValueError:
                actor_absent = True
            group_state = placement_group_table(owned_pg).get('state') if owned_pg is not None else 'ABSENT'
            if actor_absent and group_state in ('REMOVED', 'ABSENT'):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Owned resource release did not settle: actor_absent={actor_absent}, pg_state={group_state}')
            time.sleep(0.1)
        print('TOPOLOGY_RESULT_JSON=' + json.dumps({'released': name, 'actor_absent': actor_absent,
                                                   'placement_group': pg_name, 'placement_group_state': group_state}), flush=True)
        return
    if args.action == 'initialize':
        if args.recreate:
            try:
                ray.kill(ray.get_actor(name), no_restart=True)
            except ValueError:
                pass
        try:
            actor = ray.get_actor(name)
        except ValueError:
            discovery = TPUTopologyDiscovery()
            groups = discovery.discover_tpu_nodes()
            hosts = sorted([h for group in groups.values() for h in group], key=lambda h: h.worker_id)
            nodes = {n['NodeManagerAddress']: n for n in ray.nodes() if n.get('Alive')}
            if args.tp == 8:
                if not args.topology_map:
                    raise ValueError('TP8 requires observed JAX coordinates; worker IDs are not a geometry proof')
                proof = {row['ip']: row['chip_coords'] for row in args.topology_map}
                if args.host_ips:
                    ips = args.host_ips.split(',')
                else:
                    leftmost = min(min(c[0] for c in coords) for coords in proof.values())
                    ips = [ip for ip, coords in proof.items() if min(c[0] for c in coords) == leftmost]
                if len(ips) != 2 or len(set(ips)) != 2:
                    raise ValueError('Select exactly two distinct hosts for TP8')
                by_ip = {h.node_ip: h for h in hosts}
                ips.sort(key=lambda ip: min(c[1] for c in proof[ip]))
                selected = [by_ip[ip] for ip in ips]
                chips = {tuple(c) for ip in ips for c in proof[ip]}
                xs, ys, zs = ({c[axis] for c in chips} for axis in range(3))
                if len(chips) != 8 or len(xs) != 2 or len(ys) != 4 or len(zs) != 1 or max(xs) - min(xs) != 1 or max(ys) - min(ys) != 3:
                    raise RuntimeError('Observed devices do not form a contiguous 2x4 rectangle')
                bounds = '1,2,1'
            else:
                selected = hosts
                bounds = '2,2,1'
            slices = {nodes[h.node_ip].get('Labels', {}).get('ray.io/tpu-slice-name') for h in selected}
            if len(selected) != args.tp // 4 or any(h.tpu_count != 4 for h in selected) or len(slices) != 1 or None in slices:
                raise RuntimeError('Selected hosts do not form the required same-slice allocation')
            # Only this audit's abandoned reservation is eligible for cleanup.
            for info in placement_group_table().values():
                if info.get('name') == pg_name and info.get('state') in ('CREATED', 'PENDING'):
                    from ray.util.placement_group import get_placement_group
                    remove_placement_group(get_placement_group(pg_name))
            ips = [h.node_ip for h in selected]
            bundles = [{'TPU': 4.0, 'node:' + ip: 0.001} for ip in ips]
            pg = placement_group(bundles, strategy='STRICT_SPREAD', name=pg_name, lifetime='detached')
            ray.get(pg.ready(), timeout=120)
            actor = AuditWorker.options(name=name, lifetime='detached', scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=0, placement_group_capture_child_tasks=True,
            )).remote(args.tp, ips, bounds, pg)
        result = ray.get(actor.initialize.remote())
    elif args.action == 'inspect':
        actor = ray.get_actor(name)
        result = ray.get(actor.apply.remote(inspect_runtime))
    elif args.action == 'probe':
        actor = ray.get_actor(name)
        result = ray.get(actor.apply.remote(probe_scheduler))
    elif args.action == 'admission-audit':
        actor = ray.get_actor(name)
        result = ray.get(actor.apply.remote(kv_admission.conformance))
    elif args.action == 'f3-audit':
        actor = ray.get_actor(name)
        result = ray.get(actor.apply.remote(kv_eviction.installed_scheduler_sign_fixture))
    elif args.action == 'probe-profiler':
        actor = ray.get_actor(name)
        result = ray.get(actor.apply.remote(probe_profiler))
    elif args.action == 'profile-decode':
        actor = ray.get_actor(name)
        ensure_prompt_catalogue(actor, 'small')
        result = ray.get(actor.apply.remote(profile_decode))
    elif args.action == 'calibrate-prefill':
        actor = ray.get_actor(name)
        ensure_prompt_catalogue(actor, 'small')
        result = ray.get(actor.apply.remote(calibrate_prefill))
    elif args.action == 'summarize-trace':
        actor = ray.get_actor(name)
        result = ray.get(actor.apply.remote(summarize_trace,
            (args.case or {}).get('logdir', '/tmp/omp_jax_trace_decode')))
    elif args.action == 'recover':
        actor = ray.get_actor(name)
        result = ray.get(actor.apply.remote(recover_quiescence))
    elif args.action == 'policy':
        if not args.case:
            raise ValueError('--case is required for a policy run')
        actor = ray.get_actor(name)
        grouped = isinstance(args.case, list)
        # One bad arm must not discard the arms that already ran: a policy step can cost
        # hours, and results are only serialised after the whole step returns.
        measured = []
        failures = []
        for spec in (args.case if grouped else [args.case]):
            try:
                if spec.get('source') in ('drawn', 'sharegpt'):
                    ensure_prompt_catalogue(actor, (spec.get('draw') or {}).get('catalogue', 'small'))
                measured.append(run_with_progress_relay(actor, measure_policy, spec, args.repeats))
            except Exception as exc:
                if not grouped:
                    raise
                label = (spec.get('policy') or {}).get('name', '?')
                failures.append({'policy': label, 'error': f'{type(exc).__name__}: {exc}'})
                measured.append({'policy_name': label, 'failed': True,
                                 'error': f'{type(exc).__name__}: {exc}'})
        result = {'policy_cases': measured} if grouped else measured[0]
        if grouped and failures:
            result['failed_cases'] = failures
    elif args.action == 'fleet':
        names = [name]
        if args.peer_run_id:
            names.append(f'measurement_audit_{args.peer_run_id}_tp{args.tp}')
        grouped = isinstance(args.case, list)
        measured = [measure_fleet(names, spec, args.repeats) for spec in (args.case if grouped else [args.case])]
        result = {'fleet_cases': measured} if grouped else measured[0]
    else:
        if not args.case:
            raise ValueError('--case is required for measurement')
        actor = ray.get_actor(name)
        result = ray.get(actor.apply.remote(measure, args.case, args.repeats))
    print('TOPOLOGY_RESULT_JSON=' + json.dumps(result, default=str), flush=True)


if __name__ == '__main__':
    main()
