"""Live allocation snapshots and the audited synchronous ready-service discipline.

This adapter may inspect engine internals; policy models receive only public snapshots.
No target length is read here. Forecasts consume engine-observed output, not step counts.
"""
from collections import OrderedDict
from itertools import islice
import time


class ServiceDiscipline:
    def __init__(self, llm, synchronous=False, ready_first=False, strict=False):
        self.core = llm.llm_engine.engine_core.engine_core
        self.scheduler = self.core.scheduler
        self.cache = self.scheduler.kv_cache_manager
        self.pool = self.cache.block_pool
        self.config = llm.llm_engine.vllm_config
        self._allocation_managers = self.cache.coordinator.single_type_managers
        self.synchronous = synchronous
        self.ready_first = ready_first
        self.strict = strict
        self.original_step_fn = self.core.step_fn
        self.original_defer_free = self.scheduler.defer_block_free
        self.initial_free = self.pool.get_num_free_blocks()
        self.ready_ids = ()
        self.audit = {'synchronous': synchronous, 'ready_first': ready_first,
                      'strict': strict, 'ready_services': 0, 'skipped_ready_services': 0,
                      'allocation_checks': 0, 'max_unattributed_blocks': 0,
                      'first_violation': None, 'snapshot_wall_s': 0.0}
        if strict and not (synchronous and ready_first):
            raise ValueError('Release-credit certificates require synchronous ready-first service')
        if synchronous:
            if self.core.batch_queue or self.scheduler.get_num_unfinished_requests():
                raise RuntimeError('Synchronous cutover requires a quiescent engine')
            if self.initial_free != self.pool.num_gpu_blocks - 1:
                raise RuntimeError('Unreturned allocations at synchronous cutover')
            config = self.config
            if (config.cache_config.enable_prefix_caching
                    or getattr(config.cache_config, 'kv_offloading_size', None)
                    or config.speculative_config is not None
                    or config.parallel_config.pipeline_parallel_size != 1
                    or self.scheduler.num_lookahead_tokens
                    or self.scheduler.connector is not None
                    or self.scheduler.ec_connector is not None):
                raise RuntimeError('Unaudited sharing, speculation, offloading or pipeline geometry')
            self.core.step_fn = self.core.step
            # No kernel or request is in flight. Synchronous step waits for model output
            # before update/free, so no delayed allocator fence is needed in this mode.
            self.scheduler.defer_block_free = False

    def allocated_ids(self, internal_id):
        groups = self.cache.get_blocks(internal_id).get_block_ids()
        if len(groups) != 1:
            raise RuntimeError('Calendar adapter requires the audited single KV allocation group')
        null_id = self.pool.null_block.block_id
        return {block for block in groups[0] if block != null_id}

    def allocated_count(self, internal_id):
        """Count unique native slots without materializing ID lists and sets."""
        if self.config.cache_config.enable_prefix_caching:
            raise RuntimeError('Fast release counts require nonsharing native allocation')
        if len(self._allocation_managers) != 1:
            raise RuntimeError('Release counts require the audited single allocation group')
        # Borrow the exact list returned by coordinator.get_blocks, without wrappers.
        blocks = self._allocation_managers[0].req_to_blocks.get(internal_id, ())
        null = self.pool.null_block
        return sum(block is not null for block in blocks)

    def snapshot(self, admitted, internal_by_external, indices, public_requests, generated):
        """Return whitelisted live fields; include admission debts and unreturned allocations."""
        start = time.perf_counter()
        running = {request.request_id for request in self.scheduler.running}
        rows, all_blocks = [], set()
        for external in admitted:
            index = indices[external]
            internal = internal_by_external[external]
            request = self.scheduler.requests.get(internal)
            count = generated(index)
            blocks = self.allocated_ids(internal)
            if all_blocks.intersection(blocks):
                raise RuntimeError('Shared KV blocks invalidate the nonsharing certificate')
            all_blocks.update(blocks)
            ready = False
            if request is not None:
                if self.synchronous and request.num_in_flight_tokens:
                    raise RuntimeError('In-flight service survived a synchronous output boundary')
                needed = (request.num_tokens_with_spec + request.num_output_placeholders
                          - request.num_computed_tokens)
                ready = (internal in running and count > 0 and not request.is_prefill_chunk
                         and needed == 1)
                if self.strict and request.has_encoder_inputs:
                    raise RuntimeError('Encoder request is outside the ready-service contract')
            rows.append({'index': index, 'p': public_requests[index]['p'],
                         'g': count, 'blocks': len(blocks), 'ready': ready,
                         'request_id': external, 'internal_id': internal})
        used = self.pool.num_gpu_blocks - 1 - self.pool.get_num_free_blocks()
        other = used - len(all_blocks)
        if other < 0:
            raise RuntimeError('Per-request allocations exceed the measured pool allocation')
        self.audit['allocation_checks'] += 1
        self.audit['max_unattributed_blocks'] = max(self.audit['max_unattributed_blocks'], other)
        self.audit['snapshot_wall_s'] += time.perf_counter() - start
        return rows, other

    def before_schedule(self):
        if not self.ready_first:
            return
        ready, pending = [], []
        for request in self.scheduler.running:
            needed = (request.num_tokens_with_spec + request.num_output_placeholders
                      - request.num_computed_tokens)
            if not request.is_prefill_chunk and needed == 1:
                ready.append(request)
            else:
                pending.append(request)
        if len(ready) > self.config.scheduler_config.max_num_batched_tokens:
            raise RuntimeError('Decode-ready cohort exceeds the native token budget')
        self.ready_ids = tuple(request.request_id for request in ready)
        self.scheduler.running[:] = ready + pending

    def after_schedule(self, output):
        if not self.ready_first:
            return
        bad = [(request_id, output.num_scheduled_tokens.get(request_id, 0))
               for request_id in self.ready_ids
               if output.num_scheduled_tokens.get(request_id, 0) != 1]
        self.audit['ready_services'] += len(self.ready_ids) - len(bad)
        self.audit['skipped_ready_services'] += len(bad)
        if bad and self.audit['first_violation'] is None:
            self.audit['first_violation'] = {'ready_service_mismatch': bad,
                                            'preempted': list(output.preempted_req_ids or ())}
        if self.strict and (bad or output.preempted_req_ids):
            raise RuntimeError('Ready-service certificate falsified: ' + repr(self.audit))

    def new_blocks_needed(self, request, num_new_tokens):
        """Native RUNNING allocation demand, including the deployed coordinator geometry."""
        computed = request.num_computed_tokens
        main_tokens = min(computed, self.cache.max_model_len) + num_new_tokens
        slots = min(main_tokens + self.scheduler.num_lookahead_tokens, self.cache.max_model_len)
        return self.cache.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id, num_tokens=slots,
            new_computed_blocks=self.cache.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0, total_computed_tokens=computed,
            num_local_computed_tokens=computed, num_tokens_main_model=main_tokens)

    def unattributed_blocks(self):
        allocated = set()
        for internal in self.scheduler.requests:
            blocks = self.allocated_ids(internal)
            if allocated.intersection(blocks):
                raise RuntimeError('Shared blocks violate the nonsharing allocation contract')
            allocated.update(blocks)
        other = self.pool.num_gpu_blocks - 1 - self.pool.get_num_free_blocks() - len(allocated)
        if other < 0:
            raise RuntimeError('Request block ownership exceeds pool allocation')
        return other

    def homogeneous_audit(self, scheduler):
        running = scheduler.running
        safe = self.synchronous and not self.core.batch_queue and all(
            request.num_in_flight_tokens == 0 for request in running)
        homogeneous = all(not request.is_prefill_chunk and not request.spec_token_ids
            and not request.has_encoder_inputs and not request.num_output_placeholders
            and request.num_tokens_with_spec - request.num_computed_tokens == 1
            for request in running)
        deficit = 0
        single = True
        if safe and homogeneous and running:
            deficit = max(0, sum(self.new_blocks_needed(request, 1) for request in running)
                          - self.pool.get_num_free_blocks())
            single = all(len(self.allocated_ids(request.request_id)) >= deficit
                         for request in running)
        return {'safe_boundary': bool(safe), 'no_request_blocking': homogeneous,
                'no_deferred_frees': not scheduler.defer_block_free,
                'no_kv_sharing': not self.config.cache_config.enable_prefix_caching,
                'no_waiting_admission': not scheduler.waiting and not scheduler.skipped_waiting,
                'at_most_one_victim': single, 'deficit': deficit}

    def restore(self):
        self.core.step_fn = self.original_step_fn
        self.scheduler.defer_block_free = self.original_defer_free


class PlannerGate:
    """Same live admission functional for exact and conditional future profiles."""
    def __init__(self, model, peak_function, usable_blocks, certificate, request_count,
                 risk=0.01):
        if certificate not in ('ready_epoch_certificate', 'endpoint_certificate'):
            raise ValueError('Unknown planner execution certificate')
        if not 0 < risk < 1 or request_count < 1:
            raise ValueError('Invalid whole-run admission risk')
        self.model = model
        self.peak_function = peak_function
        self.capacity = usable_blocks
        self.certificate = certificate
        self.alpha = risk / request_count
        self.base_peak = 0
        self.pending_added = 0
        self.last_snapshot = None
        self.stats = {'checks': 0, 'admissions': 0, 'blocked': 0,
                      'max_certified_peak': 0, 'certificate': certificate,
                      'whole_run_risk': risk, 'per_admission_risk': self.alpha}

    def refresh(self, snapshots, other_blocks=0):
        ready, pending_blocks = [], other_blocks
        for row in snapshots:
            profile = self.model.future_profile(row['index'], row['g'], row['blocks'])
            terminal = (profile.p + profile.actual_g + profile.cap_remaining
                        + self.model.block_size - 1) // self.model.block_size
            if row['blocks'] > terminal:
                raise RuntimeError('Actual allocation exceeds the declared terminal profile')
            if self.certificate == 'ready_epoch_certificate' and row['ready']:
                ready.append(profile)
            else:
                pending_blocks += profile.terminal_blocks
        self.base_peak = self.peak_function(ready, pending_terminal_blocks=pending_blocks,
                                            alpha=self.alpha)
        # At the native post-schedule/pre-sampling boundary, a ready resident
        # computes its already-emitted token: context p+g, not p+g+1.
        # Pending prefills retain endpoint bounds until they become ready.
        self.next_allocation_bound = pending_blocks + sum(
            max(profile.actual_blocks, (profile.p + profile.actual_g
                + self.model.block_size - 1) // self.model.block_size)
            for profile in ready)
        self.pending_added = 0
        self.last_snapshot = snapshots

    def allows(self, index):
        terminal = self.model.future_profile(index).terminal_blocks
        peak = self.base_peak + self.pending_added + terminal
        self.stats['checks'] += 1
        if peak > self.capacity:
            self.stats['blocked'] += 1
            return False
        return True

    def admitted(self, index):
        self.pending_added += self.model.future_profile(index).terminal_blocks
        peak = self.base_peak + self.pending_added
        if peak > self.capacity:
            raise RuntimeError('Admission was committed without a complete certificate')
        self.stats['admissions'] += 1
        self.stats['max_certified_peak'] = max(self.stats['max_certified_peak'], peak)


class PlannedDispatcher:
    """Planned priority, optionally filtered by a live certificate; never timed replay.

    An explicit gate=None consumes the order only. Native allocation/preemption
    then owns memory safety; the timetable is not an execution certificate.
    """
    def __init__(self, views, plan, representative_manifest, gate, lookahead=24):
        self.gate = gate
        self.lookahead = lookahead
        self.resorts = 1
        self.ranking_signal_available = True
        self.queue = OrderedDict()
        for position in plan['queue_indices']:
            view = dict(views[position])
            view['d_sort'] = representative_manifest[position][1]
            self.queue[view['index']] = view
        self.ranks = {index: rank for rank, index in enumerate(self.queue)}
        self.critical = sorted(self.queue, key=lambda index: (
            -self.queue[index]['d_sort'], self.ranks[index])) if gate is not None else None
        self.selected = None

    def peek(self):
        if not self.queue:
            return None
        if self.gate is None:
            self.selected = next(iter(self.queue))
            return self.queue[self.selected]
        candidates = list(islice(self.queue, self.lookahead))
        candidates.extend(islice((index for index in self.critical if index in self.queue), 4))
        for index in sorted(set(candidates), key=self.ranks.__getitem__):
            if self.gate.allows(index):
                self.selected = index
                return self.queue[index]
        self.selected = next(iter(self.queue))
        return self.queue[self.selected]

    def pop(self):
        if self.selected is None:
            raise RuntimeError('Dispatch requires a prior feasibility selection')
        view = self.queue.pop(self.selected)
        self.selected = None
        return view

    def __len__(self):
        return len(self.queue)
