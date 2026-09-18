"""Bounded fresh-admission experiment; native asynchronous execution is untouched.

The injected call is after native waiting token/chunk arithmetic and before its
allocation. Preview executes the audited allocator's own read-only prefix. The
only omitted call frees skipped sliding-window blocks: supported fresh requests
have no blocks, and only FullAttentionManager without prefix caching is accepted.
No completion forecast releases blocks. This is a hypothesis, not a certificate.
"""
import ast
import collections
import copy
from dataclasses import dataclass
import hashlib
import inspect
import textwrap
import time
import types
from kv_eviction import _construct_known


_SCHEDULE_SHA = 'fcc6418f3c4b8e88d1932ff72dbdd872e4556649df903c5ed7a80f4d678f8023'
_MISSING = object()


def _source(fn):
    return textwrap.dedent(inspect.getsource(fn))


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _compile(tree, original):
    ast.fix_missing_locations(tree)
    scope = dict(original.__globals__)
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), scope)
    return scope[tree.body[0].name]


def allocation_preview(original):
    """Extract native arithmetic through the last allocation feasibility check."""
    if _sha(_source(original)) != 'ea1da4aa7dcefddc54ede3ffdf8982b31d7f009eb29dfeb72fabfc149f5d5d7b':
        raise RuntimeError('Unsupported native allocator source')
    tree = ast.parse(_source(original))
    fn = tree.body[0]
    body = []
    omitted = 0
    for node in fn.body:
        text = ast.unparse(node)
        if text.startswith('if new_computed_block_list is not self.empty_kv_cache_blocks.blocks'):
            break
        if text.startswith('self.coordinator.remove_skipped_blocks('):
            omitted += 1
            continue
        body.append(node)
    else:
        raise RuntimeError('Native allocation mutation boundary not found')
    if omitted != 1 or not any('required_blocks > available_blocks' in ast.unparse(n) for n in body):
        raise RuntimeError('Native allocation prefix changed')
    body += ast.parse('return num_blocks_to_allocate').body
    fn.body = body
    fn.returns = None
    return _compile(tree, original), _sha(_source(original))


def transform_schedule(original):
    source = _source(original)
    if _sha(source) != _SCHEDULE_SHA:
        raise RuntimeError('Unsupported native schedule source')
    tree = ast.parse(source)
    matches = []
    for parent in ast.walk(tree):
        body = getattr(parent, 'body', None)
        if not isinstance(body, list):
            continue
        for position, node in enumerate(body):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == 'self.kv_cache_manager.allocate_slots'
                    and any(k.arg == 'full_sequence_must_fit' for k in node.value.keywords)):
                matches.append((body, position, node))
    if len(matches) != 1:
        raise RuntimeError('Fresh allocation seam is not unique')
    body, position, allocation = matches[0]
    call = copy.deepcopy(allocation.value)
    call.func = ast.Attribute(ast.Name('_admission', ast.Load()), 'before', ast.Load())
    call.keywords += [ast.keyword('token_budget', ast.Name('token_budget', ast.Load())),
                      ast.keyword('provisional', ast.Name('num_scheduled_tokens', ast.Load()))]
    gate = ast.If(ast.UnaryOp(ast.Not(), call), ast.parse('''
request_queue.pop_request()
step_skipped_waiting.prepend_request(request)
continue
''').body, [])
    observed = ast.parse('_admission.after(request, new_blocks)').body[0]
    body[position:position + 1] = [gate, allocation, observed]
    ast.fix_missing_locations(tree)
    return tree, {'native_schedule_sha256': _sha(source),
                  'transformed_ast_sha256': _sha(ast.dump(tree)),
                  'seam': 'waiting allocate_slots after resident pass and native chunk arithmetic'}


@dataclass(frozen=True, slots=True)
class Action:
    version: int
    request_id: str
    free: int
    token_budget: int
    tokens: int
    blocks: int | None
    defer: bool
    reason: str


def action_current(action, version, request_id, free, budget, tokens):
    return (action.version == version and action.request_id == request_id
            and action.free == free and action.token_budget == budget and action.tokens == tokens)


class AdmissionController:
    def __init__(self, llm, model, index_of, observed, *, shadow=False, baseline=False,
                 online_limit_s=3.408, startup_elapsed_s=0.0):
        started = time.perf_counter()
        self.engine = llm.llm_engine
        self.core = self.engine.engine_core.engine_core
        self.scheduler = self.core.scheduler
        self.cache = self.scheduler.kv_cache_manager
        self.pool = self.cache.block_pool
        self.model, self.index_of, self.observed = model, index_of, observed
        self.shadow, self.online_limit_s = shadow, online_limit_s
        self.baseline = baseline
        self.version = 0
        self.online_s = 0.0
        self.online_cpu_s = 0.0
        self.counts = collections.Counter()
        self.first_failure = None
        self.disabled = False
        self.seen = set()
        self.inflight = collections.deque()
        self.deferred_pass = None
        self.endpoints = {row['index']: model.future_profile(row['index']).terminal_blocks
                          for row in model.requests}
        self.total_work = {row['index']: model.future_profile(row['index']).quantile(1)
                           for row in model.requests}
        self.position_high_water = {}
        self.commits = {}
        self.pending_tokens = collections.Counter()
        self.last_blocks = {}
        self.state_samples = []
        self.pending_action = None
        self.samples = []
        self.identity_checks = collections.Counter()
        self.original = self.scheduler.schedule
        self.attribute = vars(self.scheduler).get('schedule', _MISSING)
        self.installed = False
        cfg = self.engine.vllm_config
        managers = self.cache.coordinator.single_type_managers
        if (cfg.cache_config.enable_prefix_caching or cfg.cache_config.sliding_window
                or cfg.speculative_config is not None or cfg.parallel_config.pipeline_parallel_size != 1
                or getattr(cfg.cache_config, 'kv_offloading_size', None)
                or self.scheduler.connector is not None or self.scheduler.ec_connector is not None
                or self.scheduler.num_lookahead_tokens or self.scheduler.lora_config
                or self.scheduler.need_mamba_block_aligned_split or self.scheduler.is_encoder_decoder
                or len(managers) != 1 or type(managers[0]).__name__ != 'FullAttentionManager'
                or type(self.cache.coordinator).__name__ != 'KVCacheCoordinatorNoPrefixCache'
                or self.pool.num_gpu_blocks != 567 or cfg.cache_config.block_size != 256
                or self.scheduler.num_spec_tokens or self.scheduler.num_sampled_tokens_per_step != 1
                or cfg.scheduler_config.max_num_seqs != 512
                or cfg.scheduler_config.max_num_batched_tokens != 2048):
            raise RuntimeError('Unsupported source/configuration for admission preview')
        if self.core.batch_queue or self.scheduler.has_requests() or self.pool.get_num_free_blocks() != 566:
            raise RuntimeError('Admission installation requires a quiescent retained engine')
        if getattr(self.core.step_fn, '__func__', None) is not type(self.core).step_with_batch_queue:
            raise RuntimeError('Native asynchronous step function required')
        self.manager = managers[0]
        self.identities = self._identities()
        self.config_repr = repr(cfg)
        self.native_hashes = {}
        for label, fn in [('core.step', type(self.core).step),
                          ('core.step_with_batch_queue', type(self.core).step_with_batch_queue),
                          ('outer.step', type(self.engine).step),
                          ('schedule', self.original.__func__),
                          ('update', type(self.scheduler).update_from_output),
                          ('allocate_slots', type(self.cache).allocate_slots),
                          ('get_num_blocks_to_allocate', type(self.cache.coordinator).get_num_blocks_to_allocate),
                          ('manager.get_num_blocks_to_allocate', type(self.manager).get_num_blocks_to_allocate)]:
            self.native_hashes[label] = _sha(_source(fn))
        for label, expected in {
                'get_num_blocks_to_allocate': '0c839efed4de00ff3f8d0d45fbee2aab78dc0b10ce36cf68b4a6836498b28d5d',
                'manager.get_num_blocks_to_allocate': 'e873242f581e78673004dd59b3bee78642cd7870acd29269c5e29ced33501c8b'}.items():
            if self.native_hashes[label] != expected:
                raise RuntimeError('Unsupported native allocation query source: ' + label)
        if not __debug__:
            raise RuntimeError('C/O information boundary requires enabled validation')
        self.preview, self.allocator_sha = allocation_preview(type(self.cache).allocate_slots)
        tree, self.source_audit = transform_schedule(self.original.__func__)
        scope = dict(self.original.__func__.__globals__)
        scope['_admission'] = self
        exec(compile(tree, inspect.getsourcefile(self.original), 'exec'), scope)
        self.patched = types.MethodType(scope['schedule'], self.scheduler)
        self.startup_s = startup_elapsed_s + time.perf_counter() - started
        if self.startup_s > 5.0:
            raise RuntimeError('Admission startup exceeded five seconds')
        self.check_identity('installation')

    def _identities(self):
        return (getattr(self.core.step_fn, '__func__', self.core.step_fn),
                type(self.core).step, type(self.core).step_with_batch_queue,
                getattr(self.engine.step, '__func__', self.engine.step),
                self.core.batch_queue, getattr(self.core.batch_queue, 'maxlen', None),
                self.scheduler.defer_block_free)

    def check_identity(self, phase):
        current = self._identities()
        if (any(a is not b for a, b in zip(current[:5], self.identities[:5]))
                or current[5:] != self.identities[5:]):
            raise RuntimeError('Native step identity or queue/free configuration changed')
        self.identity_checks[phase] += 1

    def install(self):
        self.scheduler.schedule = self.patched
        self.installed = True

    def fail(self, reason, **facts):
        self.disabled = True
        if self.first_failure is None:
            self.first_failure = {'reason': reason, **facts}

    def allocated(self, request_id):
        return len(self.manager.req_to_blocks.get(request_id, ()))

    def before(self, request, tokens, *, token_budget, provisional, **kwargs):
        start, cpu = time.perf_counter(), time.thread_time()
        try:
            self.check_identity('decision')
            self.version += 1
            rid = request.request_id
            # Observed g=0 is not a freshness test: preempted prefills must resume.
            if rid in self.seen or request.num_preemptions:
                self.counts['resumption_bypass'] += 1
                return True
            self.counts['fresh_boundaries'] += 1
            if self.baseline:
                self.counts['static_native_boundaries'] += 1
                return True
            if self.disabled or self.online_cpu_s >= self.online_limit_s:
                self.counts['fallback'] += 1
                if not self.disabled:
                    self.counts['online_budget_violations'] += 1
                    self.disabled = True
                return True
            if self.deferred_pass == self.scheduler.current_step and not self.shadow:
                # One DEFER_NEW action covers the remaining fresh tail in this
                # pass; resumption has already bypassed this gate above.
                self.counts['deferred_tail_applications'] += 1
                return False
            if (request.has_encoder_inputs or request.num_computed_tokens
                    or request.num_in_flight_tokens or self.allocated(rid)
                    or request.num_output_placeholders):
                self.fail('unsupported_fresh_state', request_id=rid)
                self.counts['fallback'] += 1
                return True
            free = self.pool.get_num_free_blocks()
            blocks = self.preview(self.cache, request, tokens, **kwargs)
            defer, reason = False, 'floor128'
            active = [r for r in self.scheduler.requests.values()
                      if (r.request_id in self.seen or r.status.name == 'RUNNING') and not r.is_finished()]
            debt = 0
            if self.deferred_pass == self.scheduler.current_step:
                defer, reason = True, 'arrival_tail_deferred'
            elif len(active) >= 128 and blocks is not None:
                # Cached endpoint proxies come from the legal FutureModel. Residual
                # debt is endpoint minus ACTUAL ownership, not computed cursors.
                debt = sum(max(0, self.endpoints[self.index_of(resident)]
                               - self.allocated(resident.request_id)) for resident in active)
                defer = max(blocks, self.endpoints[self.index_of(request)]) + debt > free
                reason = 'residual_growth' if defer else 'extra_capacity'
                if defer:
                    self.deferred_pass = self.scheduler.current_step
            if not active:
                defer, reason = False, 'empty_active_progress'
                self.counts['empty_active_progress'] += 1
            action = Action(self.version, rid, free, token_budget, tokens, blocks, defer, reason)
            # Exercise stale-action rejection without mutating any engine state.
            if action_current(action, self.version - 1, rid, free, token_budget, tokens):
                self.fail('stale_action_accepted')
            self.counts['stale_checks'] += 1
            if not action_current(action, self.version, rid, self.pool.get_num_free_blocks(), token_budget, tokens):
                self.counts['stale_rejections'] += 1
                return True
            if time.perf_counter() - start > 0.001:
                self.counts['decision_budget_violations'] += 1
                self.counts['fallback'] += 1
                return True
            self.counts['qualified_decisions'] += 1
            self.counts['proposed_' + reason] += 1
            self.counts['provisional_boundaries'] += bool(provisional)
            self.counts['inflight_boundaries'] += bool(self.core.batch_queue)
            if defer and not self.shadow:
                self.counts['interventions'] += 1
                return False
            self.pending_action = action
            return True
        finally:
            elapsed = time.perf_counter() - start
            self.online_s += elapsed
            self.counts['max_boundary_wall_s'] = max(self.counts['max_boundary_wall_s'], elapsed)
            self.online_cpu_s += time.thread_time() - cpu

    def after(self, request, new_blocks):
        start, cpu = time.perf_counter(), time.thread_time()
        try:
            action, self.pending_action = self.pending_action, None
            if action is None:
                return
            actual = self.pool.get_num_free_blocks()
            expected = action.free - (action.blocks or 0)
            if ((new_blocks is None) != (action.blocks is None) or actual != expected
                    or new_blocks is not None and self.allocated(request.request_id) != action.blocks):
                self.fail('immediate_allocation_mismatch', action=repr(action), actual_free=actual,
                          actual_blocks=self.allocated(request.request_id), success=new_blocks is not None)
            else:
                self.counts['allocation_matches'] += 1
            if new_blocks is not None:
                self.commits[action.request_id] = action
            if len(self.samples) < 16:
                self.samples.append({'version': action.version, 'request_id': action.request_id,
                    'tokens': action.tokens, 'blocks': action.blocks, 'free_before': action.free,
                    'free_after': actual, 'reason': action.reason})
        finally:
            self.online_s += time.perf_counter() - start
            self.online_cpu_s += time.thread_time() - cpu

    def issued(self, output):
        start, cpu = time.perf_counter(), time.thread_time()
        try:
            self.check_identity('issue')
            self.version += 1
            for rid, action in self.commits.items():
                req = self.scheduler.requests[rid]
                if (output.num_scheduled_tokens.get(rid) != action.tokens
                        or req.status.name != 'RUNNING'
                        or req.num_computed_tokens != action.tokens):
                    self.fail('native_bookkeeping_mismatch', action=repr(action))
                else:
                    self.counts['bookkeeping_matches'] += 1
            self.commits.clear()
            self.pending_tokens.update(output.num_scheduled_tokens)
            for rid, tokens in output.num_scheduled_tokens.items():
                req = self.scheduler.requests[rid]
                cursor = req.num_computed_tokens
                high = self.position_high_water.get(rid, 0)
                self.counts['reissued_token_positions'] += max(0, min(high, cursor) - (cursor - tokens))
                self.position_high_water[rid] = max(high, cursor)
                blocks = self.allocated(rid)
                growth = rid in self.last_blocks and blocks > self.last_blocks[rid]
                self.counts['observed_block_growth'] += growth
                self.last_blocks[rid] = blocks
                if req.num_in_flight_tokens != self.pending_tokens[rid]:
                    self.fail('issued_work_mismatch', request_id=rid,
                              native=req.num_in_flight_tokens, tracked=self.pending_tokens[rid])
                if len(self.state_samples) < 8 or growth and self.counts['observed_block_growth'] <= 8:
                    self.state_samples.append({'event': self.version, 'id': rid,
                        'observed_g': self.observed(rid), 'computed': req.num_computed_tokens,
                        'remaining_support_tokens': self.total_work[self.index_of(req)] - self.observed(rid),
                        'inflight_tokens': req.num_in_flight_tokens,
                        'placeholders': req.num_output_placeholders, 'allocated': blocks,
                        'phase': 'prefill' if req.is_prefill_chunk else 'decode',
                        'queue': req.status.name, 'preemptions': req.num_preemptions})
            self.seen.update(output.num_scheduled_tokens)
            self.inflight.append(output)
            self.counts['issues'] += 1
            self.counts['max_outstanding'] = max(self.counts['max_outstanding'], len(self.inflight))
            self.counts['preemptions'] += len(output.preempted_req_ids or ())
        finally:
            self.online_s += time.perf_counter() - start
            self.online_cpu_s += time.thread_time() - cpu

    def retired(self, output):
        start, cpu = time.perf_counter(), time.thread_time()
        try:
            self.check_identity('retirement')
            if not self.inflight or self.inflight.popleft() is not output:
                self.fail('issue_retirement_identity_mismatch')
            self.pending_tokens.subtract(output.num_scheduled_tokens)
            if any(self.pending_tokens[rid] < 0 for rid in output.num_scheduled_tokens):
                self.fail('negative_pending_work')
            self.version += 1
            self.counts['retirements'] += 1
        finally:
            self.online_s += time.perf_counter() - start
            self.online_cpu_s += time.thread_time() - cpu

    def close(self):
        if self.installed:
            if self.attribute is _MISSING:
                del self.scheduler.schedule
            else:
                self.scheduler.schedule = self.attribute
            self.installed = False
        self.check_identity('cleanup')
        self.cleanup = {'client_requests': self.engine.get_num_unfinished_requests(),
                        'scheduler_requests': self.scheduler.get_num_unfinished_requests(),
                        'batch_queue': len(self.core.batch_queue),
                        'tracked_outstanding': len(self.inflight),
                        'pending_tokens': sum(self.pending_tokens.values()),
                        'free_blocks': self.pool.get_num_free_blocks(),
                        'weights_retained': self.engine is not None,
                        'configuration_unchanged': repr(self.engine.vllm_config) == self.config_repr}

    def summary(self):
        return {'shadow': self.shadow, 'counts': dict(self.counts), 'failure': self.first_failure,
                'disabled': self.disabled, 'startup_s': self.startup_s,
                'online_wall_s': self.online_s, 'online_cpu_s': self.online_cpu_s,
                'online_cpu_limit_s': self.online_limit_s, 'decision_limit_s': 0.001,
                'source': self.source_audit, 'native_hashes': self.native_hashes,
                'config_sha256': _sha(self.config_repr),
                'native_identity_ids': [id(value) for value in self.identities[:5]],
                'queue_capacity': self.identities[5], 'defer_block_free': self.identities[6],
                'identity_checks': dict(self.identity_checks), 'cleanup': self.cleanup,
                'state_samples': self.state_samples, 'baseline': self.baseline,
                'samples': self.samples, 'certificate': False}


def native_audit(llm):
    """Inspect installed pure-query dependencies without executing model work."""
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    cache = scheduler.kv_cache_manager
    manager = cache.coordinator.single_type_managers[0]
    sources = {}
    for label, owner, names in (
            ('coordinator', cache.coordinator, ('get_num_blocks_to_allocate', 'remove_skipped_blocks')),
            ('manager', manager, ('get_num_blocks_to_allocate', 'remove_skipped_blocks')),
            ('queue', scheduler.waiting, ('add_request', 'prepend_requests', 'pop_request'))):
        for name in names:
            sources[label + '.' + name] = inspect.getsource(getattr(type(owner), name))
    return {'sources': sources, 'config': repr(llm.llm_engine.vllm_config),
            'free_blocks': cache.block_pool.get_num_free_blocks(),
            'batch_queue_size': llm.llm_engine.engine_core.engine_core.batch_queue_size}


def conformance(llm):
    """CPU mechanics on isolated installed schedulers; never performance evidence."""
    from vllm import SamplingParams
    from vllm.v1.request import Request, RequestStatus
    live = llm.llm_engine.engine_core.engine_core.scheduler
    evidence = native_audit(llm)
    def new_scheduler():
        config = copy.deepcopy(llm.llm_engine.vllm_config)
        return type(live)(vllm_config=config, kv_cache_config=copy.deepcopy(live.kv_cache_config),
                          structured_output_manager=None, block_size=256, log_stats=False)
    def request(rid, p):
        return _construct_known(Request, {
            'request_id': rid, 'prompt_token_ids': [0] * p,
            'sampling_params': SamplingParams(max_tokens=512, ignore_eos=True, temperature=0.0),
            'pooling_params': None, 'arrival_time': 0.0, 'lora_request': None,
            'cache_salt': None, 'mm_features': None, 'prompt_embeds': None,
            'block_hasher': None, 'priority': 0})
    rows = []
    for p, tokens, reserve_all in ((1, 1, False), (256, 256, False),
                                   (257, 256, False), (257, 257, False), (257, 257, True)):
        sched = new_scheduler()
        cache, pool = sched.kv_cache_manager, sched.kv_cache_manager.block_pool
        assert cache is not live.kv_cache_manager and pool is not live.kv_cache_manager.block_pool
        req = request('fixture', p)
        preview, _ = allocation_preview(type(cache).allocate_slots)
        held = pool.get_new_blocks(pool.get_num_free_blocks()) if reserve_all else []
        free = pool.get_num_free_blocks()
        manager = cache.coordinator.single_type_managers[0]
        before = (repr(vars(req)), repr(manager.req_to_blocks), free)
        predicted = preview(cache, req, tokens, has_scheduled_reqs=False)
        assert before == (repr(vars(req)), repr(manager.req_to_blocks), pool.get_num_free_blocks())
        allocated = cache.allocate_slots(req, tokens, has_scheduled_reqs=False)
        assert (predicted is None) == (allocated is None)
        assert pool.get_num_free_blocks() == free - (predicted or 0)
        rows.append({'p': p, 'tokens': tokens, 'free_before': free,
                     'preview_blocks': predicted, 'free_after': pool.get_num_free_blocks(),
                     'preview_read_only': True})
        if held:
            pool.free_blocks(held)
        cache.free(req)
        assert pool.get_num_free_blocks() == 566
    sched = new_scheduler()
    cache, pool = sched.kv_cache_manager, sched.kv_cache_manager.block_pool
    growing = request('growing', 256)
    assert cache.allocate_slots(growing, 256, has_scheduled_reqs=False) is not None
    growing.num_computed_tokens = growing.num_in_flight_tokens = 256
    growing.append_output_token_ids([0])
    growing.status = RequestStatus.RUNNING
    free = pool.get_num_free_blocks()
    preview, _ = allocation_preview(type(cache).allocate_slots)
    growth = preview(cache, growing, 1)
    assert growth == 1 and cache.allocate_slots(growing, 1) is not None
    assert pool.get_num_free_blocks() == free - 1
    cache.free(growing)
    assert pool.get_num_free_blocks() == 566
    sched = new_scheduler()
    class Gate:
        def __init__(self):
            self.allow = False
            self.matches = []
            self.pending = None
        def before(self, req, tokens, *, token_budget, provisional, **kwargs):
            if not self.allow and not req.num_preemptions:
                return False
            cache = sched.kv_cache_manager
            preview, _ = allocation_preview(type(cache).allocate_slots)
            self.pending = (req.request_id, tokens, preview(cache, req, tokens, **kwargs),
                            cache.block_pool.get_num_free_blocks())
            return True
        def after(self, req, blocks):
            rid, tokens, predicted, free = self.pending
            assert rid == req.request_id
            assert (predicted is None) == (blocks is None)
            assert sched.kv_cache_manager.block_pool.get_num_free_blocks() == free - (predicted or 0)
            self.matches.append({'id': rid, 'tokens': tokens, 'blocks': predicted})
    gate = Gate()
    tree, seam = transform_schedule(sched.schedule.__func__)
    scope = dict(sched.schedule.__func__.__globals__)
    scope['_admission'] = gate
    exec(compile(tree, inspect.getsourcefile(sched.schedule), 'exec'), scope)
    sched.schedule = types.MethodType(scope['schedule'], sched)
    for rid in ('fresh0', 'resumed', 'fresh1'):
        req = request(rid, 257)
        sched.add_request(req)
        if rid == 'resumed':
            req.status = RequestStatus.PREEMPTED
            req.num_preemptions = 1
    first = sched.schedule()
    assert list(first.num_scheduled_tokens) == ['resumed']
    queued = [r.request_id for r in sched.skipped_waiting]
    assert queued == ['fresh0', 'fresh1']
    gate.allow = True
    second = sched.schedule()
    assert [r.req_id for r in second.scheduled_new_reqs] == ['fresh0', 'fresh1']
    action = Action(2, 'fresh', 3, 4, 1, 1, False, 'fixture')
    assert action_current(action, 2, 'fresh', 3, 4, 1)
    assert not action_current(action, 1, 'fresh', 3, 4, 1)
    assert not action_current(action, 2, 'fresh', 2, 4, 1)
    evidence.update(allocation_cases=rows, growth_blocks=growth, stale_rejection=True,
                    recomputation_bypassed_deferred_fresh=True, preserved_fresh_order=queued,
                    seam=seam, native_boundary_matches=gate.matches,
                    live_free_after=live.kv_cache_manager.block_pool.get_num_free_blocks(),
                    performance_evidence=False, passed=True)
    return evidence
