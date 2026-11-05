"""TPUFlex for CUDA vLLM v0.23.0 (TP1, H100), as a vLLM general plugin loaded in every vLLM process.

Installed by pod_run.sh for the pm-* arms (module + dist-info with a vllm.general_plugins entry point on
PYTHONPATH). Every mechanism is off unless its PM_* variable is set; with none set, register() patches nothing.
Ported from the vLLM-TorchTPU plugin (~/hetero-wt/TorchTpuLoad/scripts/pm_torchtpu_plugin.py).

- Zygote (Decompose + Overlap), PM_ZYGOTE_SOCK: the EngineCore process (vLLM spawns it with the `spawn` start
  method) is forked from pm_zygote.py, a fork server that imported the engine's stack while the API server
  imported its own. CUDA is not initialized in the zygote.
- Gate (Decompose), PM_GATE_FILE: the engine parks in WorkerWrapperBase.init_device, before the CUDA context
  and NCCL init, until the file exists (warm-pm: pre-spawned next to the engine it replaces).
- Terminal weight snapshot (Cache), PM_SNAP_DIR + PM_SNAP_MODE, taken at the end of GPUModelRunner.load_model,
  i.e. after process_weights_after_loading: every parameter, buffer and other CUDA tensor attribute of the
  model, the exact state the engine serves from.
    save     stock load_model, then the tensors are written to shard<k>.bin files (~PM_SNAP_SHARD_MB each,
             tensor-aligned) plus manifest.json (commit marker, written last).
    restore  DefaultModelLoader.load_weights only zero-fills the parameters (no checkpoint is read), the rest
             of load_model runs on them, then a background thread copies every shard into the same tensors
             (in place: CUDA graphs and compiled code keep their pointers) as soon as its shard<k>.done
             marker exists, i.e. while later shards are still being delivered. The profile run, torch.compile
             and the warm-up run on zero weights (only shapes matter there). The copy joins before CUDA graph
             capture (PM_SNAP_JOIN=capture, default) or after the whole warm-up (PM_SNAP_JOIN=serve: capture
             runs with capture_error_mode=thread_local so the copy thread's memcpys are legal meanwhile).
             Time spent in the join is logged as PM_SNAP_JOIN waited= (the "lazy" segment: waiting on state).
             The manifest must match the model after load_model (names, shapes, dtypes), else the engine fails.
  PM_SNAP_PIN=1 registers each shard's mmap with cudaHostRegister before its copies (DMA instead of staged
  pageable copies); PM_SNAP_THREADS copy threads, one CUDA stream each.
- KV geometry: pod_run.sh pins --kv-cache-memory-bytes to the value a profiled boot of the same engine shape
  logged (PM_KV_BYTES); that skips vLLM's CUDA-graph memory profiling (it captures every graph once more) and
  gives the same KV block count as the profiled boot.
- Warm hooks (Overlap), PM_WARM=1: the API server loads its tokenizer and detects the chat-template content
  format only after the engine is ready; start both in a thread when the engine core is launched.
- Weights URI (leave-one-out ablation of the snapshot only), PM_WEIGHTS_URI=gs://...: with
  --load-format runai_streamer and the weight-less model dir as the model, the stock Run:ai loader streams the
  safetensors from this URI instead of the model dir. The model path (part of vLLM's compile-cache key) stays
  the model dir, so the compile bank keyed on it still applies.
"""
import json
import multiprocessing.context
import os
import time

_logger = None


def _log():
    global _logger
    if _logger is None:
        from vllm.logger import init_logger
        _logger = init_logger('vllm.pm_cuda')
    return _logger


def register():
    if not any(k.startswith('PM_') for k in os.environ):
        return
    _patch_worker_wrapper()
    if os.environ.get('PM_ZYGOTE_SOCK'):
        _install_zygote_client(os.environ['PM_ZYGOTE_SOCK'])
    if os.environ.get('PM_WARM') == '1':
        _install_warm_hooks()


# -------------------------------------------------------------------------------------------- zygote

class ZygoteProcess(multiprocessing.context.ForkServerProcess):
    """A forkserver child gets the zygote's environment; apply the requesting parent's instead."""

    def start(self):
        self._pm_env = dict(os.environ)
        super().start()

    def run(self):
        os.environ.clear()
        os.environ.update(self._pm_env)
        super().run()


class ZygoteContext(multiprocessing.context.ForkServerContext):
    Process = ZygoteProcess


def _install_zygote_client(sock):
    import multiprocessing.forkserver as fsrv
    from multiprocessing import resource_tracker

    import vllm.utils.system_utils as su
    import vllm.v1.engine.utils as eu
    import vllm.v1.executor.multiproc_executor as me

    if getattr(eu.get_mp_context, '_pm', False):
        return
    srv = fsrv._forkserver
    srv._forkserver_address = sock
    if srv._forkserver_alive_fd is None:
        srv._forkserver_alive_fd = os.open(os.devnull, os.O_RDONLY)
    srv.ensure_running = resource_tracker.ensure_running  # never start a private forkserver
    ctx = ZygoteContext()

    def get_mp_context():
        return ctx

    get_mp_context._pm = True
    for mod in (su, eu, me):
        mod.get_mp_context = get_mp_context
    _log().info('PM_ZYGOTE client pid=%d sock=%s', os.getpid(), sock)


# ---------------------------------------------------------------------------------------------- warm

def _install_warm_hooks():
    import vllm.v1.engine.core_client as cc

    if getattr(cc.launch_core_engines, '_pm', False):
        return
    orig = cc.launch_core_engines

    def launch_core_engines(vllm_config, *args, **kwargs):
        _warm_in_thread(vllm_config)
        return orig(vllm_config, *args, **kwargs)

    launch_core_engines._pm = True
    cc.launch_core_engines = launch_core_engines


def _warm_in_thread(vllm_config):
    import threading

    def run():
        t0 = time.perf_counter()
        try:
            from vllm.renderers.hf import resolve_chat_template_content_format
            from vllm.tokenizers import cached_tokenizer_from_config
            mc = vllm_config.model_config
            tok = cached_tokenizer_from_config(mc)
            resolve_chat_template_content_format(None, None, 'auto', tok, model_config=mc)
            _log().info('PM_WARM api_server took=%.2fs', time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001  (a failed warm-up only means the later call pays)
            _log().warning('PM_WARM failed after %.2fs: %r', time.perf_counter() - t0, e)

    threading.Thread(target=run, name='pm-warm', daemon=True).start()


# ---------------------------------------------------------------------------------------------- gate

def _patch_worker_wrapper():
    from vllm.v1.worker import worker_base as wb

    orig = wb.WorkerWrapperBase.init_device
    if getattr(orig, '_pm', False):
        return

    def init_device(self):
        log = _log()
        gate = os.environ.get('PM_GATE_FILE')
        if gate:
            log.info('PM_PARKED pid=%d wall=%.3f', os.getpid(), time.time())
            while not os.path.exists(gate):
                time.sleep(0.01)
            log.info('PM_GATE_OPEN wall=%.3f', time.time())
        t0 = time.perf_counter()
        orig(self)
        log.info('PM_INIT_DEVICE took=%.2fs wall=%.3f', time.perf_counter() - t0, time.time())
        _patch_worker_class(type(self.worker))
        if os.environ.get('PM_SNAP_DIR') and os.environ.get('PM_SNAP_MODE') in ('save', 'restore'):
            _patch_snapshot()
        if os.environ.get('PM_WEIGHTS_URI'):
            _patch_weights_uri(os.environ['PM_WEIGHTS_URI'])

    init_device._pm = True
    wb.WorkerWrapperBase.init_device = init_device


def _patch_worker_class(cls):
    if getattr(cls.determine_available_memory, '_pm', False):
        return
    orig_dam = cls.determine_available_memory
    orig_warm = cls.compile_or_warm_up_model

    def determine_available_memory(self):
        t0 = time.perf_counter()
        n = orig_dam(self)
        _log().info('PM_KV_BYTES bytes=%d took=%.2fs pinned=%s', n, time.perf_counter() - t0,
                    bool(self.cache_config.kv_cache_memory_bytes))
        return n

    def compile_or_warm_up_model(self):
        out = orig_warm(self)
        _join_restore('after_warmup')  # PM_SNAP_JOIN=serve, or nothing captured: before the first request
        return out

    determine_available_memory._pm = compile_or_warm_up_model._pm = True
    cls.determine_available_memory = determine_available_memory
    cls.compile_or_warm_up_model = compile_or_warm_up_model


def _patch_weights_uri(uri):
    from vllm.model_executor.model_loader import runai_streamer_loader as rl

    orig = rl.RunaiModelStreamerLoader._prepare_weights
    if getattr(orig, '_pm', False):
        return

    def _prepare_weights(self, model_name_or_path, revision):
        files = orig(self, uri, revision)
        _log().info('PM_WEIGHTS_URI %s -> %s (%d safetensors)', model_name_or_path, uri, len(files))
        return files

    _prepare_weights._pm = True
    rl.RunaiModelStreamerLoader._prepare_weights = _prepare_weights


# ------------------------------------------------------------------------------------------ snapshot

_SNAP = {}


def _patch_snapshot():
    from vllm.model_executor.model_loader import default_loader as dl
    from vllm.v1.worker import gpu_model_runner as gmr

    if getattr(gmr.GPUModelRunner.load_model, '_pm', False):
        return
    mode, snap_dir = os.environ['PM_SNAP_MODE'], os.environ['PM_SNAP_DIR']
    orig_lw = dl.DefaultModelLoader.load_weights
    orig_lm = gmr.GPUModelRunner.load_model
    orig_capture = gmr.GPUModelRunner.capture_model
    orig_profile_cg = gmr.GPUModelRunner.profile_cudagraph_memory

    def profile_cudagraph_memory(self, *args, **kwargs):  # unpinned KV: this captures graphs too
        _join_restore('profile_cudagraph_memory')
        return orig_profile_cg(self, *args, **kwargs)

    def load_weights(self, model, model_config):
        if mode == 'restore':
            import torch
            t0 = time.perf_counter()
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()
            _SNAP['restore_pending'] = True
            _log().info('PM_SNAP restore: checkpoint skipped, %d parameters zero-filled in %.2fs',
                        sum(1 for _ in model.parameters()), time.perf_counter() - t0)
            return None
        return orig_lw(self, model, model_config)

    def load_model(self, *args, **kwargs):
        out = orig_lm(self, *args, **kwargs)
        model = self.get_model()
        if mode == 'save':
            _save(model, snap_dir)
        elif _SNAP.pop('restore_pending', False):
            _start_restore(model, snap_dir)
        bank = os.environ.get('PM_BANK_DONE')
        if bank:  # torch.compile reads the bank next: wait for its delivery (a miss would compile cold)
            waited = _wait(bank)
            _log().info('PM_BANK_WAIT waited=%.2fs', waited)
        return out

    def capture_model(self, *args, **kwargs):
        if os.environ.get('PM_SNAP_JOIN', 'capture') == 'capture':
            _join_restore('capture_model')
            return orig_capture(self, *args, **kwargs)
        import torch
        orig_graph_init = torch.cuda.graph.__init__

        def graph_init(g, cuda_graph, pool=None, stream=None, capture_error_mode='global'):
            orig_graph_init(g, cuda_graph, pool=pool, stream=stream, capture_error_mode='thread_local')

        torch.cuda.graph.__init__ = graph_init
        try:
            return orig_capture(self, *args, **kwargs)
        finally:
            torch.cuda.graph.__init__ = orig_graph_init

    load_weights._pm = load_model._pm = capture_model._pm = True
    dl.DefaultModelLoader.load_weights = load_weights
    gmr.GPUModelRunner.load_model = load_model
    gmr.GPUModelRunner.capture_model = capture_model
    gmr.GPUModelRunner.profile_cudagraph_memory = profile_cudagraph_memory


def _tensors(model):
    """name -> tensor: parameters, buffers (persistent or not), then any other CUDA tensor held as a module
    attribute (state derived during process_weights_after_loading); each storage view once."""
    import torch
    out, seen = {}, set()

    def add(name, t):
        key = (t.data_ptr(), t.dtype, tuple(t.shape), tuple(t.stride()))
        if name in out or key in seen or t.device.type != 'cuda':
            return
        seen.add(key)
        out[name] = t

    for name, t in list(model.named_parameters(remove_duplicate=True)) + list(model.named_buffers()):
        add(name, t)
    for mname, mod in model.named_modules():
        for attr, v in vars(mod).items():
            if isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Parameter) and attr not in (
                    '_parameters', '_buffers'):
                add(f'attr:{mname}.{attr}', v)
    return out


def _save(model, snap_dir):
    import torch
    log = _log()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    used0 = torch.cuda.mem_get_info()
    os.makedirs(snap_dir, exist_ok=True)
    limit = int(float(os.environ.get('PM_SNAP_SHARD_MB', '2048')) * (1 << 20))
    tensors = _tensors(model)
    entries, shards, k, off, f = [], [], 0, 0, None
    for name, t in tensors.items():
        c = t.detach().cpu().contiguous()  # no device temporaries: the stock memory profile follows
        nbytes = c.numel() * c.element_size()
        if f is None or (off and off + nbytes > limit):
            if f is not None:
                f.close()
                shards.append({'shard': k, 'bytes': off})
                k += 1
            f, off = open(os.path.join(snap_dir, f'shard{k}.bin'), 'wb'), 0
        pad = (-off) % 4096
        if pad:
            f.write(b'\0' * pad)
            off += pad
        if nbytes:
            f.write(c.reshape(-1).view(torch.uint8).numpy().data)
        entries.append({'name': name, 'dtype': str(c.dtype).removeprefix('torch.'), 'shape': list(c.shape),
                        'stride': list(t.stride()), 'shard': k, 'offset': off, 'nbytes': nbytes,
                        'kind': 'param' if isinstance(t, torch.nn.Parameter) else (
                            'attr' if name.startswith('attr:') else 'buffer')})
        off += nbytes
    f.close()
    del c
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    used1 = torch.cuda.mem_get_info()
    shards.append({'shard': k, 'bytes': off})
    total = sum(s['bytes'] for s in shards)
    manifest = {'format': 1, 'stage': 'after_load_model', 'bytes': total, 'shards': shards, 'tensors': entries}
    with open(os.path.join(snap_dir, 'manifest.json.tmp'), 'w') as fh:
        json.dump(manifest, fh)
    os.replace(os.path.join(snap_dir, 'manifest.json.tmp'), os.path.join(snap_dir, 'manifest.json'))
    for s in shards:
        open(os.path.join(snap_dir, f'shard{s["shard"]}.done'), 'w').close()
    dt = time.perf_counter() - t0
    kinds = {}
    for e in entries:
        kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
    log.info('PM_SNAP_SAVE tensors=%d %s shards=%d bytes=%d took=%.2fs (%.2f GB/s) gpu_free %d -> %d attrs=%s',
             len(entries), kinds, len(shards), total, dt, total / dt / 1e9, used0[0], used1[0],
             [e['name'] for e in entries if e['kind'] == 'attr'][:20])


def _wait(path):
    t0 = time.perf_counter()
    while not os.path.exists(path):
        time.sleep(0.005)
    return time.perf_counter() - t0


def _start_restore(model, snap_dir):
    """Check the manifest against the model, then copy shard by shard (each once delivered) on a thread."""
    import threading
    import torch
    log = _log()
    t0 = time.perf_counter()
    waited = _wait(os.path.join(snap_dir, 'manifest.json'))
    with open(os.path.join(snap_dir, 'manifest.json')) as fh:
        man = json.load(fh)
    tensors = _tensors(model)
    names = [e['name'] for e in man['tensors']]
    problems = []
    if set(names) != set(tensors):
        problems.append(f'tensor names differ: missing={sorted(set(tensors) - set(names))[:5]} '
                        f'extra={sorted(set(names) - set(tensors))[:5]}')
    else:
        for e in man['tensors']:
            t = tensors[e['name']]
            if list(t.shape) != e['shape'] or str(t.dtype).removeprefix('torch.') != e['dtype']:
                problems.append(f"{e['name']}: {list(t.shape)} {t.dtype} vs {e['shape']} {e['dtype']}")
                break
    if problems:
        raise RuntimeError('PM_SNAP: snapshot does not match the model after load_model: ' + '; '.join(problems))
    by_shard = {}
    for e in man['tensors']:
        by_shard.setdefault(e['shard'], []).append(e)
    threads = max(1, int(os.environ.get('PM_SNAP_THREADS', '2')))
    pin = os.environ.get('PM_SNAP_PIN') == '1'
    device = next(iter(tensors.values())).device
    state = {'t0': t0, 'bytes': man['bytes'], 'errors': [], 'shard_wait': 0.0, 'copy': 0.0, 'pin': 0.0,
             'done_at': None, 'shards': len(man['shards']), 'finished': 0, 'lock': threading.Lock()}

    def copy_shard(k, stream):
        s = next(x for x in man['shards'] if x['shard'] == k)
        path = os.path.join(snap_dir, f'shard{k}.bin')
        state['shard_wait'] += _wait(os.path.join(snap_dir, f'shard{k}.done'))
        flat = torch.from_file(path, shared=True, size=s['bytes'], dtype=torch.uint8) if s['bytes'] else None
        t1 = time.perf_counter()
        registered = False
        if pin and flat is not None:
            rc = torch.cuda.cudart().cudaHostRegister(flat.data_ptr(), s['bytes'], 0)
            registered = int(rc) == 0 if not isinstance(rc, tuple) else int(rc[0]) == 0
            if not registered:
                _log().warning('PM_SNAP shard%d cudaHostRegister rc=%s; pageable copy', k, rc)
        t2 = time.perf_counter()
        with torch.cuda.stream(stream), torch.no_grad():
            for e in by_shard.get(k, []):
                if not e['nbytes']:
                    continue
                dst = tensors[e['name']]
                src = flat[e['offset']:e['offset'] + e['nbytes']].view(getattr(torch, e['dtype']))
                src = src.reshape(e['shape'])
                if dst.is_contiguous():
                    dst.copy_(src, non_blocking=registered)
                else:
                    dst.copy_(src.to(device, non_blocking=registered))
        stream.synchronize()
        t3 = time.perf_counter()
        if registered:
            torch.cuda.cudart().cudaHostUnregister(flat.data_ptr())
        state['pin'] += t2 - t1
        state['copy'] += t3 - t2
        del flat

    def run(worker, ks):
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device=device)
        try:
            for k in ks:
                copy_shard(k, stream)
        except BaseException as e:  # noqa: BLE001  (re-raised by the join)
            state['errors'].append(repr(e))
            _log().error('PM_SNAP restore thread %d: %r', worker, e)
        finally:
            with state['lock']:
                state['finished'] += 1
                last = state['finished'] == threads
            if last:  # when the weights are complete, against the engine's phases (e.g. CUDA graph capture)
                state['done_at'] = time.perf_counter()
                _log().info('PM_SNAP_COPY_DONE after=%.2fs wall=%.3f', state['done_at'] - t0, time.time())

    order = [s['shard'] for s in man['shards']]  # delivered in this order
    ths = [threading.Thread(target=run, args=(i, order[i::threads]), name=f'pm-snap-{i}', daemon=True)
           for i in range(threads)]
    for th in ths:
        th.start()
    state['threads'] = ths
    _SNAP['restore'] = state
    log.info('PM_SNAP_RESTORE started: manifest waited %.2fs, %d tensors in %d shards (%d bytes), %d threads, pin=%s',
             waited, len(names), len(order), man['bytes'], threads, pin)


def _join_restore(where):
    st = _SNAP.pop('restore', None)
    if not st:
        return
    import torch
    t0 = time.perf_counter()
    for th in st['threads']:
        th.join()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    if st['errors']:
        raise RuntimeError(f"PM_SNAP restore failed: {st['errors'][:3]}")
    _log().info('PM_SNAP_JOIN at=%s waited=%.2fs restore_wall=%.2fs (%.2f GB/s) shard_wait=%.2fs copy=%.2fs pin=%.2fs '
                'wall=%.3f', where, t1 - t0, t1 - st['t0'], st['bytes'] / (t1 - st['t0']) / 1e9, st['shard_wait'],
                st['copy'], st['pin'], time.time())
