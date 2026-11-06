"""TPUFlex on vLLM-JAX (tpu-inference) on a GPU: engine-side mechanisms, installed by the same post-import hooks
as the GPU patches (patches.PATCHES). With no PM_* variable set nothing here is patched, so the stock arms run the
stock engine.

- Pre-spawn gate (Decompose + Overlap), PM_GATE_FILE: the engine core finishes its imports, vLLM and tpu-inference
  construction, then parks in TPUWorker.init_device, right before its first device touch (jax.devices()). It
  continues once the file exists. On a fresh host the file is the delivery's "compile cache landed" marker (device
  init can already compile, and the JAX cache must not be read while it is still being extracted); on a same-host
  restart (warm-pm) the driver creates it after the old engine has exited.
- Terminal weight snapshot (Cache), PM_SNAP_DIR + PM_SNAP_MODE, taken at the end of tpu-inference's
  model_loader._get_nnx_model: after the loader's host transposes, fp32 round trips, conversion programs and
  create_jit_model, i.e. the exact device arrays the engine serves from.
    save     stock _get_nnx_model, then every leaf of nnx.state(model) is copied to the host and written to
             ~1 GiB chunk files c<NNNN>.raw (raw bytes, bf16 bit-exact); manifest.json is written last.
    restore  nnx.eval_shape builds the abstract model (no weights), each leaf is device_put from the chunk that
             holds it, and nnx.update installs them; the checkpoint is never read. Chunks may still be
             arriving: a chunk counts once its final name exists (the delivery renames it into place), so the
             restore streams behind the delivery. Paths, shapes and dtypes must match the manifest exactly, else
             the engine fails (there is no checkpoint to fall back to).
  PM_SNAP_WAIT_S (default 900) bounds the wait for the manifest and every chunk; PM_SNAP_SOURCE (read: each chunk
  read into anonymous memory, the default; mmap: np.memmap views) and PM_SNAP_THREADS (default 32) device_put threads.
- KV geometry, PM_KV_BYTES: tpu-inference sizes the KV cache from live HBM usage after load_model, and the KV
  block count is in every backbone program's shapes, hence in the JAX cache key. The PM arms pin the bytes that
  the seed boot measured, so the snapshot boots reproduce the seed's KV geometry exactly. Every PM boot logs
  "PM_KV_BYTES measured=... returned=...".
- Executable bank (Cache) is the JAX persistent compilation cache with every program persisted
  (JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0, JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1, set by run_arm.sh);
  nothing to patch here.
"""
from __future__ import annotations

import json
import multiprocessing.context
import os
import time
from pathlib import Path

CHUNK_BYTES = 1 << 30
MANIFEST = "manifest.json"


def _say(msg: str) -> None:
    print(f"[pm] {msg} wall={time.time():.3f} pid={os.getpid()}", flush=True)


def active() -> bool:
    return any(os.environ.get(k) for k in ("PM_GATE_FILE", "PM_SNAP_MODE", "PM_KV_BYTES", "PM_ZYGOTE"))


# ------------------------------------------------------------------------------------------------ zygote
# Decompose: `vllm serve` spawns the engine core with the `spawn` method (the CLI sets it), and that process then
# re-imports vLLM, tpu-inference, JAX and flax (~15 s) after the API server has done the same. With PM_ZYGOTE=1 the
# API server's sitecustomize starts a multiprocessing forkserver at interpreter start, before its own imports; the
# forkserver preloads the engine core's modules in parallel with them, and vLLM's engine core is forked from it.
# The fork needs a process that never called cuInit. vLLM's import of quant_utils asks the platform for the device
# name (FP8_DTYPE), which on this stack is jax.devices() (the API server holds a CUDA context from then on); the
# forkserver is started with PM_ZYGOTE_SERVER=1, so patches._device_kind answers through NVML there instead.
ZYGOTE_PRELOAD = ["jax", "flax.nnx", "vllm.v1.engine.core", "vllm.v1.executor.uniproc_executor",
                  "tpu_inference.worker.tpu_worker", "tpu_inference.runner.tpu_runner",
                  "tpu_inference.models.common.model_loader", "tpu_inference.models.jax.gemma4"]
# The forkserver's environment at interpreter start (sitecustomize imports this module before the preload).
_SERVER_START_ENV = dict(os.environ) if os.environ.get("PM_ZYGOTE_SERVER") == "1" else None
# Import effects that prepend instead of set: tpu_inference.env_override puts "--xla_cpu_max_isa=AVX2 " (and a libtpu
# flag) in front of the inherited value, so a spawned engine core, which inherits the API server's already-prefixed
# value, ends with the prefix twice. Both variables are hashed into every JAX persistent-cache key.
PREPENDED_BY_IMPORT = ("XLA_FLAGS", "LIBTPU_INIT_ARGS")


def start_zygote() -> None:
    """Called by hetero_jax_gpu.install() in every process; acts only in the `vllm serve` API server."""
    import sys
    if os.environ.get("PM_ZYGOTE") != "1" or os.path.basename(sys.argv[0] if sys.argv else "") != "vllm" \
            or "serve" not in sys.argv[1:2]:
        return
    import multiprocessing
    import multiprocessing.forkserver as forkserver
    multiprocessing.set_forkserver_preload(ZYGOTE_PRELOAD)
    os.environ["PM_ZYGOTE_SERVER"] = "1"  # inherited by the forkserver only; its children get the API server's env
    try:
        forkserver.ensure_running()
    finally:
        del os.environ["PM_ZYGOTE_SERVER"]
    _say("PM_ZYGOTE started")


def spawn_equivalent_env(parent: dict) -> None:
    """Give this forkserver child the environment a spawned engine core has: the API server's environment at
    start() (vLLM sets the engine core's variables in the API server right before), then the engine core's own
    import effects. Set-style effects (e.g. MOSAIC_GPU_* from the JAX CUDA plugin) come from the forkserver's
    post-preload environment; tpu_inference.env_override is re-run on top of the parent's values, as a spawned
    process's import of it would. Without this the zygote's XLA_FLAGS differ from a spawned engine's and every
    persistent-cache lookup misses a cache seeded by a spawned engine (zygote dev run 2026-10-24: 4 hits, 215 misses)."""
    import importlib
    import sys
    start = _SERVER_START_ENV or {}
    effects = {k: v for k, v in os.environ.items() if start.get(k) != v
               and k not in PREPENDED_BY_IMPORT and k != "PM_ZYGOTE_SERVER"}
    os.environ.clear()
    os.environ.update(parent)
    os.environ.update(effects)
    override = sys.modules.get("tpu_inference.env_override")
    if override is not None:
        importlib.reload(override)


class ZygoteProcess(multiprocessing.context.ForkServerProcess):
    """A forkserver child carries the forkserver's environment; see spawn_equivalent_env. Pickled to the forkserver
    by reference, so it lives at module level."""

    def start(self):
        self._pm_env = dict(os.environ)
        super().start()

    def run(self):
        spawn_equivalent_env(self._pm_env)
        _say(f"PM_ZYGOTE child running XLA_FLAGS={os.environ.get('XLA_FLAGS')!r}")
        super().run()


class ZygoteContext(multiprocessing.context.ForkServerContext):
    Process = ZygoteProcess


def patch_system_utils(system_utils, done) -> None:
    if os.environ.get("PM_ZYGOTE") != "1":
        return
    import multiprocessing.forkserver as forkserver
    if forkserver._forkserver._forkserver_pid is None:  # not the API server (or it never started one)
        return
    original = system_utils.get_mp_context

    def get_mp_context():
        original()  # keeps vLLM's checks and env side effects
        return ZygoteContext()

    system_utils.get_mp_context = get_mp_context
    done("vllm.utils.system_utils.get_mp_context -> PM zygote (forkserver)")


# ------------------------------------------------------------------------------------------- worker hooks

def patch_worker(tpu_worker, done) -> None:
    if not active():
        return
    cls = tpu_worker.TPUWorker
    init_device, dam = cls.init_device, cls.determine_available_memory

    def gated_init_device(self, *args, **kwargs):
        gate = os.environ.get("PM_GATE_FILE")
        if gate:
            _say(f"PM_PARKED gate={gate}")
            Path(gate + ".parked").touch()  # run_arm.sh (warm-pm) retires the old engine once this exists
            while not os.path.exists(gate):
                time.sleep(0.01)
            _say("PM_GATE_OPEN")
        t0 = time.perf_counter()
        out = init_device(self, *args, **kwargs)
        _say(f"PM_INIT_DEVICE took={time.perf_counter() - t0:.3f}")
        return out

    def pinned_dam(self):
        measured = dam(self)
        pinned = os.environ.get("PM_KV_BYTES")
        returned = int(pinned) if pinned else measured
        _say(f"PM_KV_BYTES measured={measured} returned={returned}")
        return returned

    cls.init_device = gated_init_device
    cls.determine_available_memory = pinned_dam
    done("tpu_inference TPUWorker: PM gate before init_device, PM_KV_BYTES")


def patch_model_loader(model_loader, done) -> None:
    mode = os.environ.get("PM_SNAP_MODE")
    if mode not in ("save", "restore"):
        return
    directory = Path(os.environ["PM_SNAP_DIR"])
    original = model_loader._get_nnx_model

    def get_nnx_model(model_class, vllm_config, rng, mesh, pooler=None, is_draft_model=False):
        if is_draft_model:
            return original(model_class, vllm_config, rng, mesh, pooler=pooler, is_draft_model=is_draft_model)
        if mode == "save":
            model = original(model_class, vllm_config, rng, mesh, pooler=pooler, is_draft_model=is_draft_model)
            save(directory, model)
            return model
        return restore_model(directory, model_class, vllm_config, rng, mesh)

    model_loader._get_nnx_model = get_nnx_model
    done(f"tpu_inference model_loader._get_nnx_model: PM snapshot {mode} ({directory})")


# --------------------------------------------------------------------------------------------- snapshot

def _leaves(tree):
    import jax
    return [(jax.tree_util.keystr(p), x) for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]]


def _is_key(dtype) -> bool:
    import jax
    return jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key)


def save(directory: Path, model) -> dict:
    """Write every leaf of nnx.state(model) as raw bytes into ~1 GiB chunks, manifest last."""
    import jax
    import numpy as np
    from flax import nnx

    t0 = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST).unlink(missing_ok=True)
    leaves = _leaves(nnx.state(model))
    rows, chunk, chunk_bytes, total = [], 0, 0, 0
    handle = None
    for path, x in leaves:
        if not isinstance(x, jax.Array):
            raise RuntimeError(f"PM snapshot: {path} is {type(x).__name__}, not a jax.Array")
        key_impl = None
        if _is_key(x.dtype):
            key_impl = str(jax.random.key_impl(x))
            x = jax.random.key_data(x)
        if handle is not None and chunk_bytes and chunk_bytes + x.nbytes > CHUNK_BYTES:
            handle.close()
            handle, chunk, chunk_bytes = None, chunk + 1, 0
        if handle is None:
            handle = open(directory / f"c{chunk:04d}.raw", "wb")
        host = np.ascontiguousarray(jax.device_get(x))
        handle.write(host.reshape(-1).view(np.uint8).data)
        # The sharding kind matters beyond placement: it is lowered into every program that takes the weights,
        # so a different kind (e.g. NamedSharding(mesh, P()) for a SingleDeviceSharding) changes the persistent-cache
        # key of the backbone. Keep the exact kind.
        spec = list(x.sharding.spec) if hasattr(x.sharding, "spec") else []
        rows.append({"path": path, "shape": list(x.shape), "dtype": str(x.dtype), "key_impl": key_impl,
                     "sharding": type(x.sharding).__name__, "sharding_repr": repr(x.sharding)[:200],
                     "memory_kind": x.sharding.memory_kind,
                     "spec": [list(s) if isinstance(s, tuple) else s for s in spec],
                     "chunk": chunk, "offset": chunk_bytes, "bytes": int(host.nbytes)})
        chunk_bytes += host.nbytes
        total += host.nbytes
    if handle is not None:
        handle.close()
    manifest = {"version": 1, "arrays": rows, "chunks": chunk + 1, "bytes": total,
                "chunk_bytes": [sum(r["bytes"] for r in rows if r["chunk"] == c) for c in range(chunk + 1)],
                "jax": jax.__version__, "devices": [d.device_kind for d in jax.devices()]}
    tmp = directory / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest))
    os.replace(tmp, directory / MANIFEST)
    _say(f"PM_SNAP_SAVED arrays={len(rows)} chunks={chunk + 1} bytes={total} took={time.perf_counter() - t0:.2f}")
    return manifest


def _wait(path: Path, deadline: float) -> float:
    t0 = time.perf_counter()
    while not path.exists():
        if time.time() > deadline:
            raise TimeoutError(f"PM snapshot: {path} never arrived")
        time.sleep(0.005)
    return time.perf_counter() - t0


def restore_model(directory: Path, model_class, vllm_config, rng, mesh):
    import threading
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from jax.sharding import NamedSharding, PartitionSpec, SingleDeviceSharding
    from tpu_inference.models.jax.utils.qwix.qwix_utils import apply_qwix_on_abstract_model

    if vllm_config.model_config.quantization is not None or apply_qwix_on_abstract_model(vllm_config):
        raise RuntimeError("PM snapshot restore: quantized models are not supported")
    deadline = time.time() + float(os.environ.get("PM_SNAP_WAIT_S", "900"))
    t0 = time.perf_counter()
    with jax.set_mesh(mesh):
        abstract = nnx.eval_shape(lambda: model_class(vllm_config, rng, mesh), graph_updates=False)
    t_graph = time.perf_counter() - t0
    waited = _wait(directory / MANIFEST, deadline)
    manifest = json.loads((directory / MANIFEST).read_text())
    state = nnx.state(abstract)
    leaves = _leaves(state)
    treedef = jax.tree_util.tree_structure(state)
    rows = manifest["arrays"]
    if len(rows) != len(leaves):
        raise RuntimeError(f"PM snapshot: {len(rows)} arrays in the manifest, the model has {len(leaves)}")
    for (path, a), r in zip(leaves, rows):
        dtype = str(a.dtype) if r["key_impl"] is None else None
        if path != r["path"] or (dtype is not None and (list(a.shape) != r["shape"] or dtype != r["dtype"])):
            raise RuntimeError(f"PM snapshot: {path} {getattr(a, 'shape', None)} {a.dtype} does not match {r}")
    stats = {"wait_s": waited, "place_s": 0.0}
    # H100 probe (results/gpu-jax/dev-pm/restore_bench*.jsonl, 20 GB): device_put from np.memmap views stays at
    # ~3.4 GB/s whatever the thread count; reading each chunk into anonymous memory first with 32 threads in flight
    # across chunks reaches 5.8-6.0 GB/s; pinned_host staging gives 4.3-4.4 GB/s.
    source = os.environ.get("PM_SNAP_SOURCE", "read")  # read | mmap
    threads = int(os.environ.get("PM_SNAP_THREADS", "32"))
    locks = {r["chunk"]: threading.Lock() for r in rows}
    left = Counter(r["chunk"] for r in rows)
    hosts, waits = {}, {}

    def chunk_host(c):
        with locks[c]:
            if c not in hosts:
                path = directory / f"c{c:04d}.raw"
                waits[c] = _wait(path, deadline)  # chunks may still be arriving (delivery renames them in)
                hosts[c] = (np.memmap(path, mode="r", dtype=np.uint8) if source == "mmap"
                            else np.fromfile(path, dtype=np.uint8))
            return hosts[c]

    def done_with(c):
        with locks[c]:  # the last array of a chunk drops the chunk; a transfer keeps its own host reference
            left[c] -= 1
            if not left[c]:
                hosts.pop(c, None)

    single = SingleDeviceSharding(mesh.devices.flat[0])
    default_kind = os.environ.get("PM_SNAP_SHARDING", "NamedSharding")  # manifests written before "sharding" existed

    def place(r):
        raw = chunk_host(r["chunk"])[r["offset"]:r["offset"] + r["bytes"]]
        host = raw.view(np.dtype(jnp.dtype(r["dtype"]))).reshape(r["shape"])
        kind = r.get("sharding", default_kind)
        if kind == "SingleDeviceSharding":
            sharding = single
        elif kind == "NamedSharding":
            sharding = NamedSharding(mesh, PartitionSpec(*(tuple(s) if isinstance(s, list) else s for s in r["spec"])))
        else:
            raise RuntimeError(f"PM snapshot: {r['path']} has a {kind}, which restore does not rebuild")
        x = jax.device_put(host, sharding)
        done_with(r["chunk"])
        if r["key_impl"] is not None:
            x = jax.random.wrap_key_data(x, impl=r["key_impl"])
        return x

    t1 = time.perf_counter()
    # All arrays in the delivery's order (chunk, offset), `threads` in flight across chunks: chunk reads (or page
    # faults) overlap H2D copies.
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["chunk"], rows[i]["offset"]))
    arrays = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for i, x in zip(order, pool.map(lambda i: place(rows[i]), order)):
            arrays[i] = x
    jax.block_until_ready(arrays)
    stats["place_s"] = time.perf_counter() - t1
    stats["wait_s"] += sum(waits.values())
    hosts.clear()
    nnx.update(abstract, jax.tree_util.tree_unflatten(treedef, arrays))
    _say(f"PM_SNAP_RESTORED arrays={len(rows)} bytes={manifest['bytes']} source={source} threads={threads} "
         f"graph={t_graph:.2f} wait={stats['wait_s']:.2f} place={stats['place_s']:.2f} "
         f"total={time.perf_counter() - t0:.2f}")
    return abstract
