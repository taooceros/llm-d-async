"""Post-import patches: the TPU assumptions tpu-inference 0.32.0 makes outside
its kernels, each replaced by the GPU equivalent. Keyed by module name; each
hook runs right after that module executes (see `install`)."""

from __future__ import annotations

import glob
import logging
import os
import socket

log = logging.getLogger("hetero_jax_gpu")
APPLIED: list[str] = []


def _done(name: str) -> None:
    APPLIED.append(name)
    if os.environ.get("HETERO_JAX_GPU_VERBOSE") == "1":
        print(f"[hetero_jax_gpu] patched {name} (pid {os.getpid()})", flush=True)


def _num_gpus() -> int:
    return len(glob.glob("/dev/nvidia[0-9]*"))


def _device_kind(device_id: int = 0) -> str:
    """jax.devices()[i].device_kind, except in the PM zygote (forkserver, PM_ZYGOTE_SERVER=1; pm.start_zygote) while
    no JAX backend exists: the zygote must fork with CUDA untouched (a child of a process that called cuInit fails
    with CUDA_ERROR_NOT_INITIALIZED), and vLLM's import of quant_utils already asks for the device name
    (FP8_DTYPE). NVML gives the same string (cuDeviceGetName; checked on H100) without a CUDA context."""
    import jax
    from jax._src import xla_bridge
    if os.environ.get("PM_ZYGOTE_SERVER") == "1" and not xla_bridge._backends:
        import ctypes
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        handle, name = ctypes.c_void_p(), ctypes.create_string_buffer(96)
        if nvml.nvmlInit_v2() or nvml.nvmlDeviceGetHandleByIndex_v2(device_id, ctypes.byref(handle)) \
                or nvml.nvmlDeviceGetName(handle, name, 96):
            raise RuntimeError("PM zygote: NVML device name lookup failed")
        nvml.nvmlShutdown()
        return name.value.decode()
    return jax.devices()[device_id].device_kind


# jax: tpu-inference jits its step / sampling / decode-loop functions with
# `xla_tpu_*` compiler options, which the GPU compiler rejects. Drop them.
def patch_jax_compiler(compiler) -> None:
    get_compile_options = compiler.get_compile_options

    def gpu_compile_options(*args, env_options_overrides=None, **kwargs):
        if env_options_overrides:
            env_options_overrides = {
                k: v for k, v in env_options_overrides.items()
                if not k.startswith("xla_tpu_")
            }
        return get_compile_options(*args,
                                   env_options_overrides=env_options_overrides,
                                   **kwargs)

    compiler.get_compile_options = gpu_compile_options
    _done("jax._src.compiler.get_compile_options (drop xla_tpu_* options)")


# vLLM: libtpu is installed (tpu-inference needs it importable), so the TPU
# plugin activates and selects tpu-inference's TpuPlatform. Pin the CUDA
# plugin off so a stray pynvml can never make two builtin plugins active.
def patch_vllm_platforms(platforms) -> None:
    platforms.builtin_platform_plugins["cuda"] = lambda: None
    _done("vllm.platforms: cuda plugin off, tpu plugin -> TpuPlatform")


# tpu-inference validates JAX_PLATFORMS against ['', 'tpu', 'cpu', 'proxy'] and
# raises on 'cuda'. Accept 'cuda,cpu' (the CPU backend is needed too: weights are
# reshaped on a CPU mesh before device_put). Nothing else in tpu-inference
# branches on the value besides the 'proxy' check.
def patch_tpu_envs(envs) -> None:
    validated = envs.environment_variables["JAX_PLATFORMS"]

    def jax_platforms() -> str:
        value = os.getenv("JAX_PLATFORMS")
        parts = set((value or "").lower().split(","))
        if parts & {"cuda", "gpu"} and parts <= {"cuda", "gpu", "cpu"}:
            return value
        return validated()

    envs.environment_variables["JAX_PLATFORMS"] = jax_platforms
    _done("tpu_inference.envs: JAX_PLATFORMS accepts cuda")



# tpu-inference topology probes read the GCE TPU metadata server and
# /dev/accel*; answer them for one host with N GPUs.
def patch_tpu_info(tpu_info) -> None:
    tpu_info.get_tpu_type = lambda: "gpu"
    tpu_info.get_node_name = socket.gethostname
    tpu_info.get_node_worker_id = lambda: 0
    tpu_info.get_num_cores_per_chip = lambda: 1
    tpu_info.get_num_chips = _num_gpus
    tpu_info.get_tpu_vmem_size_bytes = lambda: 0
    _done("tpu_inference.tpu_info (GPU topology)")


def patch_utils(utils) -> None:
    import jax
    import numpy as np
    from jax.sharding import Mesh

    get_device_name = utils.get_device_name
    make_optimized_mesh = utils.make_optimized_mesh

    def device_name(num_devices=None):
        kind = _device_kind(0)
        if "TPU" in kind:
            return get_device_name(num_devices)
        return kind if num_devices is None else f"{kind}-{num_devices}"

    def device_hbm_limit() -> int:
        return jax.local_devices()[0].memory_stats()["bytes_limit"]

    def mesh(axis_shapes, axis_names, *, devices=None):
        devices = list(jax.devices() if devices is None else devices)
        if all(hasattr(d, "coords") for d in devices):
            return make_optimized_mesh(axis_shapes, axis_names, devices=devices)
        devices.sort(key=lambda d: d.id)
        return Mesh(np.asarray(devices).reshape(tuple(axis_shapes)), tuple(axis_names))

    utils.get_device_name = device_name
    utils.get_device_hbm_limit = device_hbm_limit
    utils.make_optimized_mesh = mesh
    _done("tpu_inference.utils (device name, HBM limit, mesh without coords)")


# Attention: the Pallas/Mosaic TPU RPA kernel -> Triton unified attention.
def patch_rpa_kernel(kernel) -> None:
    from hetero_jax_gpu import rpa_triton
    kernel.ragged_paged_attention = rpa_triton.ragged_paged_attention
    _done("tpu_inference RPA v3 kernel -> Triton unified attention")


def patch_attention_interface(attention_interface) -> None:
    from hetero_jax_gpu import rpa_triton
    attention_interface.ragged_paged_attention = rpa_triton.ragged_paged_attention
    _done("attention_interface.ragged_paged_attention -> Triton unified attention")


def patch_tpu_platform(tpu_platform) -> None:
    platform = tpu_platform.TpuPlatform

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return _device_kind(device_id)

    platform.get_device_name = get_device_name
    _done("TpuPlatform.get_device_name -> GPU device kind")


# Observability (every run): JAX's monitoring events (trace / lower / backend compile durations, persistent-cache
# hits, misses and retrieval times), summed per event over load_model and over the warm-up, one line each:
#   [hetero_jax_gpu] JAX_EVENTS <phase> {"event": [count, seconds], ...}
# vLLM reconfigures logging, so JAX's own debug log (JAX_DEBUG_LOG_MODULES) never reaches the job log.
_EVENTS: dict = {}


def _jax_events_install() -> None:
    import jax

    def on_duration(event, duration, **kwargs):
        c = _EVENTS.setdefault(event, [0, 0.0])
        c[0] += 1
        c[1] += duration

    def on_event(event, **kwargs):
        _EVENTS.setdefault(event, [0, 0.0])[0] += 1

    jax.monitoring.register_event_duration_secs_listener(on_duration)
    jax.monitoring.register_event_listener(on_event)


def _jax_events_flush(phase: str) -> None:
    import json
    body = {k: [v[0], round(v[1], 3)] for k, v in sorted(_EVENTS.items())}
    _EVENTS.clear()
    print(f"[hetero_jax_gpu] JAX_EVENTS {phase} {json.dumps(body)} (pid {os.getpid()})", flush=True)


def patch_worker(tpu_worker) -> None:
    cls = tpu_worker.TPUWorker
    load_model, warm_up = cls.load_model, cls.compile_or_warm_up_model

    def logged_load_model(self, *args, **kwargs):
        _jax_events_flush("before_load_model")
        try:
            return load_model(self, *args, **kwargs)
        finally:
            _jax_events_flush("load_model")

    def logged_warm_up(self, *args, **kwargs):
        _jax_events_flush("before_warm_up")
        try:
            return warm_up(self, *args, **kwargs)
        finally:
            _jax_events_flush("warm_up")

    _jax_events_install()
    cls.load_model, cls.compile_or_warm_up_model = logged_load_model, logged_warm_up
    _done("tpu_inference TPUWorker: JAX compile/cache event accounting")
    # TPUFlex (pm.py): no-op unless a PM_* variable is set.
    from hetero_jax_gpu import pm
    pm.patch_worker(tpu_worker, _done)


def patch_model_loader(model_loader) -> None:
    """Observability: the torch thread count the weight loader runs with. vLLM's "Reducing Torch threads" line
    never appears on this stack (only GPUWorker / MultiprocExecutor call set_torch_threads_for_runtime; tpu-inference
    runs TPUWorker under UniProcExecutor), so the Ray OMP_NUM_THREADS=1 inflation is checked with this line."""
    get_model = model_loader.get_model

    def logged_get_model(*args, **kwargs):
        import torch
        print(f"[hetero_jax_gpu] load_model torch_threads={torch.get_num_threads()} "
              f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} cpus={len(os.sched_getaffinity(0))} "
              f"(pid {os.getpid()})", flush=True)
        return get_model(*args, **kwargs)

    model_loader.get_model = logged_get_model
    _done("tpu_inference model_loader.get_model: log torch threads")
    from hetero_jax_gpu import pm
    pm.patch_model_loader(model_loader, _done)


def patch_system_utils(system_utils) -> None:
    from hetero_jax_gpu import pm
    pm.patch_system_utils(system_utils, _done)


PATCHES = {
    "vllm.utils.system_utils": patch_system_utils,
    "jax._src.compiler": patch_jax_compiler,
    "vllm.platforms": patch_vllm_platforms,
    "tpu_inference.envs": patch_tpu_envs,
    "tpu_inference.tpu_info": patch_tpu_info,
    "tpu_inference.utils": patch_utils,
    "tpu_inference.kernels.ragged_paged_attention.v3.kernel": patch_rpa_kernel,
    "tpu_inference.layers.common.attention_interface": patch_attention_interface,
    "tpu_inference.platforms.tpu_platform": patch_tpu_platform,
    "tpu_inference.worker.tpu_worker": patch_worker,
    "tpu_inference.models.common.model_loader": patch_model_loader,
}
