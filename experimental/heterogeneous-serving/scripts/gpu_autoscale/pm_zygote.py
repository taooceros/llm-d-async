#!/usr/bin/env python3
"""TPUFlex zygote for CUDA vLLM: a pre-imported fork server for the EngineCore process.

Stock vLLM (this image) starts the EngineCore with the `spawn` method: a fresh interpreter that imports torch,
vLLM, Triton and FlashInfer again after the API server did (~8 s on a fresh node). The zygote starts at T0 next
to the API server with the same environment, imports the engine's stack while the API server imports its own,
then serves fork requests over CPython's multiprocessing.forkserver protocol. With PM_ZYGOTE_SOCK set, the plugin
(pm_cuda) points vLLM's get_mp_context() at it. Nothing here initializes CUDA (the child does, after the fork).

Fork safety: the engine's import graph initializes the CUDA driver in three places (found by fork-testing a child
CUDA context after every import, vLLM 0.23.0 / torch 2.11 / FlashInfer 0.6.12 / nvidia-cutlass-dsl image), and a
child forked after cuInit fails with "CUDA driver initialization failed":
  - FlashInfer's CompilationContext queries the device capability at import unless FLASHINFER_CUDA_ARCH_LIST is set;
  - torch.cuda.is_available() (torch._inductor.compile_fx at import) calls cudaGetDeviceCount unless
    PYTORCH_NVML_BASED_CUDA_CHECK=1 (torch's documented fork-safe check);
  - CuTe DSL (cutlass, imported by FlashInfer) evaluates detect_gpu_arch() eagerly as the default argument of its
    <prefix>_ARCH lookup (env_manager), i.e. cuInit + cuDeviceGetAttribute even when CUTE_DSL_ARCH is set.
So the zygote sets the first two variables for itself, and replaces cutlass.base_dsl.runtime.cuda's
get_compute_capability_major_minor (right after that module executes, before env_manager binds it) with one that
returns PM_ZYGOTE_ARCH (default 9.0, the H100). These give the values autodetection gives on this GPU (FlashInfer
workspace "90a", CuTe "sm_90a"). The forked engine applies the API server's environment (plugin: ZygoteProcess).
After the imports the zygote checks that the CUDA driver is still uninitialized (cuDeviceGetCount -> 3).

Env: PM_ZYGOTE_SOCK (unix socket path), PM_ZYGOTE_PRELOAD (comma-separated modules, optional), PM_ZYGOTE_ARCH.
The socket listens before the imports start, so an early fork request simply waits for them.
"""
import ctypes
import importlib
import importlib.abc
import os
import signal
import socket
import sys
import time

PRELOAD = (
    'vllm.entrypoints.cli.main',
    'vllm.v1.engine.core',
    'vllm.v1.engine.utils',
    'vllm.v1.executor.uniproc_executor',
    'vllm.v1.executor.multiproc_executor',
    'vllm.v1.worker.gpu_worker',
    'vllm.v1.worker.gpu_model_runner',
    'vllm.compilation.backends',
    'vllm.compilation.decorators',
    'vllm.compilation.cuda_graph',
    'vllm.model_executor.model_loader',
    'vllm.model_executor.models.gemma4',
    'vllm.model_executor.models.gemma4_mm',
    'vllm.v1.attention.backends.triton_attn',
    'vllm.v1.sample.sampler',
    'vllm.v1.structured_output',
    'vllm.multimodal.registry',
    'torch._inductor.codecache',
    'torch._inductor.compile_fx',
    'torch._dynamo',
    'triton',
    'flashinfer',
    'flashinfer.sampling',
)


class _PatchAfterExec(importlib.abc.MetaPathFinder):
    """Run patch(module) right after the named module executes, before any importer binds its names."""

    def __init__(self, patches):
        self.patches = patches

    def find_spec(self, name, path, target=None):
        if name not in self.patches:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        orig_exec, patch = spec.loader.exec_module, self.patches[name]

        def exec_module(module):
            orig_exec(module)
            patch(module)

        spec.loader.exec_module = exec_module
        return spec


def _arch():
    major, _, minor = os.environ.get('PM_ZYGOTE_ARCH', '9.0').partition('.')
    return int(major), int(minor or 0)


def _make_fork_safe():
    os.environ.setdefault('FLASHINFER_CUDA_ARCH_LIST', '%d.%d' % _arch())
    os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK', '1')

    def cutlass_cuda(module):
        module.get_compute_capability_major_minor = lambda *a, **k: _arch()

    sys.meta_path.insert(0, _PatchAfterExec({'cutlass.base_dsl.runtime.cuda': cutlass_cuda}))


def _driver_initialized():
    """cuDeviceGetCount returns CUDA_ERROR_NOT_INITIALIZED (3) until someone in this process called cuInit; the
    probe itself initializes nothing."""
    try:
        cu = ctypes.CDLL('libcuda.so.1')
    except OSError:
        return False
    n = ctypes.c_int()
    return cu.cuDeviceGetCount(ctypes.byref(n)) != 3


def main():
    t0 = time.time()
    _make_fork_safe()
    ppid = os.getppid()
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG: die with the launcher
    if os.getppid() != ppid:
        return
    path = os.environ['PM_ZYGOTE_SOCK']
    if os.path.exists(path):
        os.unlink(path)
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(path)
    os.chmod(path, 0o600)
    listener.listen(64)
    print(f'PM_ZYGOTE listening pid={os.getpid()} sock={path}', flush=True)
    mods = [m for m in os.environ.get('PM_ZYGOTE_PRELOAD', ','.join(PRELOAD)).split(',') if m]
    failed = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001  (a missing optional module only costs its import later)
            failed.append(f'{m}: {e!r}'[:200])
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    import torch
    assert not torch.cuda.is_initialized() and not _driver_initialized(), 'the zygote must not initialize CUDA'
    print(f'PM_ZYGOTE_READY took={time.time() - t0:.2f}s modules={len(mods)} failed={failed}', flush=True)
    alive_r, _alive_w = os.pipe()  # the write end stays open: the zygote lives until its parent dies
    from multiprocessing import forkserver
    forkserver.main(listener.detach(), alive_r, [])


if __name__ == '__main__':
    main()
