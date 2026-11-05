"""Run the vLLM TPU stack (vllm-tpu + tpu-inference, JAX) on NVIDIA GPUs.

`install()` registers post-import hooks: each patch runs right after its
target module executes, before any other module can bind names from it, in
every process that imports `sitecustomize` from this directory (the API
server, the engine core, Ray workers). Every patch is listed in `PATCHES`.

The environment is the stock PyPI release (see image/requirements.lock); nothing
here edits installed files.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

_installed = False


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, hooks):
        self.hooks = hooks  # module name -> callable(module)

    def find_spec(self, name, path=None, target=None):
        hook = self.hooks.get(name)
        if hook is None:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            hook(module)

        loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Idempotent; a no-op unless HETERO_JAX_GPU=1."""
    global _installed
    if _installed or os.environ.get("HETERO_JAX_GPU") != "1":
        return
    _installed = True
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")  # cpu: tpu-inference reshapes weights on a CPU mesh
    # Triton writes source line info (absolute paths) into the PTX that jax-triton embeds in the custom
    # call; each Ray working_dir upload has a new path, so the JAX persistent-cache key of every
    # executable with attention changed per job (tests/cache_key_probe.py). No line info: stable keys.
    os.environ.setdefault("TRITON_DISABLE_LINE_INFO", "1")
    from hetero_jax_gpu import patches, pm

    hooks = dict(patches.PATCHES)
    sys.meta_path.insert(0, _PostImportFinder(hooks))
    for name, hook in hooks.items():  # already imported (e.g. by site hooks)
        if name in sys.modules:
            hook(sys.modules[name])
    pm.start_zygote()  # PM_ZYGOTE=1 in the `vllm serve` process only; before its heavy imports
