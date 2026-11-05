"""Environment check on a GPU node (GPU_JAX_GOAL.md, step 1).

    python -m hetero_jax_gpu.envcheck      # HETERO_JAX_GPU=1, gpu_jax/ on PYTHONPATH

Checks, printing one JSON summary line ("ENVCHECK {...}"), exit 1 on failure:
installed versions == image/requirements.lock; jax sees CUDA devices; a jax-triton
kernel runs; a jit with xla_tpu_* compiler options compiles (patch); vLLM resolves
tpu-inference's TpuPlatform; tpu-inference's attention entry point is the Triton RPA.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import re
import sys
from pathlib import Path

LOCK = Path(__file__).resolve().parent.parent / "image" / "requirements.lock"


def lock_mismatches() -> dict:
    bad = {}
    for line in LOCK.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s;]+)", line)
        if not m:
            continue
        name, want = m.group(1), m.group(3)
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            have = None
        if have != want:
            bad[name] = {"lock": want, "installed": have}
    return bad


def main() -> int:
    out: dict = {"python": sys.version.split()[0]}
    out["lock_mismatches"] = lock_mismatches()

    import jax
    import jax.numpy as jnp
    devices = jax.devices()
    out["jax"] = jax.__version__
    out["devices"] = [f"{d.platform}:{d.device_kind}" for d in devices]
    out["bytes_limit_gib"] = round(devices[0].memory_stats()["bytes_limit"] / 2**30, 2)

    import jax_triton as jt
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, o_ptr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(o_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))

    x = jnp.arange(1024, dtype=jnp.float32)
    y = jt.triton_call(x, 2 * x, kernel=add_kernel, out_type=jax.typeof(x),
                       grid=(1024 // 128,), BLOCK=128)
    out["jax_triton_add_ok"] = bool(jnp.all(y == 3 * x))

    f = jax.jit(lambda a: a @ a.T, compiler_options={
        "xla_tpu_all_gather_collective_matmul_mode": "post_spmd_conservative"})
    out["xla_tpu_options_dropped_ok"] = float(f(jnp.ones((8, 8)))[0, 0]) == 8.0

    from vllm.platforms import current_platform
    out["vllm_platform"] = f"{type(current_platform).__module__}.{type(current_platform).__name__}"
    from tpu_inference.layers.common import attention_interface
    from hetero_jax_gpu import patches, rpa_triton
    out["rpa_is_triton"] = attention_interface.ragged_paged_attention is rpa_triton.ragged_paged_attention
    from tpu_inference import utils
    out["device_name"] = utils.get_device_name()
    out["mesh"] = str(utils.make_optimized_mesh((1, 1), ("data", "model")).shape)
    out["patches"] = patches.APPLIED

    ok = (not out["lock_mismatches"] and out["devices"][0].startswith("gpu:")
          and out["jax_triton_add_ok"] and out["xla_tpu_options_dropped_ok"]
          and out["vllm_platform"].endswith("TpuPlatform") and out["rpa_is_triton"])
    out["ok"] = ok
    print("ENVCHECK " + json.dumps(out), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
