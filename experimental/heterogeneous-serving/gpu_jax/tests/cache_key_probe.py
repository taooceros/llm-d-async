"""Does a jit that calls the Triton RPA get the same JAX persistent-cache key when hetero_jax_gpu is
imported from a different directory (each Ray working_dir upload lands in a new path)?

    python tests/cache_key_probe.py CACHE_DIR    # run from two copies of gpu_jax/, same CACHE_DIR

Prints "PROBE <n cache entries> <seconds to compile>"; a stable key adds no entries on the second run.
"""
import functools
import os
import sys
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", sys.argv[1])
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from hetero_jax_gpu.rpa_triton import ragged_paged_attention  # noqa: E402

q = jnp.zeros((64, 32, 256), jnp.bfloat16)
k = jnp.zeros((64, 16, 256), jnp.bfloat16)
cache = jnp.zeros((64, 16, 16, 2, 256), jnp.bfloat16)
i32 = functools.partial(jnp.zeros, dtype=jnp.int32)
fn = jax.jit(functools.partial(ragged_paged_attention, sm_scale=0.0625, sliding_window=1024))
t = time.time()
fn.lower(q, k, k, cache, i32(8), i32(8 * 16), i32(9), i32(3)).compile()
print("PROBE", len([f for f in os.listdir(sys.argv[1]) if f.endswith("-cache")]), round(time.time() - t, 2),
      "TRITON_DISABLE_LINE_INFO=" + os.environ.get("TRITON_DISABLE_LINE_INFO", ""), flush=True)
