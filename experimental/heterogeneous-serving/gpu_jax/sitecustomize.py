# Imported by every Python process that has gpu_jax/ on PYTHONPATH (the Ray job
# entrypoint exports it, so the API server and the spawned engine core both get
# it). Installs the GPU patches when HETERO_JAX_GPU=1; otherwise does nothing.
try:
    import hetero_jax_gpu
except ImportError:
    pass
else:
    hetero_jax_gpu.install()
