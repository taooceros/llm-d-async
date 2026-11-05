#!/usr/bin/env python3
"""ServerlessLLM (Fu et al., OSDI'24) checkpoint loading for CUDA vLLM v0.23.0: the sllm-* arms' vLLM plugin and
the sllm-seed tools.

Source: ServerlessLLM @ 9f50241baa53 (github.com/ServerlessLLM/ServerlessLLM, sllm-store 0.8.0). ServerlessLLM
plugs into vLLM with sllm_store/vllm_patch/sllm_load.patch, which
  (1) adds vllm/model_executor/model_loader/sllm_loader.py (ServerlessLLMLoader),
  (2) registers it as load format "serverless_llm" in model_loader/__init__.py,
  (3) adds save_serverless_llm_state plumbing (EngineCore -> executor -> Worker -> GPUModelRunner ->
      ServerlessLLMLoader.save_model).
Here, without patching the image's vLLM:
  (1) sllm_loader.py next to this file is the patch's new file, byte for byte. It is executed with the PORT_FIXES
      below applied to its source text (two fixes, each asserted to match exactly where expected);
  (2) register() (a vllm.general_plugins entry point, SLLM_PLUGIN=1) registers it through vLLM's public
      register_model_loader when vllm.model_executor.model_loader is imported, in whichever process imports it.
      A process that never loads a model (the API server) imports nothing extra;
  (3) `save` runs ServerlessLLMLoader.save_model on the worker's model through LLM.collective_rpc, which is
      what the patch's Worker/GPUModelRunner methods do.
The registered class only adds log lines around the loader's load_dict call (SLLM_LOAD_DICT: wall time, tensors,
bytes) and load_model (SLLM_LOAD_MODEL); the loading itself is ServerlessLLMLoader.load_model.

The loader reads the checkpoint through a running sllm-store server (127.0.0.1:8073): load_dict asks it to
read <STORAGE_PATH>/<model>/rank_<r> into its pinned-memory pool (O_DIRECT where the filesystem allows it,
--num-thread I/O threads, --chunk-size chunks) and to copy the chunks into GPU memory the engine allocated and
shared over CUDA IPC, pipelined with the reads. After the copy the model stays in the pool (ServerlessLLM's DRAM
tier), so a later load of the same model on that node is served from pinned memory.

    sllm_vllm.py save --model /model --delete-source        in place: /model/rank_0/{tensor.data_*,tensor_index.json}
    sllm_vllm.py upload DIR gs://bucket/prefix               every file under DIR (GCS_TOKEN: host-minted token)
    sllm_vllm.py bench --model-path model/rank_0             one load_dict through the running store, then exit
"""
import argparse
import importlib.abc
import importlib.util
import json
import os
import sys
import time

LOAD_FORMAT = "serverless_llm"
# (old, new, occurrences) applied to sllm_loader.py's source before it is executed; the file stays verbatim
PORT_FIXES = (
    # vLLM 0.23's Gemma-4 registers every decoder-layer parameter under two module paths (model.layers.<i> and
    # model.self_decoder.decoder_layers.<i>). The loader keys the model by _filter_subtensors(state_dict()), which
    # keeps the lexicographically larger alias (self_decoder...), but matches parameters by named_parameters(),
    # which yields only the first alias: 901 of 902 keys never matched ("Missing keys", 2026-10-29). Iterating
    # every alias matches each kept key once; the aliases share one Parameter, so rebinding it rebinds all.
    ("model.named_parameters(recurse=True)", "model.named_parameters(recurse=True, remove_duplicate=False)", 2),
    # The loader rebinds parameters only, so every persistent buffer of the state dict is "missing": in vLLM 0.23
    # each Attention's _k_scale/_v_scale/_q_scale/_prob_scale (registered as buffers) and Gemma-4's per-layer
    # layer_scalar, which comes from the checkpoint. save_model wrote the whole state dict, buffers included:
    # rebind them from the loaded state the same way (the loader's last loop then moves the non-persistent ones).
    ("            if state_dict:\n                raise ValueError(",
     "            for key, buf in model.named_buffers(recurse=True, remove_duplicate=False):\n"
     "                if key in state_dict:\n"
     "                    buf.data = sllm_state_dict[key]\n"
     "                    state_dict.pop(key)\n"
     "            if state_dict:\n                raise ValueError(", 1),
)
# pod_run.sh ENGINE_ARGS, minus serving-only flags: the module tree (and so the state dict) depends on these
SAVE_ENGINE_KW = dict(max_model_len=8192, max_num_seqs=512, max_num_batched_tokens=2048, gpu_memory_utilization=0.92,
                      limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0}, seed=0)


def _log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------------- plugin

def _loader_module():
    """sllm_loader.py with PORT_FIXES applied, as module "sllm_loader"."""
    import types
    if getattr(sys.modules.get("sllm_loader"), "_port_fixed", False):
        return sys.modules["sllm_loader"]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sllm_loader.py")
    with open(path) as f:
        src = f.read()
    for old, new, n in PORT_FIXES:
        if src.count(old) != n:
            raise RuntimeError(f"sllm_loader.py: {old!r} occurs {src.count(old)} times, expected {n}")
        src = src.replace(old, new)
    mod = types.ModuleType("sllm_loader")
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    mod._port_fixed = True
    sys.modules["sllm_loader"] = mod
    return mod


def _timed_loader_class():
    ServerlessLLMLoader = _loader_module().ServerlessLLMLoader

    class TimedServerlessLLMLoader(ServerlessLLMLoader):
        """ServerlessLLMLoader; logs how long its load_dict call took (the rest of load_model is CPU-side model
        init and parameter rebinding)."""

        def load_model(self, *, vllm_config, **kwargs):
            import sllm_store.torch as st
            inner = st.load_dict

            def load_dict(model_path, device_map, storage_path=None):
                t = time.time()
                out = inner(model_path, device_map, storage_path)
                dt = time.time() - t
                n = sum(x.numel() * x.element_size() for x in out.values())
                unique = sum({x.data_ptr(): x.numel() * x.element_size() for x in out.values()}.values())
                _log(f"SLLM_LOAD_DICT took={dt:.3f}s tensors={len(out)} bytes={unique} tensor_bytes={n} "
                     f"GBps={unique / max(dt, 1e-9) / 1e9:.2f} model={model_path}")
                return out

            st.load_dict = load_dict
            t = time.time()
            try:
                return super().load_model(vllm_config=vllm_config, **kwargs)
            finally:
                st.load_dict = inner
                _log(f"SLLM_LOAD_MODEL took={time.time() - t:.3f}s")

    return TimedServerlessLLMLoader


def _register_into(mod):
    if getattr(mod, "_sllm_registered", False):
        return
    mod.register_model_loader(LOAD_FORMAT)(_timed_loader_class())
    mod._sllm_registered = True
    _log(f"SLLM_PLUGIN registered load_format={LOAD_FORMAT} pid={os.getpid()}")


class _OnImport(importlib.abc.MetaPathFinder):
    """Registers the loader right after vllm.model_executor.model_loader has been executed."""
    NAME = "vllm.model_executor.model_loader"

    def find_spec(self, name, path, target=None):
        if name != self.NAME:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_register(module):
            exec_module(module)
            _register_into(module)

        spec.loader.exec_module = exec_and_register
        return spec


def register():
    if os.environ.get("SLLM_PLUGIN") != "1":
        return
    mod = sys.modules.get(_OnImport.NAME)
    if mod is not None and hasattr(mod, "register_model_loader"):
        _register_into(mod)
    elif not any(isinstance(f, _OnImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _OnImport())


# ------------------------------------------------------------------------------------------------- seed

def _save_rank(worker, path):
    """The patch's Worker/GPUModelRunner.save_serverless_llm_state: ServerlessLLMLoader.save_model(model, path)."""
    ServerlessLLMLoader = _loader_module().ServerlessLLMLoader
    runner = getattr(worker, "worker", worker).model_runner
    t = time.time()
    ServerlessLLMLoader.save_model(runner.model, path)
    return time.time() - t


def cmd_save(a):
    """ServerlessLLM's vLLM writer (sllm_store/cli.py VllmModelDownloader._run_writer): load the checkpoint with
    the stock loader, then save the engine's model in ServerlessLLM format under <out>/rank_<r>. Differences:
    dtype bfloat16 (the CLI hard-codes float16; every other arm serves the bf16 checkpoint in bf16) and the
    serving engine's shape flags (the writer uses max_model_len=1 and one KV block), so that the saved state
    dict is the served model's."""
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # collective_rpc with a function
    from sllm_vllm import _save_rank as save_rank  # by module name (the plugin dir is on PYTHONPATH): cloudpickle
    from vllm import LLM                           # then sends a reference, not __main__'s code
    t0 = time.time()
    llm = LLM(model=a.model, dtype=a.dtype, enforce_eager=True, **SAVE_ENGINE_KW)
    t_loaded = time.time()
    removed = 0
    if a.delete_source:  # the weights are on the GPU now; frees the tmpfs before the CPU copy + write
        for name in sorted(os.listdir(a.model)):
            if os.path.splitext(name)[1] in (".bin", ".pt", ".safetensors"):
                removed += os.path.getsize(os.path.join(a.model, name))
                os.remove(os.path.join(a.model, name))
    took = llm.collective_rpc(save_rank, args=(a.out,))
    rank0 = os.path.join(a.out, "rank_0")
    index = json.load(open(os.path.join(rank0, "tensor_index.json")))
    files = {n: os.path.getsize(os.path.join(rank0, n)) for n in sorted(os.listdir(rank0))}
    data = sum(v for n, v in files.items() if n.startswith("tensor.data_"))
    check = index_check(index, data)
    _log("SLLM_SAVE " + json.dumps({"model": a.model, "out": a.out, "dtype": a.dtype, "load_s": round(t_loaded - t0, 1),
                                    "save_s": [round(x, 1) for x in took], "source_bytes_removed": removed,
                                    "tensors": len(index), "files": files, "bytes": sum(files.values()),
                                    "index_check": check}))
    if check["overlaps"] or check["end"] > data:
        raise SystemExit(f"tensor_index.json is inconsistent with the data files: {check}")


def index_check(index, data_bytes):
    """The writer's records must not overlap and must lie inside the tensor.data_* files (sllm-store 0.8.0's
    writer silently dropped a tensor > 2 GiB and wrote the next one over its offset; sllm_store_fixes.patch)."""
    spans = sorted((off, off + size, name) for name, (off, size, *_rest) in index.items())
    overlaps = [(a[2], b[2]) for a, b in zip(spans, spans[1:]) if b[0] < a[1]]
    return {"records": len(spans), "record_bytes": sum(e - s for s, e, _ in spans), "end": spans[-1][1] if spans else 0,
            "data_bytes": data_bytes, "overlaps": overlaps[:5], "n_overlaps": len(overlaps)}


def cmd_upload(a):
    from google.cloud import storage
    from google.cloud.storage import transfer_manager as tm
    bucket_name, _, prefix = a.dst.removeprefix("gs://").partition("/")
    creds = None
    if os.environ.get("GCS_TOKEN"):  # node SA scopes are devstorage.read_only: a host-minted user token
        from google.oauth2.credentials import Credentials
        creds = Credentials(os.environ["GCS_TOKEN"])
    bucket = storage.Client(credentials=creds, project="sysres-disagg-ml").bucket(bucket_name)
    t0, total, n = time.time(), 0, 0
    for root, _, names in os.walk(a.dir):
        for name in sorted(names):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, a.dir)
            blob = bucket.blob(f"{prefix.rstrip('/')}/{rel}")
            size = os.path.getsize(path)
            if size > 512 << 20:
                tm.upload_chunks_concurrently(path, blob, chunk_size=128 << 20, max_workers=a.workers,
                                              worker_type=tm.THREAD)
            else:
                blob.upload_from_filename(path)
            blob.reload()
            if blob.size != size:
                raise RuntimeError(f"{rel}: uploaded {blob.size} B != {size} B")
            total, n = total + size, n + 1
    dt = time.time() - t0
    _log("UPLOAD " + json.dumps({"dst": a.dst, "files": n, "bytes": total, "seconds": round(dt, 1),
                                 "GBps": round(total / dt / 1e9, 2)}))


def cmd_bench(a):
    """One load_dict (what ServerlessLLMLoader does after its CPU-side model init) through the running store."""
    import torch
    from sllm_store.torch import load_dict
    torch.cuda.init()
    t = time.time()
    sd = load_dict(a.model_path, {"": 0}, a.storage_path)
    torch.cuda.synchronize()
    dt = time.time() - t
    n = sum(x.numel() * x.element_size() for x in sd.values())
    _log("SLLM_BENCH " + json.dumps({"label": a.label, "seconds": round(dt, 3), "tensors": len(sd), "bytes": n,
                                     "GBps": round(n / dt / 1e9, 2)}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save")
    s.add_argument("--model", required=True)
    s.add_argument("--out", help="default: --model (in place)")
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--delete-source", action="store_true")
    s = sub.add_parser("upload")
    s.add_argument("dir")
    s.add_argument("dst")
    s.add_argument("--workers", type=int, default=16)
    s = sub.add_parser("bench")
    s.add_argument("--model-path", default="model/rank_0")
    s.add_argument("--storage-path", default="/")
    s.add_argument("--label", default="")
    a = p.parse_args()
    if a.cmd == "save":
        a.out = a.out or a.model
    {"save": cmd_save, "upload": cmd_upload, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
