#!/usr/bin/env python3
"""Fresh-host wipe on a reused GPU node (user decision 2026-10-24: node reuse, no GKE stages on GPU). Runs first
thing in a new pod (pod_run_jax.sh, WIPE=1), before the pod's own setup, so the setup and the timed run start from
the state a fresh node would have, with a cold image.

Pod-scoped state is new by construction (new container, new emptyDirs: $HOME caches = JAX persistent cache, XLA
autotune, ~/.triton, ~/.nv/ComputeCache, vLLM cache root; /model tmpfs checkpoint copy; /pmstate; /dev/shm; /tmp
and Ray's session dir in the container layer). The driver deletes the previous pod and waits until the API object
is gone, which kubelet allows only after its volumes are torn down. This script checks that and evicts what
survives a pod: the node's page cache.
  1. before: file count/bytes of every pod-scoped path, /proc/meminfo (node-wide: Shmem = every tmpfs on the
     node, so a surviving checkpoint copy shows there), mincore residency of the image trees;
  2. posix_fadvise(DONTNEED) over every file of the image trees (venv /opt/venv, the interpreter's stdlib, the
     node's GPU driver libraries), then sync and drop_caches=3 (the pod is privileged), i.e. the whole node's
     clean page cache, dentries and inodes (checkpoint pages of a gcsfuse mount or a deleted tmpfs included);
  3. after: the same measurements.
Prints one line "WIPE <json receipt>"; exits non-zero if the eviction did not happen.
"""
import ctypes
import ctypes.util
import json
import mmap
import os
import sys
import time

TREES = ["/opt/venv", "/usr/local/lib/python3.12", "/usr/local/nvidia/lib64"]
POD_PATHS = [os.environ.get("HOME", "/cache/home"), "/model", "/pmstate", "/dev/shm", "/tmp"]
MEMINFO = ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "Shmem", "Dirty")
PAGE = mmap.PAGESIZE

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]


def files(d):
    out = []
    for root, _, names in os.walk(d):
        out += [p for p in (os.path.join(root, n) for n in names) if os.path.isfile(p) and not os.path.islink(p)]
    return out


def resident_bytes(path):
    """mincore over a MAP_SHARED, PROT_READ mapping (maps without faulting pages in)."""
    size = os.path.getsize(path)
    if size == 0:
        return 0, 0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            return 0, size
        try:
            pages = (size + PAGE - 1) // PAGE
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(addr, size, vec) != 0:
                return 0, size
            return (pages - bytes(vec).count(0)) * PAGE, size
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


def residency():
    out = {}
    for d in TREES:
        res = tot = n = 0
        for p in files(d):
            try:
                r, s = resident_bytes(p)
            except OSError:
                continue
            res, tot, n = res + r, tot + s, n + 1
        out[d] = {"files": n, "bytes": tot, "resident_bytes": res}
    return out


def meminfo():
    vals = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        if k in MEMINFO:
            vals[k + "_kB"] = int(v.split()[0])
    return vals


def pod_state():
    out = {}
    for d in POD_PATHS:
        if not os.path.isdir(d):
            out[d] = None
            continue
        fs = files(d)
        out[d] = {"files": len(fs), "bytes": sum(os.path.getsize(p) for p in fs if os.path.exists(p)),
                  "entries": sorted(os.listdir(d))[:20]}
    return out


def evict():
    n = 0
    for d in TREES:
        for p in files(d):
            try:
                fd = os.open(p, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
    os.sync()
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        dropped = True
    except OSError as exc:
        dropped = f"failed: {exc!r}"
    return n, dropped


def main():
    t0 = time.time()
    receipt = {"at": t0, "host": os.uname().nodename, "node": os.environ.get("NODE_NAME"),
               "pod_paths_before": pod_state(), "meminfo_before": meminfo(), "residency_before": residency()}
    t1 = time.time()
    n, dropped = evict()
    receipt["evicted"] = {"fadvise_dontneed_files": n, "trees": TREES, "drop_caches_3": dropped,
                          "seconds": round(time.time() - t1, 2)}
    receipt["meminfo_after"] = meminfo()
    receipt["residency_after"] = residency()
    receipt["seconds"] = round(time.time() - t0, 2)
    print("WIPE " + json.dumps(receipt), flush=True)
    return 0 if dropped is True else 1


if __name__ == "__main__":
    sys.exit(main())
