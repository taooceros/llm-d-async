#!/usr/bin/env python3
"""Fresh-host emulation on a reused node (runs in the pod before T0); prints one "WIPE {json}" receipt.

Pod-local state is new with every pod: $HOME (compile caches: vLLM torch.compile/AOT/Inductor, Triton, FlashInfer,
CUDA ComputeCache) and /cache live in an emptyDir, the checkpoint/snapshot dirs (/model, /dev/shm) in memory
emptyDirs, /tmp in the container's fresh overlay layer. This script checks that they are empty, checks that
the GPU is free (no process of an earlier pod holds memory), then drops the node's clean page cache
(sync; echo 3 > /proc/sys/vm/drop_caches, privileged container), which evicts the image's files (torch, vLLM,
Triton, FlashInfer, CUDA libraries, Python) and anything a previous run read. Page-cache residency (mincore) of
those trees is measured before and after.
"""
import ctypes
import json
import mmap
import os
import subprocess
import sys
import time

EMPTY_DIRS = ("/cache/home", "/cache", "/model", "/dev/shm")


def residency(roots):
    """(resident bytes, total bytes, files) of every regular file under roots, via mmap + mincore (libc)."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    page = os.sysconf('SC_PAGE_SIZE')
    res = tot = n = 0
    for root in roots:
        for d, _, names in os.walk(root):
            for name in names:
                p = os.path.join(d, name)
                try:
                    if os.path.islink(p) or not os.path.isfile(p):
                        continue
                    size = os.path.getsize(p)
                    if not size:
                        continue
                    fd = os.open(p, os.O_RDONLY)
                except OSError:
                    continue
                try:
                    addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
                    if addr in (None, ctypes.c_void_p(-1).value):
                        continue
                    vec = ctypes.create_string_buffer((size + page - 1) // page)
                    if libc.mincore(addr, size, vec) == 0:
                        res += sum(b & 1 for b in vec.raw) * page
                    libc.munmap(addr, size)
                finally:
                    os.close(fd)
                tot += size
                n += 1
    return {'resident_bytes': min(res, tot), 'bytes': tot, 'files': n}


def meminfo():
    out = {}
    for line in open('/proc/meminfo'):
        k, v = line.split(':')
        if k in ('MemFree', 'Cached', 'Shmem', 'Buffers'):
            out[k + '_GB'] = round(int(v.split()[0]) * 1024 / 1e9, 2)
    return out


def image_roots():
    import importlib.util
    roots = []
    for mod in ('torch', 'vllm', 'flashinfer', 'triton', 'nvidia', 'transformers'):
        spec = importlib.util.find_spec(mod)
        if spec and spec.submodule_search_locations:
            roots.append(list(spec.submodule_search_locations)[0])
    return roots + [p for p in ('/usr/local/nvidia/lib64', os.path.dirname(os.__file__)) if os.path.isdir(p)]


def drop_only():
    """--drop-only (sllm arms, right before T0): the store daemon has started from a private copy of the
    site-packages; drop the page cache again and report the image trees' residency (pages that a live process
    maps stay resident)."""
    t0 = time.time()
    roots = image_roots()
    before = {'meminfo': meminfo(), 'image': residency(roots)}
    os.sync()
    with open('/proc/sys/vm/drop_caches', 'w') as f:
        f.write('3\n')
    after = {'meminfo': meminfo(), 'image': residency(roots)}
    print('WIPE_T0 ' + json.dumps({'roots': roots, 'before': before, 'after': after,
                                   'seconds': round(time.time() - t0, 2)}), flush=True)
    return 0


def main():
    if sys.argv[1:] == ['--drop-only']:
        return drop_only()
    t0 = time.time()
    roots = image_roots()
    leftovers = {}
    for d in EMPTY_DIRS:
        if os.path.isdir(d):
            items = [n for n in os.listdir(d) if n not in ('home', 'lost+found')]
            if d == '/cache':  # store-site: the sllm store daemon's private site-packages, copied by this pod
                items = [n for n in items if n not in ('home', 'store-site')]
            if items:
                leftovers[d] = items[:20]
    gpu = subprocess.run([sys.executable, '-c', 'import torch; f, t = torch.cuda.mem_get_info(); print(f, t)'],
                         capture_output=True, text=True)
    free, total = (int(x) for x in gpu.stdout.split()) if gpu.returncode == 0 else (None, None)
    before = {'meminfo': meminfo(), 'image': residency(roots)}
    os.sync()
    dropped = True
    try:
        with open('/proc/sys/vm/drop_caches', 'w') as f:
            f.write('3\n')
    except OSError as e:
        dropped = repr(e)
    after = {'meminfo': meminfo(), 'image': residency(roots)}
    receipt = {'empty_dirs_leftovers': leftovers, 'gpu_free_bytes': free, 'gpu_total_bytes': total,
               'drop_caches': dropped, 'roots': roots, 'before': before, 'after': after,
               'seconds': round(time.time() - t0, 2)}
    print('WIPE ' + json.dumps(receipt), flush=True)
    ok = (not leftovers and dropped is True and free is not None and free > 0.95 * total)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
