#!/usr/bin/env python3
"""Read-pattern micro-benchmark on the pod's gcsfuse mount (stdlib only; runs on a TPU host, no TPU use).

Why does the default vLLM loader get only tens of MB/s over /models? Each case reads a fresh, page-cache-evicted
region of one safetensors file with one access pattern and records, as before/after deltas:
  wall time, host RX (/proc/net/dev; hostNetwork pod), and the gcsfuse sidecar's Prometheus counters
  (FUSE ReadFile ops + latency, GCS reads/readers/download bytes split by Sequential/Random).
Cases: {mmap page-fault walk, 1 MiB pread} x {1 process, 4 processes on the same region (TP workers each walking
the whole checkpoint), 4 processes on disjoint quarters}.
"""
import json
import mmap
import multiprocessing as mp
import os
import re
import sys
import time
import urllib.request

FILE = sys.argv[1] if len(sys.argv) > 1 else '/models/gemma-4-31b/model-00001-of-00002.safetensors'
REGION = int(os.environ.get('FB_REGION_MIB', '2048')) << 20
SPACING = int(os.environ.get('FB_SPACING_GIB', '4')) << 30
CAP_S = float(os.environ.get('FB_CAP_S', '240'))
STEP = 1 << 20
METRICS = 'http://127.0.0.1:62990/metrics'
KEEP = re.compile(r'^(fs_ops_count|fs_ops_latency_sum|fs_ops_latency_count|gcs_download_bytes_count|gcs_read_bytes_count|'
                  r'gcs_read_count|gcs_reader_count)(\{[^}]*\})? ')


def metrics():
    with urllib.request.urlopen(METRICS, timeout=10) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if KEEP.match(line) and ('fs_op=' not in line or 'ReadFile' in line):
            k, v = line.rsplit(' ', 1)
            out[k] = float(v)
    return out


def rx():
    total = 0
    with open('/proc/net/dev') as f:
        for line in f.readlines()[2:]:
            name, _, rest = line.partition(':')
            if name.strip() != 'lo':
                total += int(rest.split()[0])
    return total


def resident(lo, hi):
    """Page-cache-resident bytes of [lo, hi) (mincore on a fresh MAP_SHARED mapping; does not fault pages in)."""
    import ctypes
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    fd = os.open(FILE, os.O_RDONLY)
    try:
        n = hi - lo
        a = libc.mmap(None, n, 1, 1, fd, lo)  # PROT_READ, MAP_SHARED; lo is page aligned
        if a in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), 'mmap')
        pages = (n + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        vec = (ctypes.c_ubyte * pages)()
        if libc.mincore(a, n, vec) != 0:
            raise OSError(ctypes.get_errno(), 'mincore')
        libc.munmap(a, n)
        return (pages - bytes(vec).count(0)) * mmap.PAGESIZE
    finally:
        os.close(fd)


def evict():
    fd = os.open(FILE, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def worker(method, lo, hi, start, q):
    fd = os.open(FILE, os.O_RDONLY)
    done = 0
    try:
        start.wait()
        t0 = time.perf_counter()
        if method == 'mmap':
            mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)  # same as safetensors' lazy mmap: data arrives via page faults
            off = lo
            while off < hi and time.perf_counter() - t0 < CAP_S:
                n = min(STEP, hi - off)
                _ = mm[off:off + n]  # copy out of the mapping (faults the pages in)
                off += n
                done += n
            mm.close()
        else:
            off = lo
            while off < hi and time.perf_counter() - t0 < CAP_S:
                n = len(os.pread(fd, min(STEP, hi - off), off))
                if n == 0:
                    break
                off += n
                done += n
        q.put({'lo': lo, 'bytes': done, 'seconds': round(time.perf_counter() - t0, 2)})
    finally:
        os.close(fd)


def case(name, method, nproc, layout, idx):
    base = SPACING * idx
    lo, hi = base, base + REGION
    evict()
    time.sleep(2)
    res0 = resident(lo, hi)
    if layout == 'same':
        spans = [(lo, hi)] * nproc
    else:
        q = REGION // nproc
        spans = [(lo + i * q, lo + (i + 1) * q) for i in range(nproc)]
    ctx = mp.get_context('fork')
    start, q = ctx.Event(), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(method, a, b, start, q)) for a, b in spans]
    for p in procs:
        p.start()
    time.sleep(0.5)
    m0, r0, t0 = metrics(), rx(), time.perf_counter()
    start.set()
    results = [q.get() for _ in procs]
    wall = time.perf_counter() - t0
    for p in procs:
        p.join()
    m1, r1 = metrics(), rx()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in set(m0) | set(m1)}
    d = {k: v for k, v in d.items() if v}
    # distinct bytes: same region -> what every worker got through; disjoint -> sum of the quarters
    covered = min(r['bytes'] for r in results) if layout == 'same' else sum(r['bytes'] for r in results)
    reads = d.get('fs_ops_count{fs_op="ReadFile"}', 0)
    out = {
        'case': name, 'method': method, 'procs': nproc, 'layout': layout, 'region_offset': lo, 'region_bytes': REGION,
        'resident_before': res0, 'resident_after': resident(lo, hi), 'wall_s': round(wall, 2), 'workers': results,
        'distinct_bytes_done': covered, 'MBps_distinct': round(covered / wall / 1e6, 1),
        'host_rx_bytes': r1 - r0, 'rx_over_distinct': round((r1 - r0) / max(covered, 1), 2),
        'fuse_readfile_ops': reads,
        'fuse_avg_read_KiB': round(d.get('gcs_read_bytes_count', 0) / reads / 1024, 1) if reads else None,
        'fuse_avg_latency_ms': round(d.get('fs_ops_latency_sum{fs_op="ReadFile"}', 0) / reads / 1000, 2) if reads else None,
        'metric_deltas': d,
    }
    print('FUSEBENCH_JSON=' + json.dumps(out), flush=True)


if __name__ == '__main__':
    print('file', FILE, os.path.getsize(FILE), 'region_MiB', REGION >> 20, 'cap_s', CAP_S, flush=True)
    try:
        dev = os.popen(f'mountpoint -d {os.path.dirname(os.path.dirname(FILE))}').read().strip()
        print('read_ahead_kb', open(f'/sys/class/bdi/{dev}/read_ahead_kb').read().strip(), flush=True)
    except OSError as e:
        print('read_ahead_kb unavailable:', e, flush=True)
    cases = [('mmap-1', 'mmap', 1, 'same'), ('mmap-4same', 'mmap', 4, 'same'), ('mmap-4disjoint', 'mmap', 4, 'disjoint'),
             ('pread1M-1', 'pread', 1, 'same'), ('pread1M-4same', 'pread', 4, 'same'), ('pread1M-4disjoint', 'pread', 4, 'disjoint')]
    only = os.environ.get('FB_CASES')
    for i, c in enumerate(cases):
        if only and c[0] not in only.split(','):
            continue
        case(*c, idx=i)
    evict()
