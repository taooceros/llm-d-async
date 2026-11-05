#!/usr/bin/env python3
"""Host-to-device bandwidth ceiling on this TPU host, to judge the TPUFlex weight restore (3.2-3.4 GB/s per
chip, 15.8 GB per rank, 4 ranks at once). JAX/PJRT on all local chips from one process; nothing else may hold
the chips. Prints H2D_JSON lines: one or all chips, one transfer or 32 MB pieces (the restore's per-tensor shape).

Usage: torchtpu_h2d_bench.py [GB_PER_CHIP=4]
"""
import json
import sys
import threading
import time

import jax
import numpy as np


def run(devs, host, chunk, reps=2):
    best = None
    for _ in range(reps):
        out = [[] for _ in devs]

        def put(i, d):
            for off in range(0, host.nbytes, chunk):
                out[i].append(jax.device_put(host[off:off + chunk], d))
            for a in out[i]:
                a.block_until_ready()

        t0 = time.perf_counter()
        ths = [threading.Thread(target=put, args=(i, d)) for i, d in enumerate(devs)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        dt = time.perf_counter() - t0
        for lst in out:
            for a in lst:
                a.delete()
        best = dt if best is None else min(best, dt)
    return best


def main():
    gb = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    host = np.ones(int(gb * (1 << 30)), dtype=np.uint8)
    devs = jax.local_devices()
    print(f'devices: {devs}', flush=True)
    for name, ds in (('one_chip', devs[:1]), ('all_chips', devs)):
        for label, chunk in (('single', host.nbytes), ('32MB', 32 << 20)):
            dt = run(ds, host, chunk)
            res = {'case': name, 'transfer': label, 'chips': len(ds), 'GB_per_chip': round(host.nbytes / 1e9, 2),
                   'seconds': round(dt, 3), 'GBps_per_chip': round(host.nbytes / dt / 1e9, 2),
                   'GBps_total': round(len(ds) * host.nbytes / dt / 1e9, 2)}
            print('H2D_JSON=' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
