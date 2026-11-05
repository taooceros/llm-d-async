#!/usr/bin/env python3
"""TPUFlex startup state for the CUDA vLLM pm-* arms: pack, publish, deliver, serve.

  pack    --model-dir D --pyc-dir P --home H --snap-dir S --out O [--meta JSON]
          model.tar   the checkpoint dir without weights (config, tokenizer, chat template, index)
          pyc.tar     the image's bytecode (PYTHONPYCACHEPREFIX tree written by an earlier boot of the same image)
          bank.tar    $HOME compile state: vLLM torch.compile/AOT/Inductor artifacts, Triton, FlashInfer, CUDA
                      ComputeCache (as the stock *-cc arms, minus pip/HF downloads)
          snap/       manifest.json + shard<k>.bin: the terminal weight snapshot (pm_cuda.py save)
          files.json  object list (name, size, role) + meta (e.g. the pinned KV bytes)
  upload  --pack O --dst gs://bucket/prefix [--workers N]            files.json last
  fetch   --src gs://bucket/prefix | http://peer:port  --root R --home H --model-dir D --pyc-dir P --snap-dir S
          [--workers W] [--keep-pack K: also keep the objects as a pack in K, to `serve` them as a peer]
          [--skip pyc,bank,snap: leave those roles out (leave-one-out ablations; .bank_done is still written)]
          files.json, then model.tar + pyc.tar + snap/manifest.json (then R/.model_done: the engine may launch),
          then bank.tar (extracted into H, then R/.bank_done), then the shards W at a time, each written
          straight into an mmap'd tmpfs file, then snap/shard<k>.done (the plugin copies a shard to the GPU as
          soon as its marker exists). gs://: the Run:ai streamer's C library via ctypes (the image's
          runai_model_streamer), in W worker processes with one streamer handle each, one file per request:
          several handles as threads of one process fail at random ("FINISHED after k of n ranges", as in the
          TPU port) and crawl at ~0.04 GB/s. http://: parallel HTTP range GETs.
          Prints one "FETCH {json}" line with per-object completion times (s after start).
  serve   --root O --port 8080         HTTP/1.1 server with Range support (sendfile) over a pack dir: the peer
          side of pm-peer, run next to a live engine.
"""
import argparse
import ctypes
import http.client
import io
import json
import mmap
import multiprocessing
import os
import queue
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RANGE = 64 << 20
SMALL = 8 << 20  # fetch: smaller objects take one plain GET, larger ones go through the parallel path
MD = 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/'


def split_gs(uri):
    bucket, _, prefix = uri.removeprefix('gs://').partition('/')
    return bucket, prefix.strip('/')


# ------------------------------------------------------------------------------------------------ pack

def _tar(path, members, exclude=()):
    with tarfile.open(path, 'w') as tf:
        for src, arc in members:
            if os.path.exists(src):
                tf.add(src, arcname=arc, filter=lambda ti: None if any(ti.name.startswith(x) for x in exclude)
                       else ti)


def cmd_pack(a):
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    files = []
    model = [(os.path.join(a.model_dir, n), n) for n in sorted(os.listdir(a.model_dir))
             if os.path.isfile(os.path.join(a.model_dir, n)) and not n.endswith('.safetensors')
             and not n.startswith('.')]
    _tar(os.path.join(a.out, 'model.tar'), model)
    files.append({'name': 'model.tar', 'role': 'model'})
    _tar(os.path.join(a.out, 'pyc.tar'), [(a.pyc_dir, '.')])
    files.append({'name': 'pyc.tar', 'role': 'pyc'})
    home = [(os.path.join(a.home, n), n) for n in ('.cache', '.triton', '.nv', '.config')]
    _tar(os.path.join(a.out, 'bank.tar'), home, exclude=('.cache/pip', '.cache/huggingface'))
    files.append({'name': 'bank.tar', 'role': 'bank'})
    snap = os.path.join(a.out, 'snap')
    if not os.path.lexists(snap):
        os.symlink(os.path.abspath(a.snap_dir), snap)
    with open(os.path.join(a.snap_dir, 'manifest.json')) as fh:
        man = json.load(fh)
    files.append({'name': 'snap/manifest.json', 'role': 'manifest'})
    files += [{'name': f'snap/shard{s["shard"]}.bin', 'role': 'shard', 'shard': s['shard']} for s in man['shards']]
    for f in files:
        f['size'] = os.path.getsize(os.path.join(a.out, f['name']))
    meta = json.loads(a.meta) if a.meta else {}
    with open(os.path.join(a.out, 'files.json'), 'w') as fh:
        json.dump({'files': files, 'meta': meta}, fh, indent=1)
    print('PACK ' + json.dumps({'seconds': round(time.time() - t0, 2), 'meta': meta,
                                'files': [(f['name'], f['size']) for f in files if f['role'] != 'shard'],
                                'shards': len(man['shards']), 'snap_bytes': man['bytes']}), flush=True)


def cmd_upload(a):
    from google.cloud import storage
    from google.cloud.storage import transfer_manager as tm
    bucket_name, prefix = split_gs(a.dst)
    creds = None
    if os.environ.get('GCS_TOKEN'):  # node SA scopes are devstorage.read_only: a host-minted user token
        from google.oauth2.credentials import Credentials
        creds = Credentials(os.environ['GCS_TOKEN'])
    bucket = storage.Client(credentials=creds, project='sysres-disagg-ml').bucket(bucket_name)
    with open(os.path.join(a.pack, 'files.json')) as fh:
        man = json.load(fh)
    t0 = time.time()
    total = 0
    for f in man['files']:
        blob = bucket.blob(f'{prefix}/{f["name"]}')
        path = os.path.join(a.pack, f['name'])
        if f['size'] > 4 * RANGE:
            tm.upload_chunks_concurrently(path, blob, chunk_size=128 << 20, max_workers=a.workers,
                                          worker_type=tm.THREAD)
        else:
            blob.upload_from_filename(path)
        total += f['size']
    bucket.blob(f'{prefix}/files.json').upload_from_string(json.dumps(man), content_type='application/json')
    dt = time.time() - t0
    print('UPLOAD ' + json.dumps({'dst': a.dst, 'bytes': total, 'seconds': round(dt, 1),
                                  'GBps': round(total / dt / 1e9, 2)}), flush=True)


# ------------------------------------------------------------------------------------- gs:// streamer

class Streamer:
    """The Run:ai streamer C API (runai_start / runai_request / runai_response / runai_end) via ctypes."""

    def __init__(self, lib_path):
        L = ctypes.CDLL(lib_path)
        vp, sz, u32 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32
        L.runai_start.argtypes, L.runai_start.restype = [ctypes.POINTER(vp)], ctypes.c_int
        L.runai_end.argtypes = [vp]
        L.runai_request.argtypes = [vp, u32, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(sz),
                                    ctypes.POINTER(sz), ctypes.POINTER(vp), ctypes.POINTER(u32),
                                    ctypes.POINTER(ctypes.POINTER(sz))] + [ctypes.c_char_p] * 5
        L.runai_request.restype = ctypes.c_int
        L.runai_response.argtypes = [vp, ctypes.POINTER(u32), ctypes.POINTER(u32)]
        L.runai_response.restype = ctypes.c_int
        L.runai_response_str.argtypes, L.runai_response_str.restype = [ctypes.c_int], ctypes.c_char_p
        self.L = L
        self.h = vp(0)
        self._check(L.runai_start(ctypes.byref(self.h)), 'start')

    def _check(self, rc, what):
        if rc != 0:
            raise RuntimeError(f'runai_{what}: {self.L.runai_response_str(rc).decode()}')

    def read(self, url, dst, base=0):
        """url[base:base+len(dst)] -> dst in RANGE pieces; waits for exactly that many range completions (a
        FINISHED before all of them is an error: seen with the TPU port's libstreamer)."""
        size = len(dst)
        ranges = [min(RANGE, size - o) for o in range(0, size, RANGE)] or [0]
        keep = ctypes.c_char.from_buffer(dst)
        c_paths = (ctypes.c_char_p * 1)(url.encode())
        c_offs, c_sizes = (ctypes.c_size_t * 1)(base), (ctypes.c_size_t * 1)(size)
        c_dsts = (ctypes.c_void_p * 1)(ctypes.addressof(keep))
        c_nr = (ctypes.c_uint32 * 1)(len(ranges))
        arr = (ctypes.c_size_t * len(ranges))(*ranges)
        c_ranges = (ctypes.POINTER(ctypes.c_size_t) * 1)(ctypes.cast(arr, ctypes.POINTER(ctypes.c_size_t)))
        self._check(self.L.runai_request(self.h, 1, c_paths, c_offs, c_sizes, c_dsts, c_nr, c_ranges,
                                         None, None, None, None, None), 'request')
        fi, ri = ctypes.c_uint32(), ctypes.c_uint32()
        done = set()
        for k in range(len(ranges)):
            rc = self.L.runai_response(self.h, ctypes.byref(fi), ctypes.byref(ri))
            if rc == 1:
                raise RuntimeError(f'runai_response: FINISHED after {k} of {len(ranges)} ranges')
            self._check(rc, 'response')
            done.add(ri.value)
        if len(done) != len(ranges):
            raise RuntimeError(f'runai_response: {len(done)} distinct of {len(ranges)} ranges')
        del keep

    def close(self):
        self.L.runai_end(self.h)


def streamer_lib():
    import importlib.util
    spec = importlib.util.find_spec('runai_model_streamer')
    base = os.path.dirname(spec.origin)
    for root, _, names in os.walk(base):
        if 'libstreamer.so' in names:
            return os.path.join(root, 'libstreamer.so')
    raise RuntimeError(f'libstreamer.so not found under {base}')


def gcs_token():
    return json.loads(urllib.request.urlopen(urllib.request.Request(
        MD + 'token', headers={'Metadata-Flavor': 'Google'}), timeout=10).read())['access_token']


def _gcs_worker(conn, lib):
    """One fetch process: one libstreamer handle, reused for file after file; a failed request gets a new
    handle. Jobs (url, path, size) arrive on conn; the file is created at path, sized, mmap'd and filled."""
    st = None
    while (job := conn.recv()) is not None:
        url, path, size = job
        t = time.time()
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size) if size else None
        os.close(fd)
        err, attempt = None, 0
        for attempt in range(3):
            if mm is None:
                break
            try:
                st = st or Streamer(lib)
                st.read(url, mm)
                err = None
                break
            except RuntimeError as e:
                err = str(e)
                print(f'fetch: {url} attempt {attempt}: {e}; retrying with a new handle', flush=True)
                if st is not None:
                    st.close()
                    st = None
        if mm is not None:
            mm.close()
        conn.send((err, time.time() - t, attempt))
    if st is not None:
        st.close()


class GcsSource:
    """gs:// through W fetch processes, one libstreamer handle each. A request that stalls (4 processes on
    2026-10-24: two shards hung for 85-100 s while the other 30 arrived at 6 GB/s) is cut off at
    STALL_S + size / STALL_BPS: its process is killed and replaced, and the file is fetched again."""
    STALL_S = float(os.environ.get('PM_FETCH_STALL_S', '10'))
    STALL_BPS = float(os.environ.get('PM_FETCH_STALL_BPS', '0.3e9'))

    def __init__(self, uri, procs):
        """Forks the fetch processes: call before this process starts any thread."""
        self.bucket, self.prefix = split_gs(uri)
        self.lib = streamer_lib()
        self.free, self.workers, self.stalls = queue.Queue(), [], []
        for _ in range(procs):
            self.free.put(self._start('fork'))
        self.token = gcs_token()

    def _start(self, method):
        """A new fetch process. Replacements use spawn: forking a process that runs threads is unsafe."""
        ctx = multiprocessing.get_context(method)
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_gcs_worker, args=(child, self.lib), daemon=True)
        proc.start()
        child.close()
        self.workers.append((proc, parent))
        return proc, parent

    def small(self, name):
        url = (f'https://storage.googleapis.com/storage/v1/b/{self.bucket}/o/'
               f'{urllib.parse.quote(f"{self.prefix}/{name}", safe="")}?alt=media')
        req = urllib.request.Request(url, headers={'Authorization': f'Bearer {self.token}'})
        return urllib.request.urlopen(req, timeout=60).read()

    def into_file(self, name, path, size):
        limit = self.STALL_S + size / self.STALL_BPS
        for attempt in range(3):
            proc, conn = self.free.get()  # an idle fetch process
            t = time.time()
            try:
                conn.send((f'gs://{self.bucket}/{self.prefix}/{name}', path, size))
                if conn.poll(limit):
                    err, _, _ = conn.recv()
                    self.free.put((proc, conn))
                    if err:
                        raise RuntimeError(f'{name}: {err}')
                    return
                why = f'no reply after {limit:.0f} s'
            except (EOFError, OSError) as e:
                why = f'fetch process lost: {e!r}'
            self.stalls.append({'name': name, 'attempt': attempt, 'pid': proc.pid, 'after_s': round(time.time() - t, 1),
                                'why': why})
            print(f'fetch: {name} {why} (pid {proc.pid}); killing it, retrying in a new process', flush=True)
            proc.kill()
            proc.join()
            conn.close()
            self.free.put(self._start('spawn'))
        raise RuntimeError(f'{name}: stalled 3 times')

    def close(self):
        for _, conn in self.workers:
            try:
                conn.send(None)
            except OSError:
                pass


class HttpSource:
    """Parallel HTTP range GETs from a peer's `pm_state.py serve`, one connection per thread."""

    def __init__(self, uri, threads):
        u = urllib.parse.urlparse(uri)
        self.host, self.port, self.base = u.hostname, u.port or 80, u.path.rstrip('/')
        self.threads = threads
        self.local = threading.local()

    def _conn(self):
        c = getattr(self.local, 'conn', None)
        if c is None:
            c = self.local.conn = http.client.HTTPConnection(self.host, self.port, timeout=120)
        return c

    def _get(self, name, lo=None, hi=None, dst=None):
        for attempt in range(3):
            c = self._conn()
            try:
                headers = {} if lo is None else {'Range': f'bytes={lo}-{hi - 1}'}
                c.request('GET', f'{self.base}/{name}', headers=headers)
                r = c.getresponse()
                if r.status not in (200, 206):
                    raise RuntimeError(f'{name}: HTTP {r.status}')
                if dst is None:
                    return r.read()
                view, got = memoryview(dst), 0
                while got < len(view):
                    n = r.readinto(view[got:])
                    if not n:
                        raise RuntimeError(f'{name}@{lo}: short read {got}/{len(view)}')
                    got += n
                return None
            except (OSError, http.client.HTTPException, RuntimeError):
                c.close()
                self.local.conn = None
                if attempt == 2:
                    raise

    def small(self, name):
        return self._get(name)

    def into(self, name, dst):
        size = len(dst)
        jobs = queue.Queue()
        for o in range(0, size, RANGE):
            jobs.put(o)
        errors = []

        def work():
            while True:
                try:
                    o = jobs.get_nowait()
                except queue.Empty:
                    return
                try:
                    self._get(name, o, min(size, o + RANGE), memoryview(dst)[o:min(size, o + RANGE)])
                except BaseException as e:  # noqa: BLE001
                    errors.append(repr(e))
                    return

        ths = [threading.Thread(target=work) for _ in range(min(self.threads, -(-size // RANGE)))]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        if errors:
            raise RuntimeError(f'{name}: {errors[:2]}')

    def into_file(self, name, path, size):
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd, size)
        mm = mmap.mmap(fd, size) if size else None
        os.close(fd)
        if mm is not None:
            self.into(name, mm)
            mm.close()

    def close(self):
        pass


# ----------------------------------------------------------------------------------------------- fetch

def extract(buf, dest):
    with tarfile.open(fileobj=io.BytesIO(buf), mode='r:') as tf:
        tf.extractall(dest, filter='tar')


def cmd_fetch(a):
    T0 = time.time()
    log = {}

    def stamp(k):
        log[k] = round(time.time() - T0, 3)

    gcs = a.src.startswith('gs://')
    src = GcsSource(a.src, a.workers) if gcs else HttpSource(a.src, a.http_threads)  # forks: before any thread
    man = json.loads(src.small('files.json'))
    stamp('files_json')
    files = {f['name']: f for f in man['files']}
    for d in (a.root, a.snap_dir, a.model_dir, a.pyc_dir, a.home):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(a.root, 'files.json'), 'w') as fh:
        json.dump(man, fh)
    if a.keep_pack:  # this node becomes a peer: keep the delivered objects in pack layout for `serve`
        os.makedirs(a.keep_pack, exist_ok=True)
        with open(os.path.join(a.keep_pack, 'files.json'), 'w') as fh:
            json.dump(man, fh)
        if not os.path.lexists(os.path.join(a.keep_pack, 'snap')):
            os.symlink(os.path.abspath(a.snap_dir), os.path.join(a.keep_pack, 'snap'))
    errors = []

    def guarded(fn, *args):
        def run():
            try:
                fn(*args)
            except BaseException as e:  # noqa: BLE001
                errors.append(repr(e))
                print(f'fetch: ERROR {e!r}', flush=True)
        return run

    def small_job(name, dest):
        size = files[name]['size']
        if size < SMALL:  # one GET
            data = src.small(name)
            if len(data) != size:
                raise RuntimeError(f'{name}: {len(data)} bytes, expected {size}')
            if name.endswith('.tar'):
                extract(data, dest)
                if a.keep_pack:
                    with open(os.path.join(a.keep_pack, name), 'wb') as fh:
                        fh.write(data)
            else:
                with open(dest + '.tmp', 'wb') as fh:
                    fh.write(data)
                os.replace(dest + '.tmp', dest)
        else:  # streamed like a shard, through a file in root
            part = os.path.join(a.root, os.path.basename(name) + '.part')
            src.into_file(name, part, size)
            if name.endswith('.tar'):  # GNU tar: pyc.tar (8,113 files) in 0.11 s, tarfile.extractall 0.85 s
                subprocess.run(['tar', '-xf', part, '-C', dest], check=True)
                if a.keep_pack:
                    os.replace(part, os.path.join(a.keep_pack, name))
                else:
                    os.remove(part)
            else:
                os.replace(part, dest)
        stamp(name)

    # 1) what the API server needs at launch, and the snapshot manifest
    skip = set(filter(None, (a.skip or '').split(',')))
    if skip - {'pyc', 'bank', 'snap'}:
        raise SystemExit(f'--skip: unknown roles {skip - {"pyc", "bank", "snap"}}')
    first_jobs = [('model.tar', a.model_dir)]
    if 'pyc' not in skip:
        first_jobs.append(('pyc.tar', a.pyc_dir))
    if 'snap' not in skip:
        first_jobs.append(('snap/manifest.json', os.path.join(a.snap_dir, 'manifest.json')))
    first = [threading.Thread(target=guarded(small_job, n, d)) for n, d in first_jobs]
    for t in first:
        t.start()
    for t in first:
        t.join()
    if errors:
        print('FETCH ' + json.dumps({'src': a.src, 'errors': errors, 'events': log}), flush=True)
        return 1
    open(os.path.join(a.root, '.model_done'), 'w').close()
    print(f'fetch: {" + ".join(n for n, _ in first_jobs)} {max(log[n] for n, _ in first_jobs)} s', flush=True)

    # 2) the compile bank, then 3) the shards, W in flight (the bank first: compile needs it at ~20 s)
    def bank_job():
        if 'bank' not in skip:
            small_job('bank.tar', a.home)
        open(os.path.join(a.root, '.bank_done'), 'w').close()

    shards = [] if 'snap' in skip else sorted((f for f in man['files'] if f['role'] == 'shard'),
                                              key=lambda f: f['shard'])
    jobs = queue.Queue()
    for f in shards:
        jobs.put(f)

    def shard_worker():
        while True:
            try:
                f = jobs.get_nowait()
            except queue.Empty:
                return
            path = os.path.join(a.snap_dir, os.path.basename(f['name']))
            src.into_file(f['name'], path, f['size'])
            open(path[:-4] + '.done', 'w').close()
            stamp(f['name'])

    bank = threading.Thread(target=guarded(bank_job))
    bank.start()
    if a.bank_first:
        bank.join()
    workers = [threading.Thread(target=guarded(shard_worker)) for _ in range(a.workers)]
    for t in workers:
        t.start()
    for t in workers + [bank]:
        t.join()
    stamp('all_done')
    src.close()
    total = sum(f['size'] for f in man['files'])
    snap = sum(f['size'] for f in shards)
    first_shard = min((log[f['name']] for f in shards if f['name'] in log), default=None)
    print('FETCH ' + json.dumps({'kind': 'pm', 'src': a.src, 'bytes': total, 'snap_bytes': snap,
                                 'seconds': log['all_done'], 'GBps': round(total / log['all_done'] / 1e9, 2),
                                 'shards_GBps': round(snap / max(1e-9, log['all_done'] - log.get('bank.tar', 0))
                                                      / 1e9, 2) if a.bank_first else None,
                                 'first_shard_s': first_shard, 'workers': a.workers, 'skip': sorted(skip),
                                 'stalls': getattr(src, 'stalls', []),
                                 'mode': 'processes' if gcs else 'threads',
                                 'streamer_concurrency': os.environ.get('RUNAI_STREAMER_CONCURRENCY'),
                                 'errors': errors,
                                 'meta': man.get('meta'), 'events': log}), flush=True)
    return 1 if errors else 0


# ----------------------------------------------------------------------------------------------- serve

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    root = '.'

    def log_message(self, *args):
        pass

    def do_GET(self):
        rel = urllib.parse.unquote(self.path.lstrip('/'))
        path = os.path.realpath(os.path.join(self.root, rel))
        if '..' in rel.split('/') or not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        lo, hi, status = 0, size, 200
        rng = self.headers.get('Range')
        if rng and rng.startswith('bytes='):
            a, _, b = rng[6:].partition('-')
            lo, hi, status = int(a), min(size, int(b) + 1 if b else size), 206
        self.send_response(status)
        self.send_header('Content-Length', str(hi - lo))
        if status == 206:
            self.send_header('Content-Range', f'bytes {lo}-{hi - 1}/{size}')
        self.end_headers()
        with open(path, 'rb') as fh:
            off = lo
            while off < hi:
                n = os.sendfile(self.connection.fileno(), fh.fileno(), off, hi - off)
                if n == 0:
                    break
                off += n


def cmd_serve(a):
    Handler.root = a.root
    srv = ThreadingHTTPServer(('0.0.0.0', a.port), Handler)
    srv.daemon_threads = True
    srv.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
    print(f'SERVE root={a.root} port={a.port}', flush=True)
    srv.serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('pack')
    for k in ('--model-dir', '--pyc-dir', '--home', '--snap-dir', '--out'):
        s.add_argument(k, required=True)
    s.add_argument('--meta')
    s = sub.add_parser('upload')
    s.add_argument('--pack', required=True)
    s.add_argument('--dst', required=True)
    s.add_argument('--workers', type=int, default=32)
    s = sub.add_parser('fetch')
    for k in ('--src', '--root', '--home', '--model-dir', '--pyc-dir', '--snap-dir'):
        s.add_argument(k, required=True)
    s.add_argument('--workers', type=int, default=3,
                   help='files in flight (gs://: one fetch process each; 3 measured fastest, 4 stalled)')
    s.add_argument('--http-threads', type=int, default=16, help='range GETs in flight per shard (http://)')
    s.add_argument('--bank-first', type=int, default=1, help='bank.tar before the shards start')
    s.add_argument('--skip', default='', help='comma-separated roles not to deliver: pyc, bank, snap')
    s.add_argument('--keep-pack', help='also keep the delivered objects in pack layout here (a peer built from a '
                                       'published state)')
    s = sub.add_parser('serve')
    s.add_argument('--root', required=True)
    s.add_argument('--port', type=int, default=8080)
    a = p.parse_args()
    return {'pack': cmd_pack, 'upload': cmd_upload, 'fetch': cmd_fetch, 'serve': cmd_serve}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
