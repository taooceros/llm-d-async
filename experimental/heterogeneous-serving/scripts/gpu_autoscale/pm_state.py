#!/usr/bin/env python3
"""TPUFlex startup-state delivery for the vLLM-JAX GPU arms (gpu_jax/hetero_jax_gpu/pm.py is the engine side).

State of one seed, as laid out in GCS (gs://BUCKET/pm-state/<seed>/) and on a host (/pmstate):
    model/      the checkpoint's small files (config, tokenizer, chat template; no safetensors)
    snap/       manifest.json + c<NNNN>.raw chunks: the terminal weight snapshot (after tpu-inference's conversion)
    cc.tar      $HOME caches of a warmed engine: JAX persistent compilation cache with every program
                (min compile time 0) + XLA autotune results

    pm_state.py fetch-gcs gs://.../pm-state/<seed> --root /pmstate --home $HOME --gate FILE [--workers 32]
                          [--no-snap] [--cc-uri gs://.../other.tar]   (leave-one-out arms, results/gpu-jax/loo-*)
    pm_state.py fetch-peer HOST:PORT --root /pmstate --home $HOME --gate FILE [--conns 16]
    pm_state.py serve --root /pmstate [--port 7070]        (on a live peer; the root must hold cc.tar)
    pm_state.py upload --root /pmstate --home $HOME --ckpt gs://.../gemma-4-31b --dest gs://.../pm-state/<seed>

A fetch delivers in this order and prints "MARK <event> <epoch>" after each step: model/ (meta_done: the engine can be
launched), cc.tar extracted into $HOME, then the gate file is created (cc_done), then the snapshot. Snapshot
chunks are fetched as 64 MiB slices in chunk order by parallel workers; each chunk is renamed from .part to its final
name once complete, so the engine's restore (which waits for each final name) streams right behind the delivery.
The manifest lands before any chunk. Prints one "FETCH {json}" line at the end. <root>/.meta_done is created with
MARK meta_done (run_arm.sh launches the engine on it).
"""
import argparse
import json
import os
import socket
import socketserver
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

SLICE = 64 << 20
MANIFEST = "manifest.json"
HOME_TAR = ["--exclude=./.cache/pip", "--exclude=./.cache/huggingface", "."]


def mark(event):
    print(f"MARK {event} {time.time():.6f}", flush=True)


def split(uri):
    bucket, _, name = uri.removeprefix("gs://").partition("/")
    return bucket, name.rstrip("/")


# ------------------------------------------------------------------------------------------------ GCS

_client = None


def _gcs_init():
    global _client
    from google.cloud import storage
    _client = storage.Client()


def _gcs_slice(bucket, name, start, end, path):
    """Bytes [start, end) of gs://bucket/name -> path at offset start (the file exists, full size)."""
    blob = _client.bucket(bucket).blob(name)
    with open(path, "r+b") as f:
        f.seek(start)
        blob.download_to_file(f, start=start, end=end - 1, raw_download=True, checksum=None)
    return end - start


def _gcs_whole(bucket, name, path):
    _client.bucket(bucket).blob(name).download_to_filename(path, raw_download=True, checksum=None)
    return Path(path).stat().st_size


def _ordered_chunks(snap, manifest, submit_slice):
    """Create every chunk's .part file, submit its slices in chunk order; returns {future: chunk}."""
    futures = {}
    for c, size in enumerate(manifest["chunk_bytes"]):
        part = snap / f"c{c:04d}.raw.part"
        with open(part, "wb") as f:
            f.truncate(size)
        for start in range(0, size, SLICE):
            futures[submit_slice(c, start, min(size, start + SLICE), str(part))] = c
    return futures


def _complete_chunks(snap, manifest, futures):
    left = {c: 0 for c in range(len(manifest["chunk_bytes"]))}
    for c in futures.values():
        left[c] += 1
    for c in [c for c, n in left.items() if n == 0]:  # empty chunk
        os.replace(snap / f"c{c:04d}.raw.part", snap / f"c{c:04d}.raw")
    total = 0
    for fut in as_completed(futures):
        total += fut.result()
        c = futures[fut]
        left[c] -= 1
        if left[c] == 0:
            os.replace(snap / f"c{c:04d}.raw.part", snap / f"c{c:04d}.raw")
    return total


def fetch_gcs(uri, root, home, gate, workers, snap_on=True, cc_uri=None):
    """snap_on=False: no snapshot (manifest or chunks; the engine loads the checkpoint itself). cc_uri: stream this
    $HOME cache tarball (e.g. a stock 1 s-threshold cache) instead of the state's cc.tar."""
    bucket, prefix = split(uri)
    root, started = Path(root), time.time()
    model, snap = root / "model", root / "snap"
    model.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)
    out = {"source": uri, "workers": workers, "snap": snap_on, "cc_source": cc_uri or f"{uri.rstrip('/')}/cc.tar"}
    _gcs_init()
    names = [b.name for b in _client.list_blobs(bucket, prefix=prefix + "/model/")]
    with ProcessPoolExecutor(workers, initializer=_gcs_init) as pool:
        meta = [pool.submit(_gcs_whole, bucket, n, str(model / Path(n).name)) for n in names]
        out["model_bytes"] = sum(f.result() for f in meta)
        (root / ".meta_done").touch()
        mark("meta_done")
        out["meta_s"] = time.time() - started
        futures = {}
        if snap_on:
            _gcs_whole(bucket, f"{prefix}/snap/{MANIFEST}", str(snap / (MANIFEST + ".part")))
            manifest = json.loads((snap / (MANIFEST + ".part")).read_text())
            futures = _ordered_chunks(snap, manifest, lambda c, s, e, p: pool.submit(
                _gcs_slice, bucket, f"{prefix}/snap/c{c:04d}.raw", s, e, p))
            os.replace(snap / (MANIFEST + ".part"), snap / MANIFEST)
        t = time.time()  # the cache streams in this process while the pool fetches the snapshot
        cc_bucket, cc_name = split(cc_uri) if cc_uri else (bucket, f"{prefix}/cc.tar")
        blob = _client.bucket(cc_bucket).get_blob(cc_name)
        with blob.open("rb", chunk_size=SLICE) as stream, tarfile.open(fileobj=stream, mode="r|*") as archive:
            archive.extractall(home, filter="tar")
        out["cc_bytes"], out["cc_s"] = blob.size, time.time() - t
        Path(gate).touch()
        mark("cc_done")
        out["snap_bytes"] = _complete_chunks(snap, manifest, futures) if snap_on else 0
    out["seconds"] = time.time() - started
    mark("download_end")
    return out


# ----------------------------------------------------------------------------------------------- peer

class _Handler(socketserver.BaseRequestHandler):
    """Line protocol: "LIST\\n" -> JSON {relpath: size}; "GET relpath offset length\\n" -> exactly length bytes."""

    def handle(self):
        root = self.server.root
        rfile = self.request.makefile("rb")
        for line in rfile:
            parts = line.decode().split()
            if parts == ["LIST"]:
                files = {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file()
                         and not p.name.endswith(".part")}
                body = json.dumps(files).encode()
                self.request.sendall(len(body).to_bytes(8, "big") + body)
            elif len(parts) == 4 and parts[0] == "GET":
                path = (root / parts[1]).resolve()
                if root not in path.parents:
                    return
                offset, length = int(parts[2]), int(parts[3])
                with open(path, "rb") as f:
                    while length:
                        sent = os.sendfile(self.request.fileno(), f.fileno(), offset, length)
                        if sent == 0:
                            raise EOFError(path)
                        offset += sent
                        length -= sent
            else:
                return


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(root, port):
    server = _Server(("0.0.0.0", port), _Handler)
    server.root = Path(root).resolve()
    print(f"pm_state serving {server.root} on :{port}", flush=True)
    server.serve_forever()


class _Conn:
    def __init__(self, addr):
        host, port = addr.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)))
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)

    def recv_into(self, view):
        while len(view):
            n = self.sock.recv_into(view)
            if n == 0:
                raise EOFError("peer closed")
            view = view[n:]

    def list(self):
        self.sock.sendall(b"LIST\n")
        size = bytearray(8)
        self.recv_into(memoryview(size))
        body = bytearray(int.from_bytes(size, "big"))
        self.recv_into(memoryview(body))
        return json.loads(body)

    def get(self, rel, offset, length, path):
        """Bytes [offset, offset+length) of the peer's rel -> path at offset (mapped, received in place)."""
        import mmap
        self.sock.sendall(f"GET {rel} {offset} {length}\n".encode())
        if not length:
            return 0
        with open(path, "r+b") as f:
            page = offset - offset % mmap.ALLOCATIONGRANULARITY
            with mmap.mmap(f.fileno(), length + offset - page, offset=page) as m:
                view = memoryview(m)
                try:
                    self.recv_into(view[offset - page:])
                finally:
                    view.release()
        return length


def fetch_peer(addr, root, home, gate, conns):
    import queue
    root, started = Path(root), time.time()
    model, snap = root / "model", root / "snap"
    model.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)
    out = {"source": addr, "conns": conns}
    first = _Conn(addr)
    files = first.list()
    for rel in [r for r in files if r.startswith("model/")]:
        path = root / rel
        with open(path, "wb") as f:
            f.truncate(files[rel])
        first.get(rel, 0, files[rel], path)
    out["model_bytes"] = sum(v for r, v in files.items() if r.startswith("model/"))
    (root / ".meta_done").touch()
    mark("meta_done")
    out["meta_s"] = time.time() - started
    first.get(f"snap/{MANIFEST}", 0, files[f"snap/{MANIFEST}"], _truncated(snap / (MANIFEST + ".part"),
                                                                           files[f"snap/{MANIFEST}"]))
    manifest = json.loads((snap / (MANIFEST + ".part")).read_text())
    tasks = queue.Queue()
    pool = ThreadPoolExecutor(conns)
    lock = threading.Lock()

    def worker():
        conn, done = _Conn(addr), 0
        while True:
            item = tasks.get()
            if item is None:
                return done
            c, s, e, p = item
            done += conn.get(f"snap/c{c:04d}.raw", s, e - s, p)
            item_done(c)

    left = {}

    def item_done(c):
        with lock:
            left[c] -= 1
            if left[c] == 0:
                os.replace(snap / f"c{c:04d}.raw.part", snap / f"c{c:04d}.raw")

    for c, size in enumerate(manifest["chunk_bytes"]):
        part = _truncated(snap / f"c{c:04d}.raw.part", size)
        left[c] = max(1, -(-size // SLICE))
        if size == 0:
            left[c] = 1
            tasks.put((c, 0, 0, part))
        for s in range(0, size, SLICE):
            tasks.put((c, s, min(size, s + SLICE), part))
    for _ in range(conns):
        tasks.put(None)
    os.replace(snap / (MANIFEST + ".part"), snap / MANIFEST)
    futures = [pool.submit(worker) for _ in range(conns)]
    t = time.time()  # cache tar on the first connection while the others fetch the snapshot
    tar_part = _truncated(root / "cc.tar.part", files["cc.tar"])
    first.get("cc.tar", 0, files["cc.tar"], tar_part)
    with tarfile.open(tar_part) as archive:
        archive.extractall(home, filter="tar")
    os.unlink(tar_part)
    out["cc_bytes"], out["cc_s"] = files["cc.tar"], time.time() - t
    Path(gate).touch()
    mark("cc_done")
    out["snap_bytes"] = sum(f.result() for f in futures)
    pool.shutdown()
    if out["snap_bytes"] != sum(manifest["chunk_bytes"]):
        raise RuntimeError(f"peer delivered {out['snap_bytes']} of {sum(manifest['chunk_bytes'])} bytes")
    out["seconds"] = time.time() - started
    mark("download_end")
    return out


def _truncated(path, size):
    with open(path, "wb") as f:
        f.truncate(size)
    return str(path)


# --------------------------------------------------------------------------------------------- upload

def home_tar(home, path):
    subprocess.run(["tar", "-C", str(home), "-cf", str(path), *HOME_TAR], check=True)
    return Path(path).stat().st_size


def upload(root, home, ckpt, dest, workers):
    """The GPU nodes' service account has read-only storage scopes: the caller passes a short-lived OAuth access
    token (gcloud auth print-access-token) on stdin; it is only held in this process's memory."""
    from google.cloud import storage
    from google.cloud.storage import transfer_manager as tm
    from google.oauth2.credentials import Credentials
    started, root = time.time(), Path(root)
    client = storage.Client(credentials=Credentials(sys.stdin.readline().strip()), project="sysres-disagg-ml")
    sb, sp = split(ckpt)
    db, dp = split(dest)
    src_bucket, dst_bucket = client.bucket(sb), client.bucket(db)
    meta = [b for b in client.list_blobs(sb, prefix=sp + "/") if "/" not in b.name[len(sp) + 1:]
            and not b.name.endswith(".safetensors")]
    for b in meta:
        src_bucket.copy_blob(b, dst_bucket, f"{dp}/model/{Path(b.name).name}")
    tar_bytes = home_tar(home, root / "cc.tar")
    snap = root / "snap"
    names = sorted(p.name for p in snap.iterdir() if p.name.endswith(".raw"))
    results = tm.upload_many_from_filenames(dst_bucket, names, source_directory=str(snap),
                                            blob_name_prefix=f"{dp}/snap/", max_workers=workers,
                                            worker_type="process", raise_exception=True)
    if any(isinstance(r, Exception) for r in results):
        raise RuntimeError(f"upload failures: {[r for r in results if isinstance(r, Exception)][:3]}")
    dst_bucket.blob(f"{dp}/cc.tar").upload_from_filename(str(root / "cc.tar"))
    dst_bucket.blob(f"{dp}/snap/{MANIFEST}").upload_from_filename(str(snap / MANIFEST))  # last: commit marker
    return {"dest": dest, "meta_files": len(meta), "chunks": len(names), "cc_bytes": tar_bytes,
            "snap_bytes": sum((snap / n).stat().st_size for n in names), "seconds": time.time() - started}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch-gcs", "fetch-peer"):
        p = sub.add_parser(name)
        p.add_argument("source")
        p.add_argument("--root", required=True)
        p.add_argument("--home", required=True)
        p.add_argument("--gate", required=True)
        p.add_argument("--workers", type=int, default=32)
        p.add_argument("--conns", type=int, default=16)
        p.add_argument("--no-snap", action="store_true", help="fetch-gcs: skip the snapshot")
        p.add_argument("--cc-uri", help="fetch-gcs: this cache tarball instead of <source>/cc.tar")
    p = sub.add_parser("serve")
    p.add_argument("--root", required=True)
    p.add_argument("--port", type=int, default=7070)
    p = sub.add_parser("upload")
    p.add_argument("--root", required=True)
    p.add_argument("--home", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--workers", type=int, default=32)
    p = sub.add_parser("home-tar")
    p.add_argument("--home", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "serve":
        return serve(a.root, a.port)
    if a.cmd == "home-tar":
        print(home_tar(a.home, a.out))
        return
    result = {"fetch-gcs": lambda: fetch_gcs(a.source, a.root, a.home, a.gate, a.workers, not a.no_snap, a.cc_uri),
              "fetch-peer": lambda: fetch_peer(a.source, a.root, a.home, a.gate, a.conns),
              "upload": lambda: upload(a.root, a.home, a.ckpt, a.dest, a.workers)}[a.cmd]()
    print("FETCH " + json.dumps({"kind": a.cmd, **result}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
