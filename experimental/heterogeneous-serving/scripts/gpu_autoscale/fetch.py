#!/usr/bin/env python3
"""In-pod GCS delivery for the GPU autoscale arms; prints one "FETCH {json}" line.

    fetch.py ckpt gs://bucket/prefix DEST [--workers 32]   every top-level object -> DEST/<basename>
    fetch.py tar gs://bucket/object.tar DEST               stream-extract a tarball into DEST
    fetch.py object gs://bucket/object PATH                one object -> PATH

`ckpt` is the TPU campaign's gcs_download_dir (scripts/startup_state.py): google-cloud-storage
transfer_manager sliced downloads with process workers and 64 MiB chunks; small files via download_many.
"""
import argparse
import json
import tarfile
import time
from pathlib import Path

CHUNK = 64 << 20


def split(uri):
    bucket, _, name = uri.removeprefix("gs://").partition("/")
    return bucket, name


def ckpt(uri, dest, workers):
    from google.cloud import storage
    from google.cloud.storage import transfer_manager as tm
    bucket, prefix = split(uri)
    prefix = prefix.rstrip("/") + "/"
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise RuntimeError(f"{dest} is not empty")
    blobs = [b for b in storage.Client().list_blobs(bucket, prefix=prefix) if "/" not in b.name[len(prefix):] and b.size]
    if not any(b.name.endswith(".safetensors") for b in blobs):
        raise RuntimeError(f"{uri}: no safetensors objects")
    started = time.time()
    small = [b for b in blobs if b.size < 4 * CHUNK]
    failed = [r for r in tm.download_many([(b, str(dest / Path(b.name).name)) for b in small], max_workers=workers,
                                          worker_type="process", raise_exception=True) if isinstance(r, Exception)]
    if failed:
        raise RuntimeError(f"download failures: {failed[:3]!r}")
    files = []
    for blob in (b for b in blobs if b.size >= 4 * CHUNK):
        began = time.time()
        tm.download_chunks_concurrently(blob, str(dest / Path(blob.name).name), chunk_size=CHUNK,
                                        max_workers=workers, worker_type="process")
        files.append({"name": Path(blob.name).name, "bytes": blob.size, "seconds": time.time() - began})
    for blob in blobs:
        if (dest / Path(blob.name).name).stat().st_size != blob.size:
            raise RuntimeError(f"size mismatch: {blob.name}")
    return {"files": len(blobs), "bytes": sum(b.size for b in blobs), "seconds": time.time() - started,
            "workers": workers, "large_files": files}


def tar(uri, dest):
    from google.cloud import storage
    bucket, name = split(uri)
    started = time.time()
    blob = storage.Client().bucket(bucket).blob(name)
    Path(dest).mkdir(parents=True, exist_ok=True)
    with blob.open("rb", chunk_size=CHUNK) as stream, tarfile.open(fileobj=stream, mode="r|*") as archive:
        archive.extractall(dest, filter="tar")
    return {"bytes": blob.size, "seconds": time.time() - started}


def one(uri, path):
    from google.cloud import storage
    bucket, name = split(uri)
    started = time.time()
    blob = storage.Client().bucket(bucket).blob(name)
    blob.download_to_filename(path)
    return {"bytes": Path(path).stat().st_size, "seconds": time.time() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=("ckpt", "tar", "object"))
    parser.add_argument("uri")
    parser.add_argument("dest")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    result = {"ckpt": lambda: ckpt(args.uri, args.dest, args.workers), "tar": lambda: tar(args.uri, args.dest),
              "object": lambda: one(args.uri, args.dest)}[args.kind]()
    print("FETCH " + json.dumps({"kind": args.kind, "uri": args.uri, **result}), flush=True)


if __name__ == "__main__":
    main()
