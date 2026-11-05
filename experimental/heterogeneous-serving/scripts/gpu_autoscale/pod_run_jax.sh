#!/bin/bash
# Pod side of one JAX-arm run (scripts/gpu_jax_autoscale.py): setup only, then the pod is a Ray worker
# and the driver submits the timed part (gpu_jax/hetero_jax_gpu/run_arm.sh) as a Ray job pinned to it
# through the custom resource run_$RUN_ID. Prints "MARK <event> <epoch seconds>" lines.
#   WIPE=1 (fresh-host arms on a reused node): wipe.py first, before any setup: page-cache eviction of the node
#          (image venv, stdlib, driver libs; drop_caches=3) plus a "WIPE <json>" receipt with residency before/after
#   setup (excluded from the bar, reported separately): gcsfuse mount (raw), Ray worker join
# Env: ARM, BUCKET, PREFIX, RUN_ID, RAY_ADDRESS (head GCS host:port), WIPE, NODE_NAME
set -euo pipefail
mark() { echo "MARK $1 $(date +%s.%N)"; }
mark pod_start
echo "ARM=$ARM BUCKET=$BUCKET PREFIX=$PREFIX RUN_ID=$RUN_ID HOST=$(hostname)"
head -1 /proc/driver/nvidia/version || true
if [[ ${WIPE:-0} == 1 ]]; then
  mark wipe_start
  python3 /scripts/wipe.py
  mark wipe_done
fi

case "$ARM" in
  stock-lazy)
    # Same gcsfuse build and mount options as the CUDA raw arm (pod_run.sh).
    command -v fusermount3 >/dev/null || { apt-get update -qq && apt-get install -y -qq fuse3 >/dev/null; }
    python3 /scripts/fetch.py object "gs://$BUCKET/tools/gcsfuse-sidecar-v1.23.30-gke.1" /usr/local/bin/gcsfuse
    chmod +x /usr/local/bin/gcsfuse
    mkdir -p /model
    gcsfuse --implicit-dirs --only-dir "$PREFIX" --log-file /tmp/gcsfuse.log "$BUCKET" /model
    ls -l /model
    ;;
esac

# run_$RUN_ID: 8 so a same-pod restart (warm-*) can start while the previous job's engine still serves
ray start --address="$RAY_ADDRESS" --num-gpus=1 --object-store-memory=2000000000 \
  --resources="{\"run_$RUN_ID\": 8}" --disable-usage-stats
mark setup_done
sleep infinity
