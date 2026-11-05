#!/bin/bash
# In-pod runner for one GPU autoscale arm (mounted from a ConfigMap by scripts/gpu_autoscale.py).
# Prints "MARK <event> <epoch seconds>" lines; the driver also reads kubectl log timestamps.
#   setup (excluded from the bar): gcsfuse mount (raw), google-cloud-storage install (ckpt / cache arms)
#   delivery (the bar's "download" segment): checkpoint -> tmpfs, peer compile cache -> $HOME
#   engine: `vllm serve` until /health answers, then one greedy completion as the output check
# Env: ARM, BUCKET, PREFIX, CC_URI (cache tarball for *-cc arms), FETCH_WORKERS, VLLM_EXTRA_ARGS
set -euo pipefail
mark() { echo "MARK $1 $(date +%s.%N)"; }
mark pod_start
echo "ARM=$ARM BUCKET=$BUCKET PREFIX=$PREFIX CC_URI=${CC_URI:-} HOST=$(hostname)"
head -1 /proc/driver/nvidia/version || true

if ! python3 -c 'import google.cloud.storage' 2>/dev/null; then
  python3 -m pip install --quiet --root-user-action=ignore 'google-cloud-storage>=2.14' >/dev/null
fi
case "$ARM" in
  raw*)
    # Same gcsfuse build as the TPU cluster's CSI sidecar (v1.23.30-gke.1 -> gcsfuse 3.10.0-gke.0) and the
    # same mount option (implicit-dirs); no file cache, so every read goes to GCS.
    command -v fusermount3 >/dev/null || { apt-get update -qq && apt-get install -y -qq fuse3 >/dev/null; }
    python3 /scripts/fetch.py object "gs://$BUCKET/tools/gcsfuse-sidecar-v1.23.30-gke.1" /usr/local/bin/gcsfuse
    chmod +x /usr/local/bin/gcsfuse
    mkdir -p /model
    gcsfuse --implicit-dirs --only-dir "$PREFIX" --log-file /tmp/gcsfuse.log "$BUCKET" /model
    ls -l /model
    ;;
esac
mark setup_done

mark download_start
case "$ARM" in
  ckpt*) python3 /scripts/fetch.py ckpt "gs://$BUCKET/$PREFIX" /model --workers "${FETCH_WORKERS:-32}" ;;
esac
case "$ARM" in
  *-cc) python3 /scripts/fetch.py tar "$CC_URI" "$HOME" ;;
esac
mark download_end

case "$ARM" in
  runai*) MODEL="gs://$BUCKET/$PREFIX"; LOAD=(--load-format runai_streamer) ;;
  *) MODEL=/model; LOAD=() ;;
esac
mark engine_start
# shellcheck disable=SC2086
vllm serve "$MODEL" "${LOAD[@]}" --served-model-name gemma --port 8000 \
  --max-model-len 8192 --max-num-seqs 512 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image": 0, "audio": 0, "video": 0}' --seed 0 \
  --disable-uvicorn-access-log ${VLLM_EXTRA_ARGS:-} &
vllm_pid=$!
until curl -sf -o /dev/null localhost:8000/health; do
  kill -0 "$vllm_pid" 2>/dev/null || { mark engine_died; wait "$vllm_pid"; exit 1; }
  sleep 0.1
done
mark ready
# Chat endpoint: the template supplies <bos>; Gemma's tokenizer does not add it to raw completion prompts.
check=$(curl -sf localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "gemma",
  "messages": [{"role": "user", "content": "Name the three primary colors in one short sentence."}],
  "max_tokens": 32, "temperature": 0}')
echo "CHECK $check"
mark check_done
wait "$vllm_pid"
