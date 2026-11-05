#!/bin/bash
# In-pod runner for one GPU autoscale arm (mounted from a ConfigMap by scripts/gpu_autoscale.py).
# Prints "MARK <event> <epoch seconds>" lines; the driver also reads kubectl log timestamps.
#   setup (excluded from the bar): gcsfuse mount (raw), google-cloud-storage install (ckpt / cache arms)
#   delivery (the bar's "download" segment): checkpoint -> tmpfs, peer compile cache -> $HOME
#   engine: `vllm serve` until /health answers, then one greedy chat completion as the output check
# Env: ARM, BUCKET, PREFIX, CC_URI (cache tarball for *-cc arms and warm-stock), FETCH_WORKERS, VLLM_EXTRA_ARGS,
#      STATE_SRC (pm-*: gs://bucket/prefix or http://peer:port), STATE_DST (pm-seed: gs://bucket/prefix),
#      SLLM_SRC / SLLM_WHEELS (sllm arms: ServerlessLLM checkpoint and wheelhouse prefixes), SLLM_DST (sllm-seed),
#      SLLM_THREADS / SLLM_CHUNK / SLLM_POOL (sllm-store --num-thread / --chunk-size / --mem-pool-size)
#
# Stock arms: raw, ckpt, ckpt-cc, runai, runai-cc (one fresh boot each).
# warm-stock: ckpt-cc delivery and boot (the "prime" boot), then the engine is stopped and restarted on the same
#   node with the checkpoint in tmpfs and every cache warm: T0 = relaunch. The restart's lines carry "W2| ".
# TPUFlex (scripts/gpu_autoscale/pm_cuda.py plugin, pm_zygote.py, pm_state.py):
#   pm-gcs / pm-peer: T0 = first byte of delivery. pm_state.py fetch streams the startup state (weight-less model
#     dir, bytecode, compile bank, terminal weight snapshot shards) from STATE_SRC; the engine (zygote + vllm
#     serve) launches as soon as the model dir and bytecode landed (~0.5 s), and restores the snapshot while the
#     shards keep arriving. pm-gcs then also runs warm-pm on the same node: a new engine is pre-spawned on port
#     8001 and parks before init_device while the first one serves; once parked, the first one is stopped and the
#     gate opens when it has exited (T0 = gate open; launch and stop times are recorded too).
#   pm-seed (setup, not an arm): builds the state (S1-S4 below), uploads it to STATE_DST, then keeps serving and
#     serves the state over HTTP (pm-peer's peer). With STATE_SRC instead: boots from that published state as
#     pm-gcs does, keeps the delivered objects as a pack and serves them (a live peer in ~1 min, same state).
# ServerlessLLM (scripts/gpu_autoscale/sllm_vllm.py plugin + sllm_loader.py, sllm-store 0.8.0 @ SLLM_COMMIT):
#   setup: the sllm-store wheelhouse (built in this image by sllm-seed) is installed, and the per-node store
#     daemon is started (`sllm-store start`: pins its memory pool, then listens on :8073). Like a DaemonSet on a
#     node that has just booted, it is up before the serving pod's T0 (its start-up is recorded, not counted).
#   sllm / sllm-cc: T0 = first byte of delivery. The ServerlessLLM-format checkpoint (rank_0/tensor.data_*) is
#     downloaded to tmpfs exactly as ckpt downloads the safetensors (same fetch, 32 process workers); sllm-cc also
#     gets the stock compile cache. vllm serve /model --load-format serverless_llm: the engine builds the model on
#     the CPU, then load_dict has the store read the checkpoint into its pinned pool and copy it into the GPU.
#   warm-sllm: sllm-cc prime boot, then a restart in the same pod (T0 = relaunch, "W2| "): the checkpoint is in
#     the store's pinned pool (ServerlessLLM's DRAM tier) and every cache is warm.
#   sllm-seed (setup, not an arm): builds sllm-store from source in this image and a wheelhouse, converts the
#     checkpoint with ServerlessLLM's vLLM writer, verifies it (S1 cold boot, S2 restart from the pool, each with
#     the greedy check), times bare load_dict through the store per I/O thread count, and uploads both to SLLM_DST.
set -euo pipefail
mark() { echo "MARK $1 $(date +%s.%N)"; }
mark pod_start
echo "ARM=$ARM BUCKET=$BUCKET PREFIX=$PREFIX CC_URI=${CC_URI:-} STATE_SRC=${STATE_SRC:-} HOST=$(hostname)"
head -1 /proc/driver/nvidia/version || true
if [ "${SLLM_STORE_ISOLATE:-0}" = 1 ]; then  # sllm arms: the store daemon's private site-packages (see start_store),
  # copied before the wipe so that the wipe evicts everything the copy read and wrote
  SITE=$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("torch").origin)))')
  mark sllm_isolate_copy_start
  cp -a "$SITE" /cache/store-site
  mark sllm_isolate_copy_done
  echo "SLLM_ISOLATE $SITE -> /cache/store-site ($(du -sb /cache/store-site | cut -f1) B)"
fi
if [ "${WIPE:-0}" = 1 ]; then  # reused node: fresh-host emulation + receipt (scripts/gpu_autoscale/wipe.py)
  python3 /scripts/wipe.py || { mark wipe_failed; exit 1; }
  mark wipe_done
fi

ENGINE_ARGS=(--served-model-name gemma --max-model-len 8192 --max-num-seqs 512 --max-num-batched-tokens 2048
  --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image": 0, "audio": 0, "video": 0}' --seed 0
  --disable-uvicorn-access-log)
PMROOT=/cache/pm SNAP=/dev/shm/pm/snap PYC=/cache/pyc SLLM_SITE=/cache/sllm/site
SLLM_THREADS=${SLLM_THREADS:-4} SLLM_CHUNK=${SLLM_CHUNK:-32MB} SLLM_POOL=${SLLM_POOL:-64GB}
# a ServerlessLLM engine: its plugin, the store's storage root (/), and the load format
SLLM_ENV=(PYTHONPATH="$SLLM_SITE" SLLM_PLUGIN=1 STORAGE_PATH=/)
SLLM_LOAD=(--load-format serverless_llm)

check() {  # $1 port; chat endpoint: the template supplies <bos>, Gemma's tokenizer does not add it to raw prompts
  curl -sf "localhost:$1/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model": "gemma",
    "messages": [{"role": "user", "content": "Name the three primary colors in one short sentence."}],
    "max_tokens": 32, "temperature": 0}'
}
wait_health() {  # $1 port, $2 session id, $3 mark prefix, [$4 fetch pid: a failed delivery ends the run too]
  until curl -sf -o /dev/null "localhost:$1/health"; do
    alive "$2" || { mark "${3}engine_died"; exit 1; }
    if [ -n "${4:-}" ] && ! kill -0 "$4" 2>/dev/null; then
      wait "$4" || { mark fetch_failed; stop_session "$2"; mark "${3}engine_died"; exit 1; }
      set -- "$1" "$2" "$3"  # delivered: stop checking
    fi
    sleep 0.1
  done
}
alive() {  # $1 session id: any live (non-zombie) process left in it? Orphans are reaped by PID 1, which in a
  # `sleep infinity` dev pod never happens
  ps -o stat= -s "$1" 2>/dev/null | grep -qv '^Z'
}
stop_session() {  # $1 session id: stop every process of that engine (API server, EngineCore, zygote)
  pkill -TERM -s "$1" 2>/dev/null || true
  for _ in $(seq 300); do alive "$1" || return 0; sleep 0.1; done
  pkill -KILL -s "$1" 2>/dev/null || true
  while alive "$1"; do sleep 0.1; done
}
triton_count() {  # $1 point: Triton's own JIT cache ($HOME/.triton/cache) kernel binaries, one .cubin per variant
  echo "TRITON_CACHE at=$1 cubins=$(find "$HOME/.triton/cache" -name '*.cubin' 2>/dev/null | wc -l)"
}
install_pm() {  # the plugin as a vllm.general_plugins entry point on PYTHONPATH
  mkdir -p "$PMROOT/site/pm_cuda-0.0.0.dist-info" "$SNAP" "$PYC"
  cp /scripts/pm_cuda.py "$PMROOT/site/pm_cuda.py"
  printf 'Metadata-Version: 2.1\nName: pm_cuda\nVersion: 0.0.0\n' > "$PMROOT/site/pm_cuda-0.0.0.dist-info/METADATA"
  printf '[vllm.general_plugins]\npm_cuda = pm_cuda:register\n' > "$PMROOT/site/pm_cuda-0.0.0.dist-info/entry_points.txt"
}
install_sllm_plugin() {  # sllm_vllm.py (+ the verbatim sllm_loader.py) as a vllm.general_plugins entry point
  mkdir -p "$SLLM_SITE/sllm_vllm-0.0.0.dist-info"
  cp /scripts/sllm_vllm.py /scripts/sllm_loader.py "$SLLM_SITE/"
  printf 'Metadata-Version: 2.1\nName: sllm_vllm\nVersion: 0.0.0\n' > "$SLLM_SITE/sllm_vllm-0.0.0.dist-info/METADATA"
  printf '[vllm.general_plugins]\nsllm_vllm = sllm_vllm:register\n' > "$SLLM_SITE/sllm_vllm-0.0.0.dist-info/entry_points.txt"
}
install_sllm() {  # sllm-store and the runtime deps the image lacks, from the wheelhouse sllm-seed built in this image
  mkdir -p /cache/sllm/wheels
  python3 /scripts/fetch.py tree "$SLLM_WHEELS" /cache/sllm/wheels --workers 8
  python3 -m pip install --quiet --root-user-action=ignore --no-index --no-deps /cache/sllm/wheels/*.whl
  echo "SLLM_WHEELS $(cd /cache/sllm/wheels && ls *.whl | tr '\n' ' ')"
  install_sllm_plugin
}
start_store() {  # [I/O threads]: ServerlessLLM's per-node store daemon, own session; its pool is pinned before it listens.
  # SLLM_STORE_ISOLATE=1: the daemon runs from a private copy of the image's site-packages (/cache/store-site,
  # copied before the wipe, plus the wheelhouse; as from its own container image) and the page cache is dropped
  # again once it listens (WIPE_T0 receipt), so the engine's libraries are as cold at T0 as in the stock arms.
  # Otherwise the daemon's torch/CUDA libraries are the engine's own files, which it has just paged in.
  local threads=${1:-$SLLM_THREADS} pypath=""
  if [ "${SLLM_STORE_ISOLATE:-0}" = 1 ]; then
    [ -d /cache/store-site/sllm_store ] || python3 -m pip install --quiet --root-user-action=ignore --no-index \
      --no-deps --target /cache/store-site /cache/sllm/wheels/*.whl
    pypath=/cache/store-site
  fi
  mark sllm_store_start
  # shellcheck disable=SC2016
  setsid env ${pypath:+PYTHONPATH=$pypath} bash -c 'sllm-store start --storage-path / --mem-pool-size "$1" \
    --num-thread "$2" --chunk-size "$3" 2>&1 | sed -u "s/^/SLLM| /"' _ "$SLLM_POOL" "$threads" "$SLLM_CHUNK" &
  STORE_SID=$!
  until (exec 3<>/dev/tcp/127.0.0.1/8073) 2>/dev/null; do
    # the session leader itself, not `alive` (ps -s): right after the fork the session may not exist yet
    # (sllm-2-8w48 was marked dead 8 ms after the start, behind a 17.7 GB copy)
    kill -0 "$STORE_SID" 2>/dev/null || { mark sllm_store_died; exit 1; }
    sleep 0.1
  done
  mark sllm_store_ready
  echo "SLLM_STORE threads=$threads chunk=$SLLM_CHUNK pool=$SLLM_POOL isolate=${SLLM_STORE_ISOLATE:-0}"
  echo "SLLM_STORE_LIBS $(for p in $(pgrep -s "$STORE_SID"); do grep -o '/[^ ]*/libtorch_cuda\.so' "/proc/$p/maps" 2>/dev/null
    done | sort -u | tr '\n' ' ')"
  if [ "${SLLM_STORE_ISOLATE:-0}" = 1 ]; then
    python3 /scripts/wipe.py --drop-only
    mark sllm_isolate_dropped
  fi
}
# launch_engine PORT MODEL TAG [ENV=VALUE ...] -- [extra vllm args]: own session; TAG prefixes its log lines.
# PM engines (PM_ZYGOTE_SOCK in the env) start the zygote first, in the same session (fork safety: pm_zygote.py).
launch_engine() {
  local port=$1 model=$2 tag=$3; shift 3
  local envs=() extra=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
  [ "${1:-}" = -- ] && shift
  extra=("$@")
  local zyg=""
  for e in "${envs[@]}"; do case "$e" in PM_ZYGOTE_SOCK=*) zyg=1;; esac; done
  # shellcheck disable=SC2016
  setsid env "${envs[@]}" bash -c '
    port=$1 model=$2 tag=$3 zyg=$4; shift 4
    { [ -n "$zyg" ] && python3 /scripts/pm_zygote.py & vllm serve "$model" --port "$port" "$@"; } 2>&1 |
      sed -u "s/^/$tag/"' _ "$port" "$model" "$tag" "$zyg" "${ENGINE_ARGS[@]}" ${VLLM_EXTRA_ARGS:-} "${extra[@]}" &
  ENGINE_SID=$!
}

warm_stock_restart() {  # $1 serving session, $2 model dir, [ENV=VALUE ... -- vllm args]: restart on the same node
  local sid=$1 model=$2; shift 2  # (T0 = relaunch)
  [ $# -gt 0 ] || set -- --
  mark warm_stop
  stop_session "$sid"
  mark warm_launch
  launch_engine 8001 "$model" "W2| " "$@"
  wait_health 8001 "$ENGINE_SID" warm_
  mark warm_ready
  echo "W2| CHECK $(check 8001)"
  mark warm_check_done
  stop_session "$ENGINE_SID"
}

if ! python3 -c 'import google.cloud.storage' 2>/dev/null; then
  case "$ARM" in pm-gcs|pm-peer) ;; *)
    python3 -m pip install --quiet --root-user-action=ignore 'google-cloud-storage>=2.14' >/dev/null ;;
  esac
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
  pm-*) install_pm ;;
  sllm-seed) ;;  # builds its own wheelhouse below
  sllm*|warm-sllm) install_sllm; start_store ;;
esac
mark setup_done

# ------------------------------------------------------------------------------------------ ServerlessLLM seed
if [ "$ARM" = sllm-seed ]; then
  SLLM_COMMIT=${SLLM_COMMIT:-9f50241baa5386e06a9321c51f19a9ef5f964c2b}
  WH=/cache/sllm/wheels SRC=/cache/sllm/src
  mkdir -p "$WH"
  if [ "${SLLM_SEED_REUSE:-0}" = 1 ]; then  # development: the published wheelhouse + checkpoint; steps 3-4 only
    SLLM_WHEELS=$SLLM_DST/wheels
    install_sllm
    python3 /scripts/fetch.py tree "$SLLM_DST/$PREFIX" /model --workers "${FETCH_WORKERS:-32}"
  else
    # 1. sllm-store from source against this image's torch / CUDA / Python (PyPI has a cp310 wheel only, built
    #    against another torch), plus the runtime deps the image lacks, as a wheelhouse
    mark sllm_build_start
    CU=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\),.*/\1-\2/p')
    apt-get update -qq
    for p in git "cuda-cudart-dev-$CU" "cuda-nvrtc-dev-$CU" "cuda-driver-dev-$CU" "cuda-nvtx-$CU" "libcufft-dev-$CU" \
        "libcusparse-dev-$CU" "libcusolver-dev-$CU" "libnvjitlink-dev-$CU" "cuda-nvml-dev-$CU"; do
      if apt-get install -y -qq --no-install-recommends "$p" >/dev/null 2>&1; then echo "apt: $p"; else echo "apt: $p unavailable"; fi
    done
    python3 -m pip install --quiet --root-user-action=ignore 'cmake>=3.20,<4.0.0' ninja
    git clone --quiet https://github.com/ServerlessLLM/ServerlessLLM.git "$SRC"
    git -C "$SRC" checkout --quiet "$SLLM_COMMIT"
    echo "SLLM_SRC_COMMIT $(git -C "$SRC" rev-parse HEAD)"
    awk '/^\+\+\+ b\/vllm\/model_executor\/model_loader\/sllm_loader.py/{f=1;next} f&&/^diff --git/{f=0}
         f&&/^@@/{next} f{print substr($0,2)}' "$SRC/sllm_store/vllm_patch/sllm_load.patch" | cmp - /scripts/sllm_loader.py ||
      { echo "SLLM_LOADER differs from the patch's sllm_loader.py"; mark sllm_build_failed; exit 1; }
    echo "SLLM_LOADER verbatim: /scripts/sllm_loader.py == the patch's sllm_loader.py"
    # sllm_store_fixes.patch (both found on this H100, 2026-10-29): (a) the writer's direct pwrite of a > 2 GiB
    # tensor came back short (Linux moves <= 0x7ffff000 bytes per call) and the tensor was silently dropped
    # (Gemma-4's 2.8 GB embedding): loop until written; (b) load_dict packed the tensors back to back in GPU
    # memory, leaving half of them only 8-byte aligned (slow kernels, CUDA graph capture 24 s instead of 8 s):
    # start every tensor on a 256-byte boundary
    git -C "$SRC" apply /scripts/sllm_store_fixes.patch
    echo "SLLM_STORE_FIXES $(sha256sum /scripts/sllm_store_fixes.patch | cut -c1-12): $(git -C "$SRC" diff --stat | tail -n 1)"
    (cd "$SRC/sllm_store" && TORCH_CUDA_ARCH_LIST=9.0 python3 -m pip wheel --no-build-isolation --no-deps -w "$WH" . \
      > /cache/sllm/build.log 2>&1) || { tail -n 200 /cache/sllm/build.log; mark sllm_build_failed; exit 1; }
    tail -n 5 /cache/sllm/build.log
    for spec in 'grpcio==1.76.0 grpc' 'accelerate>=1.7.0 accelerate' 'peft>=0.16.0 peft' 'click==8.1.7 click'; do
      req=${spec% *} mod=${spec#* }  # sllm_store/requirements.txt pins; an image's own copy is kept as is
      if python3 -c "import $mod" 2>/dev/null; then echo "dep: $mod from the image"
      else python3 -m pip download --quiet --no-deps -d "$WH" "$req"; fi
    done
    python3 -m pip install --quiet --root-user-action=ignore --no-index --no-deps "$WH"/*.whl
    python3 -m pip check || true
    python3 -c 'import sllm_store, sllm_store._C, sllm_store.server; print("SLLM_IMPORT ok", sllm_store.__file__)'
    echo "SLLM_WHEELS $(cd "$WH" && ls *.whl | tr '\n' ' ')"
    install_sllm_plugin
    mark sllm_build_done
    python3 /scripts/sllm_vllm.py upload "$WH" "$SLLM_DST/wheels" || { mark sllm_seed_upload_failed; exit 1; }
    # 2. conversion: ServerlessLLM's vLLM writer, in place (/model/rank_0) unless this tmpfs refuses O_DIRECT,
    #    which the writer opens its partitions with
    python3 /scripts/fetch.py ckpt "gs://$BUCKET/$PREFIX" /model --workers "${FETCH_WORKERS:-32}"
    OUT=/model
    python3 -c 'import os; os.close(os.open("/model/.odirect", os.O_WRONLY | os.O_CREAT | os.O_DIRECT))' 2>/dev/null ||
      OUT=/cache/sllm/out
    rm -f /model/.odirect
    echo "SLLM_SAVE_TO $OUT"
    mark sllm_save_start
    env PYTHONPATH="$SLLM_SITE" python3 /scripts/sllm_vllm.py save --model /model --out "$OUT" --delete-source 2>&1 |
      sed -u 's/^/S0| /' || { mark sllm_save_failed; exit 1; }
    [ "$OUT" = /model ] || { cp -a "$OUT/rank_0" /model/ && rm -rf "$OUT"; }
    mark sllm_save_done
    ls -la /model /model/rank_0
    python3 /scripts/sllm_vllm.py upload /model "$SLLM_DST/$PREFIX" || { mark sllm_seed_upload_failed; exit 1; }
  fi
  # 3. verification through the store: S1 cold (compile cache empty, store pool empty), S2 restart (both warm)
  start_store
  mark seed1_start
  launch_engine 8000 /model "S1| " "${SLLM_ENV[@]}" -- "${SLLM_LOAD[@]}"
  wait_health 8000 "$ENGINE_SID" seed1_
  mark seed1_ready
  echo "S1| CHECK $(check 8000)"
  stop_session "$ENGINE_SID"
  mark seed2_start
  launch_engine 8000 /model "S2| " "${SLLM_ENV[@]}" -- "${SLLM_LOAD[@]}"
  wait_health 8000 "$ENGINE_SID" seed2_
  mark seed2_ready
  echo "S2| CHECK $(check 8000)"
  stop_session "$ENGINE_SID"
  stop_session "$STORE_SID"
  # 4. bare load_dict through a fresh store per I/O thread count (pool empty), then once more from its pool
  for t in ${SLLM_BENCH_THREADS:-4 8 16}; do
    start_store "$t"
    python3 /scripts/sllm_vllm.py bench --label "threads=$t pool-empty"
    python3 /scripts/sllm_vllm.py bench --label "threads=$t pool-hit"
    stop_session "$STORE_SID"
  done
  mark sllm_seed_done
  exit 0
fi

# ---------------------------------------------------------------------------------------------- TPUFlex
PM_RESTORE_ENV=(PYTHONPATH="$PMROOT/site" PYTHONPYCACHEPREFIX="$PYC" PM_SNAP_DIR="$SNAP" PM_SNAP_MODE=restore
  PM_WARM=1 PM_SNAP_THREADS="${PM_SNAP_THREADS:-2}" PM_SNAP_PIN="${PM_SNAP_PIN:-0}"
  PM_SNAP_JOIN="${PM_SNAP_JOIN:-serve}")
if [ "$ARM" = pm-seed ] && [ -n "${STATE_SRC:-}" ]; then
  # pm-peer's live peer, built from a published state instead of S1-S3: a TPUFlex boot from STATE_SRC (as
  # pm-gcs) that keeps the delivered objects as a pack, then serves them on :8080 next to the serving engine.
  python3 /scripts/pm_state.py fetch --src "$STATE_SRC" --root "$PMROOT" --home "$HOME" --model-dir /model \
    --pyc-dir "$PYC" --snap-dir "$SNAP" --workers "${PM_FETCH_WORKERS:-3}" --keep-pack "$PMROOT/pack" &
  FETCH_PID=$!
  until [ -e "$PMROOT/.model_done" ]; do
    kill -0 "$FETCH_PID" 2>/dev/null || { mark engine_died; exit 1; }
    sleep 0.01
  done
  KV=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["meta"]["kv_bytes"])' "$PMROOT/files.json")
  mark seed4_start
  launch_engine 8000 /model "S4| " "${PM_RESTORE_ENV[@]}" PM_BANK_DONE="$PMROOT/.bank_done" \
    PM_ZYGOTE_SOCK=/tmp/pm_zygote_s4.sock -- --kv-cache-memory-bytes "$KV"
  wait_health 8000 "$ENGINE_SID" seed4_ "$FETCH_PID"
  echo "S4| CHECK $(check 8000)"
  wait "$FETCH_PID" 2>/dev/null || true
  python3 /scripts/pm_state.py serve --root "$PMROOT/pack" --port 8080 &
  mark seed_ready
  wait "$ENGINE_SID"
  exit 1
fi
if [ "$ARM" = pm-seed ]; then
  # S1 stock loader + snapshot save (cold compile). S2 PM restore, unpinned, warm stock-key cache: its memory
  # profile gives the KV bytes of a warm-cache stock boot (a cold-compile boot leaves ~0.7 GiB less: 30,963 vs
  # 32,949 tokens in campaign-549bd16). S3 PM restore pinned to those bytes (the KV bytes are part of the compile
  # key): compiles the bank. Pack + upload. S4 PM restore pinned, warm bank: verification, then the live peer.
  # Development only: SEED_REUSE=1 keeps an existing snapshot + checkpoint (starts at S2), SEED_KV=<bytes> skips S2.
  if [ "${SEED_REUSE:-0}" = 1 ] && [ -e "$SNAP/manifest.json" ]; then
    echo "seed: reusing $(ls "$SNAP" | wc -l) snapshot files"
  else
    python3 /scripts/fetch.py ckpt "gs://$BUCKET/$PREFIX" /model --workers "${FETCH_WORKERS:-32}"
    mark seed1_start
    launch_engine 8000 /model "S1| " PYTHONPATH="$PMROOT/site" PYTHONPYCACHEPREFIX="$PYC" PM_SNAP_DIR="$SNAP" \
      PM_SNAP_MODE=save
    wait_health 8000 "$ENGINE_SID" seed1_
    echo "S1| CHECK $(check 8000)"
    stop_session "$ENGINE_SID"
  fi
  if [ -n "${SEED_KV:-}" ]; then  # development: KV bytes of an earlier S2
    KV=$SEED_KV
  else
    mark seed2_start
    launch_engine 8000 /model "S2| " "${PM_RESTORE_ENV[@]}" PM_ZYGOTE_SOCK=/tmp/pm_zygote_s2.sock \
      > >(tee "$PMROOT/s2.log") 2>&1
    wait_health 8000 "$ENGINE_SID" seed2_
    echo "S2| CHECK $(check 8000)"
    stop_session "$ENGINE_SID"
    KV=$(grep -o 'PM_KV_BYTES bytes=[0-9]*' "$PMROOT/s2.log" | head -1 | cut -d= -f2)
  fi
  echo "seed: KV bytes of the warm-cache boot = $KV ($(grep -o 'GPU KV cache size: [0-9,]* tokens' "$PMROOT/s2.log"))"
  rm -rf "$HOME/.cache/vllm/torch_compile_cache"  # the unpinned key's; the bank keeps the pinned key only
  mark seed3_start
  launch_engine 8000 /model "S3| " "${PM_RESTORE_ENV[@]}" PM_ZYGOTE_SOCK=/tmp/pm_zygote_s3.sock -- \
    --kv-cache-memory-bytes "$KV"
  wait_health 8000 "$ENGINE_SID" seed3_
  echo "S3| CHECK $(check 8000)"
  stop_session "$ENGINE_SID"
  python3 /scripts/pm_state.py pack --model-dir /model --pyc-dir "$PYC" --home "$HOME" --snap-dir "$SNAP" \
    --out "$PMROOT/pack" --meta "{\"kv_bytes\": $KV}"
  if [ -n "${STATE_DST:-}" ]; then
    python3 /scripts/pm_state.py upload --pack "$PMROOT/pack" --dst "$STATE_DST" || { mark seed_upload_failed; exit 1; }
  fi
  mark seed4_start
  launch_engine 8000 /model "S4| " "${PM_RESTORE_ENV[@]}" PM_ZYGOTE_SOCK=/tmp/pm_zygote_s4.sock -- \
    --kv-cache-memory-bytes "$KV"
  wait_health 8000 "$ENGINE_SID" seed4_
  echo "S4| CHECK $(check 8000)"
  python3 /scripts/pm_state.py serve --root "$PMROOT/pack" --port 8080 &
  mark seed_ready
  wait "$ENGINE_SID"
  exit 1
fi
if [ "$ARM" = pm-gcs ] || [ "$ARM" = pm-peer ]; then
  # PM_LOO (leave-one-out ablation, gpu_autoscale.py loo-* arms; empty = full TPUFlex): one mechanism off,
  # its phase falls back to what the best stock arm (runai-cc) does for it:
  #   snapshot  no snapshot delivered or restored: the stock Run:ai loader streams the safetensors from gs://
  #             (PM_WEIGHTS_URI; the model stays /model, the compile key's model path)
  #   bank      no compile bank in the delivery: the stack's own compile cache for this key (CC_URI) is
  #             downloaded into $HOME before the engine launches, as the *-cc arms do
  #   bytecode  no pyc.tar and no PYTHONPYCACHEPREFIX: the image's own bytecode handling
  #   kvpin     no --kv-cache-memory-bytes (vLLM profiles KV and CUDA-graph memory); the pin is part of the
  #             compile key, so the bank goes too and CC_URI is the stock cache of the unpinned key
  #   zygote    no PM_ZYGOTE_SOCK: vLLM spawns the EngineCore (fresh interpreter)
  #   overlap   the delivery runs to completion before the engine launches
  #   warmup    PM_WARM=0
  LOO=${PM_LOO:-}
  PM_ENV=("${PM_RESTORE_ENV[@]}" PM_BANK_DONE="$PMROOT/.bank_done")
  drop_env() { local out=() e; for e in "${PM_ENV[@]}"; do [ "${e%%=*}" = "$1" ] || out+=("$e"); done; PM_ENV=("${out[@]}"); }
  SKIP="" ZYG=(PM_ZYGOTE_SOCK=/tmp/pm_zygote_1.sock) LOAD=() CC=""
  case "$LOO" in
    "") ;;
    snapshot) SKIP=snap; drop_env PM_SNAP_DIR; drop_env PM_SNAP_MODE; PM_ENV+=(PM_WEIGHTS_URI="gs://$BUCKET/$PREFIX")
      LOAD=(--load-format runai_streamer) ;;
    bank) SKIP=bank; drop_env PM_BANK_DONE; CC=1 ;;
    bytecode) SKIP=pyc; drop_env PYTHONPYCACHEPREFIX ;;
    kvpin) SKIP=bank; drop_env PM_BANK_DONE; CC=1 ;;
    zygote) ZYG=() ;;
    overlap) ;;
    warmup) drop_env PM_WARM; PM_ENV+=(PM_WARM=0) ;;
    *) echo "unknown PM_LOO=$LOO"; mark engine_died; exit 1 ;;
  esac
  if [ -n "$CC" ] && [ -z "${CC_URI:-}" ]; then echo "PM_LOO=$LOO needs CC_URI"; mark engine_died; exit 1; fi
  echo "PM_LOO=${LOO:-none} skip=${SKIP:-none} cc=${CC_URI:-none} env: ${PM_ENV[*]} ${ZYG[*]}"
  mark download_start
  if [ -n "$CC" ]; then  # the stock compile-cache delivery (fetch.py tar, as the *-cc arms), next to the PM one
    python3 /scripts/fetch.py tar "$CC_URI" "$HOME" &
    CC_PID=$!
  fi
  python3 /scripts/pm_state.py fetch --src "$STATE_SRC" --root "$PMROOT" --home "$HOME" --model-dir /model \
    --pyc-dir "$PYC" --snap-dir "$SNAP" --workers "${PM_FETCH_WORKERS:-3}" --skip "$SKIP" &
  FETCH_PID=$!
  until [ -e "$PMROOT/.model_done" ]; do
    kill -0 "$FETCH_PID" 2>/dev/null || { mark engine_died; exit 1; }
    sleep 0.01
  done
  if [ -n "$CC" ]; then
    wait "$CC_PID" || { mark fetch_failed; mark engine_died; exit 1; }
    mark cc_done
  fi
  WAIT_FETCH=$FETCH_PID
  if [ "$LOO" = overlap ]; then
    wait "$FETCH_PID" || { mark fetch_failed; mark engine_died; exit 1; }
    mark fetch_done
    WAIT_FETCH=""
  fi
  KV=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["meta"]["kv_bytes"])' "$PMROOT/files.json")
  PIN=(--kv-cache-memory-bytes "$KV")
  [ "$LOO" = kvpin ] && PIN=()
  triton_count pre_launch
  mark engine_start
  launch_engine 8000 /model "" "${PM_ENV[@]}" "${ZYG[@]}" -- "${PIN[@]}" "${LOAD[@]}"
  SID1=$ENGINE_SID
  wait_health 8000 "$SID1" "" $WAIT_FETCH
  mark ready
  triton_count ready
  echo "CHECK $(check 8000)"
  mark check_done
  triton_count check_done
  wait "$FETCH_PID" 2>/dev/null || true  # reaped by wait_health if delivery ended first; FETCH has the times
  if [ "$ARM" = pm-gcs ] && [ -z "$LOO" ]; then  # warm-pm on the same node (full TPUFlex only)
    rm -f "$PMROOT/gate2"
    mark warm_launch
    launch_engine 8001 /model "W2| " "${PM_ENV[@]}" PM_ZYGOTE_SOCK=/tmp/pm_zygote_2.sock \
      PM_GATE_FILE="$PMROOT/gate2" PM_BANK_DONE= -- --kv-cache-memory-bytes "$KV" > >(tee "$PMROOT/w2.log") 2>&1
    SID2=$ENGINE_SID
    until grep -q PM_PARKED "$PMROOT/w2.log" 2>/dev/null; do
      alive "$SID2" || { mark warm_engine_died; exit 1; }
      sleep 0.02
    done
    mark warm_parked
    mark warm_stop
    stop_session "$SID1"
    mark warm_gate
    touch "$PMROOT/gate2"
    wait_health 8001 "$SID2" warm_
    mark warm_ready
    echo "W2| CHECK $(check 8001)"
    mark warm_check_done
    stop_session "$SID2"
  fi
  exit 0
fi

# ------------------------------------------------------------------------- stock (and ServerlessLLM's loader)
mark download_start
case "$ARM" in
  ckpt*|warm-stock) python3 /scripts/fetch.py ckpt "gs://$BUCKET/$PREFIX" /model --workers "${FETCH_WORKERS:-32}" ;;
  sllm*|warm-sllm) python3 /scripts/fetch.py tree "$SLLM_SRC" /model --workers "${FETCH_WORKERS:-32}" ;;
esac
case "$ARM" in
  *-cc|warm-stock|warm-sllm) python3 /scripts/fetch.py tar "$CC_URI" "$HOME" ;;
esac
mark download_end

EXTRA_ENV=()
case "$ARM" in
  runai*) MODEL="gs://$BUCKET/$PREFIX"; LOAD=(--load-format runai_streamer) ;;
  sllm*|warm-sllm) MODEL=/model; EXTRA_ENV=("${SLLM_ENV[@]}"); LOAD=("${SLLM_LOAD[@]}") ;;
  *) MODEL=/model; LOAD=() ;;
esac
triton_count pre_launch
mark engine_start
launch_engine 8000 "$MODEL" "" "${EXTRA_ENV[@]}" -- "${LOAD[@]}"
SID1=$ENGINE_SID
wait_health 8000 "$SID1" ""
mark ready
triton_count ready
echo "CHECK $(check 8000)"
mark check_done
triton_count check_done
case "$ARM" in  # restart on the same node: checkpoint in tmpfs (and, for warm-sllm, in the store's pool), caches warm
  warm-stock|warm-sllm) warm_stock_restart "$SID1" /model "${EXTRA_ENV[@]}" -- "${LOAD[@]}"; exit 0 ;;
esac
wait "$SID1"
