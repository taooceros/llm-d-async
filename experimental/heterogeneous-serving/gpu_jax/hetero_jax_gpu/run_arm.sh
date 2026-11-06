#!/bin/bash
# Timed part of one vLLM-JAX run, submitted as a Ray job (working_dir = gpu_jax/) onto the run's own GPU pod by
# scripts/gpu_jax_autoscale.py. T0 = MARK download_start (first byte of delivery, or engine launch when nothing is
# delivered; warm-pm: gate open). Arms (canonical names, see the driver):
#   stock-lazy                          lazy safetensors reads through the pod's gcsfuse mount of the checkpoint
#   stock-ckpt / stock-ckpt-cc          checkpoint -> tmpfs (sliced GCS GETs) / + a peer's $HOME caches (CC_URI)
#   stock-runai / stock-runai-cc        vLLM --load-format runai_streamer from gs:// / + CC_URI
#   warm-stock                          same pod after stock-ckpt: restart, checkpoint in tmpfs, stock cache warm
#   pm-seed                             (untimed) stock-ckpt boot that writes the TPUFlex state to /pmstate
#   pm-gcs / pm-peer                    TPUFlex: engine launched at T0 next to the state delivery
#                                       (scripts/gpu_autoscale/pm_state.py fetch-gcs PM_STATE_URI | fetch-peer PM_PEER)
#   warm-pm                             same pod after pm-gcs: pre-spawned engine parks, the old engine exits, gate opens
#   loo-<mech>                          leave-one-out (results/gpu-jax/loo-*): pm-gcs with ONE mechanism replaced by
#                                       what stock-runai-cc does for that phase:
#     loo-snapshot      no snapshot: stock loader, weights via vLLM's runai_streamer from gs:// (rest of PM on)
#     loo-cache0        stock-runai-cc's cache tar (CC_URI, JAX's default 1 s threshold) instead of the state's
#                       0 s cache; no JAX cache env from process start (tpu-inference enables its cache as in stock)
#     loo-zygote        PM_ZYGOTE=0: vLLM's stock spawn of the engine core
#     loo-overlap       sequential: the whole state is fetched, then the engine is launched
#     loo-read-threads  restore from np.memmap views with 4 threads (the pre-bfb02b9 restore) instead of read + 32
#     loo-snapshot-cache0  factorial cell: both loo-snapshot's and loo-cache0's stand-ins (results/gpu-jax/loo-factorial-*)
# Env (runtime_env): ARM, BUCKET, PREFIX, CC_URI, PM_STATE_URI, PM_PEER, PM_KV_BYTES, KEEP, HETERO_JAX_GPU=1, ...
set -euo pipefail
mark() { echo "MARK $1 $(date +%s.%N)"; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# A Ray job entrypoint runs with OMP_NUM_THREADS=1 (entrypoint_num_cpus=0); vLLM then keeps torch at one thread for
# weight loading (+100 s). Unset, vLLM picks its own startup thread count and logs "Reducing Torch threads" for
# serving, exactly as a vllm serve outside Ray (and the CUDA vLLM pods).
echo "RAY_OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
unset OMP_NUM_THREADS
mark job_start
echo "ARM=$ARM CC_URI=${CC_URI:-} PM_STATE_URI=${PM_STATE_URI:-} PM_PEER=${PM_PEER:-} HOST=$(hostname) WORKING_DIR=$PWD"
keep=(); [[ ${KEEP:-0} == 1 ]] && keep=(--keep)
ROOT=/pmstate

pm_env() {  # TPUFlex engine settings (gpu_jax/hetero_jax_gpu/pm.py)
  export PM_SNAP_DIR=$ROOT/snap PM_SNAP_MODE=${1:-restore} PM_KV_BYTES=${PM_KV_BYTES:-}
  export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/vllm/xla_cache   # tpu-inference's VLLM_XLA_CACHE_PATH, from process start
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1  # persist every program
  [[ -n $PM_KV_BYTES ]] || unset PM_KV_BYTES
  # engine core forked from a forkserver that preloaded its modules next to the API server's imports (pm.py zygote;
  # dev pod 2026-10-24: spawn 33.0 -> 22.8 s, same JAX cache keys as a spawned engine)
  export PM_ZYGOTE=${PM_ZYGOTE:-1}
}

nvidia_holders() {  # pids with a /dev/nvidia0 handle (the image has no ps/fuser)
  for fd in /proc/[0-9]*/fd/*; do [[ $(readlink "$fd" 2>/dev/null) == /dev/nvidia0 ]] && echo "${fd#/proc/}" || true; done \
    | cut -d/ -f1 | sort -u
}

case "$ARM" in
  stock-lazy|stock-runai|stock-runai-cc|stock-ckpt|stock-ckpt-cc|pm-seed)
    mark download_start
    case "$ARM" in
      stock-ckpt*|pm-seed) python3 /scripts/fetch.py ckpt "gs://$BUCKET/$PREFIX" /model --workers "${FETCH_WORKERS:-32}" ;;
    esac
    case "$ARM" in
      *-cc) python3 /scripts/fetch.py tar "$CC_URI" "$HOME" ;;
    esac
    mark download_end
    case "$ARM" in
      # tpu-inference's flax_nnx Gemma-4 (LoadableWithIterator + StandardWeightLoader) takes vLLM's streamer
      stock-runai*) exec python3 -m hetero_jax_gpu.serve "gs://$BUCKET/$PREFIX" -- --load-format runai_streamer ;;
      pm-seed) pm_env save; exec python3 -m hetero_jax_gpu.serve "${keep[@]}" /model ;;
      *) exec python3 -m hetero_jax_gpu.serve /model ;;
    esac ;;

  warm-stock)  # the fresh stock-ckpt job on this pod has exited (engine stopped); nothing is delivered
    mark download_start
    mark download_end
    exec python3 -m hetero_jax_gpu.serve /model ;;

  pm-gcs|pm-peer|loo-snapshot|loo-cache0|loo-zygote|loo-overlap|loo-read-threads|loo-snapshot-cache0)
    rm -rf "$ROOT/model" "$ROOT/snap" "$ROOT/.meta_done" "$ROOT/.cc_ready"
    mkdir -p "$ROOT"
    fetch_args=(); model=("$ROOT/model"); fetched=0
    case "$ARM" in
      loo-snapshot) fetch_args=(--no-snap); model=("gs://$BUCKET/$PREFIX" -- --load-format runai_streamer) ;;
      loo-cache0) fetch_args=(--cc-uri "$CC_URI") ;;
      loo-snapshot-cache0) fetch_args=(--no-snap --cc-uri "$CC_URI")
                           model=("gs://$BUCKET/$PREFIX" -- --load-format runai_streamer) ;;
    esac
    mark download_start
    if [[ $ARM == pm-peer ]]; then
      python3 /scripts/pm_state.py fetch-peer "$PM_PEER" --root "$ROOT" --home "$HOME" --gate "$ROOT/.cc_ready" &
    else
      python3 /scripts/pm_state.py fetch-gcs "$PM_STATE_URI" --root "$ROOT" --home "$HOME" --gate "$ROOT/.cc_ready" \
        "${fetch_args[@]}" &
    fi
    fetch=$!
    if [[ $ARM == loo-overlap ]]; then  # no overlap: the engine starts once the whole state has landed
      wait $fetch || { echo "delivery failed"; exit 1; }
      fetched=1
    fi
    until [[ -e $ROOT/.meta_done ]]; do kill -0 $fetch 2>/dev/null || { wait $fetch; exit 1; }; sleep 0.02; done
    pm_env restore
    case "$ARM" in
      loo-snapshot) unset PM_SNAP_MODE PM_SNAP_DIR ;;
      loo-cache0) unset JAX_COMPILATION_CACHE_DIR JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS \
                        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES ;;
      loo-snapshot-cache0) unset PM_SNAP_MODE PM_SNAP_DIR JAX_COMPILATION_CACHE_DIR \
                                 JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES ;;
      loo-zygote) export PM_ZYGOTE=0 ;;
      loo-read-threads) export PM_SNAP_SOURCE=mmap PM_SNAP_THREADS=4 ;;
    esac
    export PM_GATE_FILE=$ROOT/.cc_ready
    echo "PM_CONFIG arm=$ARM zygote=${PM_ZYGOTE:-unset} snap_mode=${PM_SNAP_MODE:-unset}" \
         "snap_source=${PM_SNAP_SOURCE:-read} snap_threads=${PM_SNAP_THREADS:-32}" \
         "cache_dir=${JAX_COMPILATION_CACHE_DIR:-unset}" \
         "cache_min_compile_s=${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-unset} kv_bytes=${PM_KV_BYTES:-unset}" \
         "fetched_before_launch=$fetched model=${model[*]}"
    python3 -m hetero_jax_gpu.serve --pidfile "$ROOT/engine.pid" "${keep[@]}" "${model[@]}" &
    engine=$!
    if [[ $fetched == 0 ]] && ! wait $fetch; then
      echo "delivery failed"; kill -TERM $engine; wait $engine || true; exit 1
    fi
    wait $engine ;;

  warm-pm)  # pre-spawn next to the live engine of this pod's pm-gcs job; T0 = gate open once the old engine exited
    old=$(cat "$ROOT/engine.pid")
    holders=$(nvidia_holders | tr '\n' ' ')  # the old engine's processes, before the new one exists
    gate=$ROOT/.gate-$$
    rm -f "$gate" "$gate.parked"
    pm_env restore
    export PM_GATE_FILE=$gate
    # its own port: the old engine still answers on 8000 until it is retired below
    python3 -m hetero_jax_gpu.serve --port 8001 --pidfile "$ROOT/engine.pid.new" "${keep[@]}" "$ROOT/model" &
    engine=$!
    until [[ -e $gate.parked ]]; do kill -0 $engine 2>/dev/null || { wait $engine; exit 1; }; sleep 0.02; done
    mark prespawn_parked
    echo "old engine: serve pid $old; /dev/nvidia0 holders: $holders"
    kill -TERM "$old"
    while kill -0 "$old" 2>/dev/null; do sleep 0.02; done
    for p in $holders; do while [[ -d /proc/$p ]]; do sleep 0.02; done; done
    mark old_engine_exited
    mv "$ROOT/engine.pid.new" "$ROOT/engine.pid"
    mark download_start
    mark download_end
    touch "$gate"
    mark gate_open
    wait $engine ;;

  *) echo "unknown ARM $ARM"; exit 2 ;;
esac
