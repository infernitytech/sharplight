#!/bin/bash
# Long-context comparison against vLLM (docs/benchmarks.md section 58.8): prefill at LENS tokens (also the single-prompt TTFT), decode
# of one stream at each length (`--bench`), decode of STREAMS streams at DEPTHS (BatchBench), CUDA profiles of the prefill at
# PROFILE_LENS, then the same points on vLLM. Every run is bounded (RUN_TIMEOUT) and waits for a GPU without other compute processes
# (up to QUIET_MAX seconds); the vLLM run holds LOCK.
# usage: s58_long.sh <out dir> <name> <model dir> [sharplight|vllm|both]
# env: LENS (default "16384 32768 65536 131072"), DEPTHS ("32768 131072"), STREAMS ("4"), PROFILE_LENS ("32768 131072"),
#      VLLM_VENV, VLLM_ARGS (e.g. "--gpu-util 0.92 --kv fp8"), SL_ARGS, RUN_TIMEOUT (1800 s), QUIET_MAX (900 s), LOCK.
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT=${1:?out dir}; NAME=${2:?name}; MODEL=${3:?model}; WHICH=${4:-both}; mkdir -p "$OUT"
LENS=${LENS:-"16384 32768 65536 131072"}; DEPTHS=${DEPTHS:-"32768 131072"}; STREAMS=${STREAMS:-"4"}; PROFILE_LENS=${PROFILE_LENS:-"32768 131072"}
T=${RUN_TIMEOUT:-1800}; LOCK=${LOCK:-/root/.sl-verify.lock}
CLI=$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight
BB=$HERE/BatchBench/bin/Release/net11.0/BatchBench
quiet() { local t=0; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 10; t=$((t + 10)); [ $t -ge "${QUIET_MAX:-900}" ] && { echo "(GPU busy)"; return 1; }; done; return 0; }
if [ "$WHICH" != vllm ]; then
  S=$OUT/sl_$NAME.txt; : > "$S"
  for N in $LENS; do
    q=$(quiet)
    # shellcheck disable=SC2086
    r=$(timeout "$T" "$CLI" "$MODEL" --provider cuda --bench "$N" --ctx $((N + 256)) $SL_ARGS 2>&1 | grep -E '^prefill|rror|xception' | head -3)
    echo "[N=$N] ${r:-failed or timed out (exit ${PIPESTATUS[0]})} $q" >> "$S"
  done
  for D in $DEPTHS; do
    q=$(quiet)
    # shellcheck disable=SC2086
    r=$(timeout "$T" "$BB" "$MODEL" --provider cuda --slots "$(echo $STREAMS | tr ' ' '\n' | sort -n | tail -1)" --ctx $((D + 512)) --depth "$D" --batch 2048 \
      --sizes "1,$(echo $STREAMS | tr ' ' ',')" --modes greedy --reps 5 2>&1 | grep -E '^\| [0-9]|rror|xception' | head -4)
    echo "[decode depth $D] ${r:-failed or timed out} $q" | tr '\n' ' ' >> "$S"; echo >> "$S"
  done
  for N in $PROFILE_LENS; do
    q=$(quiet)
    SHARPLIGHT_CUDA_PROFILE=1 timeout "$T" "$CLI" "$MODEL" --provider cuda --bench "$N" --ctx $((N + 256)) $SL_ARGS > "$OUT/profile_${NAME}_$N.txt" 2>&1
    python3 "$HERE/profile_categories.py" "$OUT/profile_${NAME}_$N.txt" prefill > "$OUT/profile_${NAME}_$N.categories.txt" 2>&1
  done
fi
if [ "$WHICH" != sharplight ]; then
  quiet > /dev/null
  MAXN=0; for N in $LENS $DEPTHS; do [ "$N" -gt "$MAXN" ] && MAXN=$N; done
  # shellcheck disable=SC2086
  flock -w 1800 "$LOCK" timeout $((T * 2)) "$VLLM_VENV/bin/python" -I "$HERE/s58_long_vllm.py" "$MODEL" "$NAME" --lens "$(echo $LENS | tr ' ' ,)" \
    --depths "$(echo $DEPTHS | tr ' ' ,)" --streams "1,$(echo $STREAMS | tr ' ' ',')" --max-len $((MAXN + 512)) ${VLLM_ARGS:---gpu-util 0.92} \
    --out "$OUT/vllm_$NAME.json" > "$OUT/vllm_$NAME.log" 2>&1
  echo "vllm exit $?" >> "$OUT/vllm_$NAME.log"
fi
