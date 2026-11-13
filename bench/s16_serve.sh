#!/bin/bash
# Serving runs of docs/benchmarks.md section 16.
# usage: s16_serve.sh <sharplight|llama-cuda13|llama-vulkan|vllm> <name> <model> "<concurrency list>" [extra server args...]
# Starts ONE server sized for the largest concurrency (sharplight --slots, llama-server -np, vLLM --max-num-seqs), then runs
# bench/loadgen.py (closed loop, ~250-token random-word prompts or --natural instructions, 128 output tokens, ignore_eos, temperature
# 0, server /metrics deltas) ROUNDS times per concurrency, then stops the server.
# Results: $OUT_DIR/serve_<engine>_<name>_c<N>_r<round>.json (+ .gpu.txt), server log $OUT_DIR/serve_<engine>_<name>.server.log
# env: OUT_DIR (bench/results/s16), PORT (18080), CTX_PER_SLOT (512), SLOTS (max concurrency), ROUNDS (2), NATURAL=1 (natural
#      prompts), SHARPLIGHT_SERVER (default: the Release build in this checkout), LLAMA_DIR / LLAMA_BUILD (b11177), VLLM_VENV
#      (virtualenv with vLLM and httpx; its python also runs loadgen.py; LOADGEN_PY overrides).
HERE="$(cd "$(dirname "$0")" && pwd)"
ENG=$1; NAME=$2; MP=$3; CONCS=$4; shift 4; EXTRA=("$@")
OUT_DIR=${OUT_DIR:-$HERE/results/s16}; mkdir -p "$OUT_DIR"
PORT=${PORT:-18080}; CTX=${CTX_PER_SLOT:-512}; ROUNDS=${ROUNDS:-2}
SLOTS=${SLOTS:-$(echo $CONCS | tr ' ' '\n' | sort -n | tail -1)}
PY=${LOADGEN_PY:-${VLLM_VENV:?set VLLM_VENV}/bin/python}
LOG=$OUT_DIR/serve_${ENG}_${NAME}.server.log
export VLLM_USE_FLASHINFER_SAMPLER=0
case $ENG in
  sharplight)
    SRV=${SHARPLIGHT_SERVER:-$HERE/../samples/Sharplight.Server/bin/Release/net11.0/sharplight-server}
    SHARPLIGHT_LOAD_TIMING=1 "$SRV" "$MP" --slots "$SLOTS" --ctx "$CTX" --listen "http://127.0.0.1:$PORT" --temp 0 "${EXTRA[@]}" > "$LOG" 2>&1 & ;;
  llama-*)
    B=${ENG#llama-}; L=${LLAMA_DIR:?set LLAMA_DIR}/runtime.llama-${LLAMA_BUILD:-b11177}-$B
    LD_LIBRARY_PATH=$L "$L/llama-server" -m "$MP" -ngl 99 -fa on -np "$SLOTS" -c $((SLOTS * CTX)) --port "$PORT" --no-webui --metrics "${EXTRA[@]}" > "$LOG" 2>&1 & ;;
  vllm)
    "$VLLM_VENV/bin/vllm" serve "$MP" --served-model-name m --max-model-len "$CTX" --max-num-seqs "$SLOTS" --port "$PORT" \
      --no-enable-prefix-caching "${EXTRA[@]}" > "$LOG" 2>&1 & ;;
  *) echo "unknown engine $ENG"; exit 1 ;;
esac
SP=$!; T0=$(date +%s.%N); OK=0
while kill -0 $SP 2>/dev/null; do
  curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { OK=1; break; }
  [ "$(echo "$(date +%s.%N)-$T0>1500" | bc)" = 1 ] && break; sleep 1
done
echo "ready=$OK after $(echo "$(date +%s.%N)-$T0" | bc) s" | tee "$LOG.ready"
NAT=""; [ "${NATURAL:-0}" = 1 ] && NAT="--natural"
if [ $OK = 1 ]; then
  for N in $CONCS; do
    case $N in 1) PER=8;; 4) PER=6;; 16) PER=4;; *) PER=3;; esac
    for R in $(seq "$ROUNDS"); do
      OUT=$OUT_DIR/serve_${ENG}_${NAME}_c${N}_r$R; "$HERE/gpu_check.sh" > "$OUT.gpu.txt"
      timeout 1800 "$PY" -I "$HERE/loadgen.py" --url "http://127.0.0.1:$PORT" --model m -c "$N" --per-worker $PER --ignore-eos \
        --metrics --seed "$R" $NAT --json "$OUT.json" | cut -c1-400
    done
  done
fi
kill $SP 2>/dev/null; sleep 3; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null; sleep 5
