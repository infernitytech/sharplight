#!/bin/bash
# Serving runs of docs/benchmarks.md section 16.
# usage: s16_serve.sh <sharplight|llama-cuda13|llama-vulkan|vllm> <name> <model> "<concurrency list>" [extra server args...]
# Starts ONE server sized for the largest concurrency (sharplight --slots, llama-server -np, vLLM --max-num-seqs), then runs
# bench/loadgen.py (closed loop, ~250-token random-word prompts or --natural instructions, 128 output tokens, ignore_eos, temperature
# 0, server /metrics deltas) ROUNDS times per concurrency, then stops the server.
# Results: $OUT_DIR/serve_<engine>_<name>_c<N>_r<round>.json (+ .gpu.txt), server log $OUT_DIR/serve_<engine>_<name>.server.log
# env: OUT_DIR (bench/results/s16), PORT (18080), READY_TIMEOUT (1500 s), CTX_PER_SLOT (512), SLOTS (max concurrency), ROUNDS (2), NATURAL=1 (natural
#      prompts), SHARPLIGHT_SERVER (default: the Release build in this checkout), LLAMA_DIR / LLAMA_BUILD (b11177), VLLM_VENV
#      (virtualenv with vLLM and httpx; its python also runs loadgen.py; LOADGEN_PY overrides), QUIET_GPU=1 (repeat a round, up to
#      3 tries, when another process computed on the GPU during it; then it is kept and marked <round>.contaminated),
#      PER_OVERRIDE (requests per worker, default by concurrency), RAMP (seconds: closed loop with staggered worker starts),
#      RATES ("16:7.5 64:18": open loop, Poisson arrivals of that many requests per second at that concurrency; see loadgen.py),
#      TEXT_CHECK (1: after the rounds, bench/textcheck.py compares batched greedy texts of natural prompts with single-request ones
#      and the script exits 3 when it finds garbage; 0 skips it), TEXT_CHECK_WAVES (concurrent requests per wave, default "16,<SLOTS>").
# Text check result: $OUT_DIR/serve_<engine>_<name>.textcheck.json (+ .textcheck.FAIL when garbage was found).
HERE="$(cd "$(dirname "$0")" && pwd)"
ENG=$1; NAME=$2; MP=$3; CONCS=$4; shift 4; EXTRA=("$@")
OUT_DIR=${OUT_DIR:-$HERE/results/s16}; mkdir -p "$OUT_DIR"
PORT=${PORT:-18080}; CTX=${CTX_PER_SLOT:-512}; ROUNDS=${ROUNDS:-2}
SLOTS=${SLOTS:-$(echo $CONCS | tr ' ' '\n' | sort -n | tail -1)}
PY=${LOADGEN_PY:-${VLLM_VENV:?set VLLM_VENV}/bin/python}
LOG=$OUT_DIR/serve_${ENG}_${NAME}.server.log
export VLLM_USE_FLASHINFER_SAMPLER=0
# Fail fast when the port is taken or another server already runs (a leftover server would be measured instead).
if curl -s -o /dev/null "localhost:$PORT" 2>/dev/null || (command -v ss >/dev/null && ss -ltn "sport = :$PORT" | grep -q LISTEN); then
  echo "port $PORT is in use; stop that server first" >&2; exit 2
fi
if pgrep -f "sharplight serve|vllm serve|llama-server" >/dev/null; then
  echo "another server is running: $(pgrep -af 'sharplight serve|vllm serve|llama-server' | cut -c1-120 | head -3)" >&2; exit 2
fi
set -m
case $ENG in
  sharplight)
    SRV=${SHARPLIGHT_SERVER:-$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight}
    SHARPLIGHT_LOAD_TIMING=1 "$SRV" serve "$MP" --slots "$SLOTS" --ctx "$CTX" --listen "http://127.0.0.1:$PORT" --temp 0 "${EXTRA[@]}" > "$LOG" 2>&1 & ;;
  llama-*)
    B=${ENG#llama-}; L=${LLAMA_DIR:?set LLAMA_DIR}/runtime.llama-${LLAMA_BUILD:-b11177}-$B
    LD_LIBRARY_PATH=$L "$L/llama-server" -m "$MP" -ngl 99 -fa on -np "$SLOTS" -c $((SLOTS * CTX)) --port "$PORT" --no-webui --metrics "${EXTRA[@]}" > "$LOG" 2>&1 & ;;
  vllm)
    "$VLLM_VENV/bin/vllm" serve "$MP" --served-model-name m --max-model-len "$CTX" --max-num-seqs "$SLOTS" --port "$PORT" \
      --no-enable-prefix-caching "${EXTRA[@]}" > "$LOG" 2>&1 & ;;
  *) echo "unknown engine $ENG"; exit 1 ;;
esac
SP=$!; T0=$SECONDS; OK=0
MON=
# Stop the server's process group (and the round monitor) however this script ends.
cleanup() {
  [ -n "$MON" ] && kill "$MON" 2>/dev/null
  kill -- -$SP 2>/dev/null; sleep 3; kill -9 -- -$SP 2>/dev/null; wait $SP 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT TERM
# True when a process outside the server's process group computes on the GPU.
foreign_gpu() {
  local p
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    [ "$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')" = "$SP" ] || return 0
  done
  return 1
}
while kill -0 $SP 2>/dev/null; do
  curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { OK=1; break; }
  [ $((SECONDS - T0)) -gt "${READY_TIMEOUT:-1500}" ] && break; sleep 1
done
echo "ready=$OK after $((SECONDS - T0)) s" | tee "$LOG.ready"
NAT=""; [ "${NATURAL:-0}" = 1 ] && NAT="--natural"
if [ $OK = 1 ]; then
  for N in $CONCS; do
    case $N in 1) PER=8;; 4) PER=6;; 16) PER=4;; *) PER=3;; esac
    PER=${PER_OVERRIDE:-$PER}
    ARR=()
    [ -n "${RAMP:-}" ] && ARR+=(--ramp "$RAMP")
    for RN in ${RATES:-}; do [ "${RN%%:*}" = "$N" ] && ARR+=(--rate "${RN#*:}"); done
    for R in $(seq "$ROUNDS"); do
      OUT=$OUT_DIR/serve_${ENG}_${NAME}_c${N}_r$R
      for TRY in 1 2 3; do
        # QUIET_GPU=1: start only while no other process computes on the GPU, sample during the round, repeat it if one appeared.
        if [ "${QUIET_GPU:-0}" = 1 ]; then
          for W in $(seq 150); do foreign_gpu || break; sleep 2; done
          rm -f "$OUT.foreign"; ( while :; do foreign_gpu && touch "$OUT.foreign"; sleep 1; done ) & MON=$!
        fi
        "$HERE/gpu_check.sh" > "$OUT.gpu.txt"
        timeout 1800 "$PY" -I "$HERE/loadgen.py" --url "http://127.0.0.1:$PORT" --model m -c "$N" --per-worker $PER --ignore-eos \
          --metrics --seed "$R" $NAT "${ARR[@]}" --json "$OUT.json" | cut -c1-400
        [ "${QUIET_GPU:-0}" = 1 ] || break
        kill $MON 2>/dev/null; wait $MON 2>/dev/null; MON=
        [ -e "$OUT.foreign" ] || break
        if [ "$TRY" = 3 ]; then
          # Three tries with another process on the GPU: keep the last one, marked as contaminated.
          echo "round $R: another GPU process in every try, kept as contaminated" | tee "$OUT.contaminated"
        else
          echo "round $R: another GPU process appeared, repeating (try $TRY)"
        fi
      done
      rm -f "$OUT.foreign"
    done
  done
fi
TC=0
if [ $OK = 1 ] && [ "${TEXT_CHECK:-1}" = 1 ]; then
  # Batched greedy text against single-request text (docs/benchmarks.md section 102): speed rounds never read their text.
  TCOUT=$OUT_DIR/serve_${ENG}_${NAME}.textcheck
  WAVES=${TEXT_CHECK_WAVES:-$(printf '%s\n' 16 "$SLOTS" | sort -nu | paste -sd,)}
  rm -f "$TCOUT.FAIL"
  timeout 1800 "$PY" -I "$HERE/textcheck.py" --url "http://127.0.0.1:$PORT" --model m --waves "$WAVES" --json "$TCOUT.json"
  TC=$?
  [ $TC = 0 ] || { echo "TEXT CHECK FAILED (rc=$TC): batched greedy text differs from single-request text, see $TCOUT.json" | tee "$TCOUT.FAIL" >&2; }
fi
# The server runs in its own process group (set -m): the EXIT trap stops the group, so worker processes (vLLM's engine core) go with it.
cleanup; trap - EXIT; sleep 5
[ $TC = 0 ] || exit 3
