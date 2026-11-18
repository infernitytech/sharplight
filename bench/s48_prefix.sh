#!/bin/bash
# Prefix caching against vLLM (docs/benchmarks.md section 48): one server per engine and cache setting (64 sequences, 8192-token
# contexts, the same KV capacity in tokens), then bench/prefix_bench.py: 1500-token shared system prompt at 16 and 64 concurrent chats,
# 16 multi-turn conversations of 6 turns resending their history, and a RAG load over 4 shared 1000-token documents.
# usage: s48_prefix.sh <out dir> <model dir> [engine...]   engines: sharplight sharplight-nocache vllm vllm-nocache (default: all)
# env: VLLM_VENV, SHARPLIGHT_SERVER (default: this checkout's Release build), PORT (18090), KV_TOKENS (KV capacity of both engines in
#      tokens, 73728), KV_BYTES_PER_TOKEN (147456: Qwen3-8B, 36 layers x 8 KV heads x 128 x K/V x f16; vLLM --kv-cache-memory-bytes =
#      KV_TOKENS x this), VLLM_KV_BYTES (overrides that product, e.g. for hybrid models whose Mamba state shares the budget), ROUNDS (2)
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${1:?out dir}; MP=${2:?model dir}; shift 2
ENGINES=${*:-"sharplight sharplight-nocache vllm vllm-nocache"}
PORT=${PORT:-18090}; ROUNDS=${ROUNDS:-2}; PY=${VLLM_VENV:?set VLLM_VENV}/bin/python
mkdir -p "$OUT"
set -m
for E in $ENGINES; do
  LOG=$OUT/$E.server.log
  case $E in
    sharplight|sharplight-nocache)
      SRV=${SHARPLIGHT_SERVER:-$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight}
      EXTRA=(); [ "$E" = sharplight-nocache ] && EXTRA=(--no-prefix-cache)
      "$SRV" serve "$MP" --slots 64 --ctx 8192 --kv-tokens "${KV_TOKENS:-73728}" --listen "http://127.0.0.1:$PORT" --temp 0 "${EXTRA[@]}" > "$LOG" 2>&1 & ;;
    vllm|vllm-nocache)
      EXTRA=(--enable-prefix-caching); [ "$E" = vllm-nocache ] && EXTRA=(--no-enable-prefix-caching)
      VLLM_USE_FLASHINFER_SAMPLER=0 "$VLLM_VENV/bin/vllm" serve "$MP" --served-model-name m --max-model-len 8192 --max-num-seqs 64 \
        --kv-cache-memory-bytes ${VLLM_KV_BYTES:-$(( ${KV_TOKENS:-73728} * ${KV_BYTES_PER_TOKEN:-147456} ))} --enable-prompt-tokens-details --port "$PORT" "${EXTRA[@]}" > "$LOG" 2>&1 & ;;
    *) echo "unknown engine $E"; exit 1 ;;
  esac
  SP=$!; T0=$SECONDS; OK=0
  while kill -0 $SP 2>/dev/null; do
    curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { OK=1; break; }
    [ $((SECONDS - T0)) -gt 1500 ] && break; sleep 1
  done
  echo "$E ready=$OK after $((SECONDS - T0)) s"
  if [ $OK = 1 ]; then
    for R in $(seq "$ROUNDS"); do
      for spec in "shared 16 4" "shared 64 3" "multiturn 16 0" "rag 16 4"; do
        set -- $spec
        "$PY" -I "$HERE/prefix_bench.py" --url "http://127.0.0.1:$PORT" --workload "$1" -c "$2" --per-worker "$3" --seed "$R" \
          --json "$OUT/${E}_$1_c$2_r$R.json" | cut -c1-300
      done
    done
  fi
  kill -- -$SP 2>/dev/null; sleep 3; kill -9 -- -$SP 2>/dev/null; wait $SP 2>/dev/null; sleep 5
done
