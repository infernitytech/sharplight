#!/bin/bash
# Per-model gap table of docs/benchmarks.md section 42: prefill and decode of Sharplight against vLLM (same Hugging Face
# checkpoint) or llama.cpp (same GGUF) with one procedure on both sides (section 40.9): one process run per engine and model
# (Sharplight: one per prompt length), per length one warm-up and 5 timed prefills, reported as the median of the 5.
#   Sharplight: `--bench N --ctx N+256` (prefill median of 5; decode: median of 64 steps after the N-token prompt).
#   vLLM:       bench/vllm_single.py --points pp,tg --reps 5 (pp N; tg at depth 16 and DEPTH, median of 5 128-token runs).
#   llama.cpp:  llama-bench -p LENS -n 128 -r 5 (median of samples_ns; tg128 after an empty context).
# usage: s42_gap.sh <out dir> <name> <sharplight|vllm|llama> <model path> [engine args...]
# env: LENS (prompt lengths, default "512 2048 8192"), VLLM_VENV, LLAMA_BENCH, SL_ARGS (extra Sharplight CLI arguments),
#      VLLM_ARGS (extra vllm_single.py arguments, e.g. "--gpu-util 0.92 --max-len 8704"), LLAMA_ARGS (e.g. "-ngl 99 -fa 1").
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT=${1:?out dir}; NAME=${2:?name}; ENG=${3:?engine}; MODEL=${4:?model}; mkdir -p "$OUT"
LENS=${LENS:-"512 2048 8192"}
CLI=$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight
MAXN=0; for N in $LENS; do [ "$N" -gt "$MAXN" ] && MAXN=$N; done
case $ENG in
  sharplight)
    : > "$OUT/sl_$NAME.txt"
    for N in $LENS; do
      # shellcheck disable=SC2086
      "$CLI" "$MODEL" --bench "$N" --ctx $((N + 256)) $SL_ARGS 2>&1 | grep -E '^prefill|rror' | sed "s/^/[N=$N] /" >> "$OUT/sl_$NAME.txt"
    done ;;
  vllm)
    # shellcheck disable=SC2086
    "$VLLM_VENV/bin/python" -I "$HERE/vllm_single.py" "$MODEL" "$NAME" --points pp,tg --reps 5 --depth "$MAXN" --max-len $((MAXN + 512)) \
      ${VLLM_ARGS:---gpu-util 0.85} --out "$OUT/vllm_$NAME.json" > "$OUT/vllm_$NAME.log" 2>&1 ;;
  llama)
    # shellcheck disable=SC2086
    LD_LIBRARY_PATH=$(dirname "$LLAMA_BENCH") "$LLAMA_BENCH" -m "$MODEL" ${LLAMA_ARGS:--ngl 99 -fa 1} -p "$(echo $LENS | tr ' ' ,)" -n 128 -r 5 \
      -o json > "$OUT/llama_$NAME.json" 2> "$OUT/llama_$NAME.err" ;;
esac
