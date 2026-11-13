#!/bin/bash
# usage: run_serve_all.sh <model-key> [engines...]   (model keys as in models.sh; vLLM path via VLLM_MODEL)
HERE="$(cd "$(dirname "$0")" && pwd)"; . "$HERE/models.sh"
KEY=$1; shift; ENGS=${@:-sharplight llama-cuda13 llama-vulkan vllm}
for e in "${MODELS[@]}"; do [ "${e%%|*}" = $KEY ] && P=${e##*|}; done
[ $KEY = qwen27b-q8 ] && SLP=$HOME/models/sharplight/Qwen3.8-27B-Q8_0.slm || SLP=$P
for eng in $ENGS; do
  echo "=== $KEY $eng"
  case $eng in
    sharplight) "$HERE/serve_bench.sh" sharplight $KEY $SLP "1 4 16 64" ;;
    vllm) "$HERE/serve_bench.sh" vllm $KEY "$VLLM_MODEL" "1 4 16 64" --gpu-memory-utilization ${VLLM_UTIL:-0.3} ;;
    *) "$HERE/serve_bench.sh" $eng $KEY $P "1 4 16 64" ;;
  esac
done
