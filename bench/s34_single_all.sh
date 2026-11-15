#!/bin/bash
# Single-stream baseline of docs/benchmarks.md section 34 (one GPU of a multi-GPU machine).
# usage: s34_single_all.sh <sharplight|llama-cuda13|llama-vulkan> [model names...]   (default: every model of models.sh + S34_EXTRA)
# Runs bench/s16_single.sh per model with the GPU selected by GPU_INDEX (default 1) for every engine:
#   Sharplight SHARPLIGHT_VK_DEVICES, llama.cpp CUDA CUDA_VISIBLE_DEVICES, llama.cpp Vulkan GGML_VK_VISIBLE_DEVICES.
# env: OUT_DIR (bench/results/s34/single), GPU_INDEX (1), LLAMA_DIR (default $LLAMA_ROOT), plus those of s16_single.sh.
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/models.sh"
ENG=$1; shift
S34_EXTRA=(
  "qwen3-8b-q4km|$M/gguf/Qwen3-8B/Qwen3-8B-Q4_K_M.gguf"
  "gpt-oss-20b-mxfp4|$M/gguf/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf"
)
export OUT_DIR=${OUT_DIR:-$HERE/results/s34/single}
export LLAMA_DIR=${LLAMA_DIR:-$LLAMA_ROOT}
G=${GPU_INDEX:-1}
export SHARPLIGHT_VK_DEVICES=$G CUDA_VISIBLE_DEVICES=$G GGML_VK_VISIBLE_DEVICES=$G
for e in "${MODELS[@]}" "${S34_EXTRA[@]}"; do
  n=${e%%|*}; p=${e##*|}
  if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$n"; then continue; fi
  echo "== $ENG $n $(date +%T)"
  "$HERE/s16_single.sh" "$ENG" "$n" "$p"
done
