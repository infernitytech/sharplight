#!/bin/bash
# Section "§OFF" of docs/benchmarks.md: CPU expert offload against llama.cpp with the same statistic (same_stat_offload.sh), Sharplight with
# llama-bench's prompt distribution (SHARPLIGHT_BENCH_PROMPT=vocab). The "before" rows (whole-bank copies) used
# SHARPLIGHT_STREAM_SELECTED_EXPERTS=0, a switch removed in 0.3; their jobs are gone.
# The caller takes the GPU lock for each job; env: LLAMA_ROOT, MODELS (GGUF directory), OSS (a local-disk copy of gpt-oss-120b-MXFP4.gguf).
# usage: offload_selected_experts.sh <out dir> <tag> <sl-first|llama-first> <job...>
#   jobs: oss oss-pin oss-vk cn cn-vk q122
OUT=$1; R=$2; ORD=$3; shift 3
HERE="$(cd "$(dirname "$0")" && pwd)"
export SHARPLIGHT_BENCH_PROMPT=vocab
CN=$MODELS/Qwen3-Coder-Next/Qwen3-Coder-Next-Q4_K_M.gguf
Q=$MODELS/Qwen3.5-122B-A10B/Qwen3.5-122B-A10B-UD-IQ2_XXS.gguf
for job in "$@"; do
  case $job in
    oss) LLAMA_EXTRA="-lm none" "$HERE/same_stat_offload.sh" "$OUT" gpt-oss-120b "$OSS" cuda 22 "$R" "$ORD" ;;
    oss-pin) SHARPLIGHT_LIMIT_PINNED_HOST_MEMORY=40G "$HERE/same_stat_offload.sh" "$OUT" gpt-oss-120b-pin40 "$OSS" cuda 22 "$R" sl ;;
    oss-vk) LLAMA_EXTRA="-lm none" "$HERE/same_stat_offload.sh" "$OUT" gpt-oss-120b "$OSS" vulkan 22 "$R" "$ORD" ;;
    cn) "$HERE/same_stat_offload.sh" "$OUT" coder-next "$CN" cuda 16 "$R" "$ORD" ;;
    cn-vk) "$HERE/same_stat_offload.sh" "$OUT" coder-next "$CN" vulkan 16 "$R" "$ORD" ;;
    q122) "$HERE/same_stat_offload.sh" "$OUT" qwen35-122b "$Q" cuda 6 "$R" "$ORD" ;;
  esac
done
