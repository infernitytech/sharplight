#!/bin/bash
# BatchBench steps, develop vs perf/llm-r34 (NVRTC SASS for gemm_fp8_tma) vs r34 with SHARPLIGHT_CUDA_NVRTC_SASS=0, alternated.
. $ENV_SH
M=${M:-$HOME/models/hf/Qwen3-8B-FP8}; N=$(basename $M)
for r in 1 2; do for f in ${FORMS:-base new off}; do
  unset SHARPLIGHT_CUDA_NVRTC_SASS
  case $f in base) W=$BASE;; new) W=$TREE;; off) W=$TREE; export SHARPLIGHT_CUDA_NVRTC_SASS=0;; esac
  export XDG_CACHE_HOME=$WORK/cache_ab_$f
  echo "== $N $f round $r"
  $W/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider cuda --slots 64 --ctx 640 --depth 256 --batch 2048 --modes greedy --reps 5 \
    --mixed "${MIXED:-16+0,64+0,0+250x3,0+250x8,15+250x1,13+250x3,48+250x4}" 2>&1 | grep -E "^\||rror|xception"
done; done
