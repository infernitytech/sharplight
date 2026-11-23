#!/bin/bash
. $ENV_SH
M=${M:-$HOME/models/hf/Qwen3-8B-FP8}; N=$(basename $M)
for r in 1 2; do for f in base sasswg0 sasswg1 drvwg0; do
  unset SHARPLIGHT_CUDA_NVRTC_SASS SHARPLIGHT_FP8_WG
  case $f in base) W=$BASE;; sasswg0) W=$TREE; export SHARPLIGHT_FP8_WG=0;; sasswg1) W=$TREE;; drvwg0) W=$TREE; export SHARPLIGHT_FP8_WG=0 SHARPLIGHT_CUDA_NVRTC_SASS=0;; esac
  export XDG_CACHE_HOME=$WORK/cache_wg_$f
  echo "== $N $f round $r"
  $W/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider cuda --slots 64 --ctx 640 --depth 256 --batch 2048 --modes greedy --reps 5 \
    --mixed "${MIXED:-16+0,64+0,0+250x3,0+250x8,15+250x1,13+250x3,48+250x4}" 2>&1 | grep -E "^\||rror|xception"
done; done
