#!/bin/bash
# NVRTC machine code for quantize_fp8 / decode_combine: default vs SASS_EXTRA, decode and mixed steps, alternated
. <out>/env.sh
for M in $M8 $MOSS; do N=$(basename $M)
A="--slots 64 --ctx 640 --depth 256 --batch 2048 --modes greedy --reps 10 --mixed 16+0,64+0,0+250x3,15+250x1,48+250x4"
for r in 1 2; do for f in def sx; do
  unset SHARPLIGHT_CUDA_NVRTC_SASS SHARPLIGHT_CUDA_DUMP_SOURCE
  [ $f = sx ] && export SHARPLIGHT_CUDA_NVRTC_SASS="if:$SASS_EXTRA"
  [ $f = sx ] && [ $r = 1 ] && export SHARPLIGHT_CUDA_DUMP_SOURCE=$WORK/dump_$N
  export XDG_CACHE_HOME=$WORK/cache_steps_$f
  echo "== $N $f round $r"
  $TREE/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider cuda $A 2>&1 | grep -E "^\||rror|xception"
done; done; done
