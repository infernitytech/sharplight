#!/bin/bash
# decode steps: default vs NVRTC machine code also for quantize_fp8 / decode_combine, 3 rounds alternated
. <out>/env.sh
for M in $M8 $MOSS; do N=$(basename $M)
A="--slots 64 --ctx 640 --depth 256 --modes greedy --reps 20 --sizes 16,32,48,64"
for r in 1 2 3; do for f in def sx; do
  unset SHARPLIGHT_CUDA_NVRTC_SASS
  [ $f = sx ] && export SHARPLIGHT_CUDA_NVRTC_SASS="if:$SASS_EXTRA"
  export XDG_CACHE_HOME=$WORK/cache_steps_$f
  echo "== $N $f round $r"
  $TREE/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider cuda $A 2>&1 | grep -E "^\||rror|xception"
done; done; done
