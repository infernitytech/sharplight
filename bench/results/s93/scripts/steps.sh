#!/bin/bash
# develop (9-warp form) vs perf/llm-r35 (copy warpgroup + fence, E4M3 store) prompt steps, plus the r35 tree with SHARPLIGHT_FP8_WG=0, alternated
. $OUT/env.sh
for M in $M8 $M27; do N=$(basename $M)
if [ $N = Qwen3-8B-FP8 ]; then A="--slots 64 --ctx 640 --depth 256 --batch 2048 --modes greedy --reps 5 --mixed 16+0,64+0,0+250x3,0+250x8,15+250x1,13+250x3,48+250x4"
else A="--slots 4 --ctx 512 --depth 128 --batch 1024 --modes greedy --reps 5 --mixed 4+0,0+250x2,0+250x4,3+250x1"; fi
for r in 1 2; do for f in base new wg0; do
  [ $f = base ] && W=$BASE || W=$TREE
  [ $f = wg0 ] && export SHARPLIGHT_FP8_WG=0 || unset SHARPLIGHT_FP8_WG
  export XDG_CACHE_HOME=$WORK/cache_steps_$f
  echo "== $N $f round $r"
  $W/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider cuda $A 2>&1 | grep -E "^\||rror|xception"
done; done; done
