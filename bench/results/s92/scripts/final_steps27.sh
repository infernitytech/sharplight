#!/bin/bash
# develop vs perf/llm-r34 prompt steps on a second FP8 model (Qwen3.8-27B-FP8), alternated
. $ENV_SH
M=~/models/hf/Qwen3.8-27B-FP8; N=$(basename $M)
for r in 1 2; do for f in base new; do
  [ $f = base ] && W=$BASE || W=$TREE
  export XDG_CACHE_HOME=$WORK/cache_final_$f
  echo "== $N $f round $r"
  $W/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider cuda --slots 4 --ctx 512 --depth 128 --batch 1024 --modes greedy --reps 5 \
    --mixed "4+0,0+250x2,0+250x4,3+250x1" 2>&1 | grep -E "^\||rror|xception"
done; done
