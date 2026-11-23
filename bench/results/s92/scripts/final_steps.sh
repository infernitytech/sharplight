#!/bin/bash
# develop vs perf/llm-r34 prompt steps on a second FP8 model (Qwen3.8-27B-FP8) and Qwen3-8B-FP8 again, alternated
. $ENV_SH
for M in ~/models/hf/Qwen3.8-27B-FP8 ~/models/hf/Qwen3-8B-FP8; do N=$(basename $M)
for r in 1 2; do for f in base new; do
  [ $f = base ] && W=$BASE || W=$TREE
  export XDG_CACHE_HOME=$WORK/cache_final_$f
  echo "== $N $f round $r"
  $W/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider cuda --slots 16 --ctx 1024 --depth 256 --batch 2048 --modes greedy --reps 5 \
    --mixed "16+0,0+250x3,0+250x8,15+250x1,13+250x3" 2>&1 | grep -E "^\||rror|xception"
done; done; done
