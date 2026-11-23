#!/bin/bash
# prompt steps with a warm kernel cache vs a fresh one
. <out>/env.sh
A="--slots 64 --ctx 640 --depth 256 --batch 2048 --modes greedy --reps 10 --mixed 16+0,0+250x3,15+250x1,48+250x4"
for f in warm fresh warm fresh2 warm; do
  case $f in warm) export XDG_CACHE_HOME=$WORK/cache_steps_def ;; fresh) rm -rf $WORK/cache_fresh; export XDG_CACHE_HOME=$WORK/cache_fresh ;; fresh2) export XDG_CACHE_HOME=$WORK/cache_fresh ;; esac
  echo "== Qwen3-8B-FP8 $f round 1"
  $TREE/bench/BatchBench/bin/Release/net11.0/BatchBench $M8 --provider cuda $A 2>&1 | grep -E "^\||rror|xception"
done
