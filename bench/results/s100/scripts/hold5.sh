#!/bin/bash
. /workspace/work/perf-r40/env.sh
cd $OUT; mkdir -p prof
REPS=2 SIZES=16 ./steps.sh any $MOSS "new|new|X=1" "any|new|SHARPLIGHT_CUDA_PDL_SHARED=any" "base|base|X=1"
exec 9>/root/.sl-verify.lock; flock 9
export CUDA_VISIBLE_DEVICES=0
for v in "new|new|X=1" "any|new|SHARPLIGHT_CUDA_PDL_SHARED=any" "base|base|X=1"; do IFS='|' read -r label tr envs <<< "$v"
  T=$TREE; [ $tr = base ] && T=$BASE
  f=$OUT/prof/n16_$label; rm -f $f.nsys-rep $f.sqlite
  env $envs XDG_CACHE_HOME=$WORK/cache_bb_$tr nsys profile -t cuda -s none --cpuctxsw=none --cuda-graph-trace=node -o $f $T/bench/BatchBench/bin/Release/net11.0/BatchBench $MOSS --provider cuda --slots 64 --ctx 640 --depth 256 --sizes 16 --modes greedy --reps 40 --text 9>&- > $f.txt 2>&1
done
