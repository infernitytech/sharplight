#!/bin/bash
. <env.sh>
cd $OUT
MODEL=~/models/gguf/Qwen3-8B/Qwen3-8B-Q4_K_M.gguf ./steps.sh kq
for v in base new; do T=$TREE; [ $v = base ] && T=$BASE; XDG_CACHE_HOME=$WORK/cache_bb_$v $T/bench/BatchBench/bin/Release/net11.0/BatchBench $MOSS --provider cuda --slots 64 --ctx 640 --depth 256 --sizes 64 --modes greedy --reps 20 --text --profile 64 --profile-top 15 > $OUT/steps/profile64_$v.txt 2>&1; done
