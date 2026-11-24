#!/bin/bash
# first-run step costs; usage: first.sh <tag> <tree> <model> [env...]
. <path>
export CUDA_VISIBLE_DEVICES=0
tag=$1; T=$2; path=$3; shift 3
export XDG_CACHE_HOME=$WORK/cache_bb_$(basename $T)
env "$@" timeout 1200 dotnet $T/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $path --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 5 --text \
  --mixed 15+10,14+10,13+11,12+12,16+16,15+17,16+240,15+241 > $OUT/steps/first_${tag}.txt 2>&1
echo "$tag rc=$? $(grep -E '^\| [0-9]+ \| [0-9]+ \(' $OUT/steps/first_${tag}.txt | awk -F'|' '{printf "%s+%s:%s ", $2+0, $3+0, $5+0}')"
