#!/bin/bash
# gpu240.sh <tag> <tree> <cache> <model> [env]
. <path>
export CUDA_VISIBLE_DEVICES=0
tag=$1; T=$2; C=$3; path=$4; shift 4
export XDG_CACHE_HOME=$WORK/$C
f=$OUT/steps/g240_${tag}.txt
env "$@" timeout 1500 dotnet $T/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $path --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps ${REPS:-10} --text \
    --mixed 16+0,16+240,16+250,16+496,16+512 > $f 2>&1
echo "$tag: $(grep -E '^\| [0-9]+ \| [0-9]+ \(' $f | awk -F'|' '{split($6,a,","); printf "%s+%s:%s ", $2+0, $3+0, a[2]+0}')"
