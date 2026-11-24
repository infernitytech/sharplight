#!/bin/bash
# prof.sh <tag> <tree> <model> <mixed> <profile idx list> [env...]
. <path>
export CUDA_VISIBLE_DEVICES=0
tag=$1; T=$2; path=$3; mixed=$4; pidx=$5; shift 5
export XDG_CACHE_HOME=$WORK/cache_bb_$(basename $T)
mkdir -p $OUT/prof
env "$@" timeout 1800 dotnet $T/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $path --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed $mixed --profile $pidx --profile-mode greedy --profile-top 60 > $OUT/prof/${tag}.txt 2>&1
echo "$tag rc=$?"
