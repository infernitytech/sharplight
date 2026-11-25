#!/bin/bash
# bb3.sh <tag> <mixed> <model> <tree>... (order: trees, then reversed)
. <path>
export CUDA_VISIBLE_DEVICES=0
tag=$1; mixed=$2; path=$3; shift 3
m=$(basename $path); trees=("$@"); n=${#trees[@]}
order=(); for i in $(seq 0 $((n-1))); do order+=($i); done; for i in $(seq $((n-1)) -1 0); do order+=($i); done
k=0
for i in "${order[@]}"; do T=${trees[$i]}; k=$((k+1))
  export XDG_CACHE_HOME=$WORK/cache_bb_$(basename $T)
  f=$OUT/steps/${tag}_${m}_$(basename $T)_$k.txt
  timeout 1800 dotnet $T/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $path --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed $mixed > $f 2>&1
  echo "$tag $m $(basename $T) $k: $(grep -E '^\| [0-9]+ \| [0-9]+ \(' $f | awk -F'|' '{split($6,a,","); printf "%s+%s:%s ", $2+0, $3+0, a[2]+0}')"
done
