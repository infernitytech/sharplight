#!/bin/bash
# ABBA BatchBench: bbab.sh <tag> <treeA> <treeB> <mixed> <model> [extra args]
. <path>
export CUDA_VISIBLE_DEVICES=0
tag=$1; A=$2; B=$3; mixed=$4; path=$5; shift 5
m=$(basename $path)
for lab in A1 B2 B3 A4; do
  case $lab in A*) T=$A;; B*) T=$B;; esac
  export XDG_CACHE_HOME=$WORK/cache_bb_$(basename $T)
  f=$OUT/steps/${tag}_${m}_${lab}.txt
  timeout 1800 dotnet $T/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $path --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed $mixed "$@" > $f 2>&1
  echo "$tag $m $lab: $(grep -E '^\| [0-9]+ \| [0-9]+ \(' $f | awk -F'|' '{split($6,a,","); printf "%s+%s:%s ", $2+0, $3+0, a[2]+0}')"
done
