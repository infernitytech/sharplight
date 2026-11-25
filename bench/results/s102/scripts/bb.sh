#!/bin/bash
# BatchBench mixed steps base vs new alternated; usage: bb.sh <tag> <mixed list> <rounds> <model paths...>
. <path>
export CUDA_VISIBLE_DEVICES=0
tag=$1; mixed=$2; rounds=$3; shift 3
mkdir -p $OUT/steps
for r in $(seq $rounds); do for path in "$@"; do m=$(basename $path); for lab in base new; do
  [ $lab = base ] && T=$BASE || T=$TREE
  export XDG_CACHE_HOME=$WORK/cache_bb_$lab
  timeout 1200 dotnet $T/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $path --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed $mixed > $OUT/steps/${tag}_${m}_${lab}_$r.txt 2>&1
  echo "$tag $m $lab r$r rc=$? $(grep -E "^\| [0-9]+ \| [0-9]+ \(" $OUT/steps/${tag}_${m}_${lab}_$r.txt | awk -F"|" "{split(\$6,a,\",\"); printf \"%s+%s:%s \", \$2+0, \$3+0, a[2]}")"
done; done; done
