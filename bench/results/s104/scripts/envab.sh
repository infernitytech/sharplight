#!/bin/bash
# same tree, env A/B ABBA: envab.sh <tag> <tree> <mixed> <model> <envA> <envB>
. <path>
export CUDA_VISIBLE_DEVICES=0
tag=$1; T=$2; mixed=$3; path=$4; EA=$5; EB=$6
m=$(basename $path); export XDG_CACHE_HOME=$WORK/cache_bb_$(basename $T)
for lab in A1 B2 B3 A4; do
  case $lab in A*) E=$EA;; B*) E=$EB;; esac
  f=$OUT/steps/${tag}_${m}_${lab}.txt
  env $E timeout 1800 dotnet $T/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $path --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed $mixed > $f 2>&1
  echo "$tag $m $lab ($E): $(grep -E '^\| [0-9]+ \| [0-9]+ \(' $f | awk -F'|' '{split($6,a,","); printf "%s+%s:%s ", $2+0, $3+0, a[2]+0}')"
done
