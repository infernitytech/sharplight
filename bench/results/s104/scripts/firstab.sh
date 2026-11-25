#!/bin/bash
# first-run A/B: firstab.sh <tag> <rounds> <provider> <models...>
. <path>
export CUDA_VISIBLE_DEVICES=0
tag=$1; rounds=$2; prov=$3; shift 3
for r in $(seq $rounds); do for path in "$@"; do m=$(basename $path); for lab in base new; do
  [ $lab = base ] && T=$BASE || T=$TREE
  [ $r = 2 ] && { [ $lab = base ] && T=$TREE || T=$BASE; lab=$([ $lab = base ] && echo new || echo base); }
  export XDG_CACHE_HOME=$WORK/cache_bb_$lab
  f=$OUT/steps/${tag}_${m}_${prov}_${lab}_$r.txt
  timeout 1500 dotnet $T/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $path --provider $prov --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 5 --text \
    --mixed 15+10,14+10,13+11,12+12,16+16,15+17,16+240,15+241,16+250,8+0,24+0 > $f 2>&1
  echo "$tag $m $prov $lab r$r: $(grep -E '^\| [0-9]+ \| [0-9]+ \(' $f | awk -F'|' '{split($6,a,","); printf "%s+%s:%s/%s ", $2+0, $3+0, $4+0, a[2]+0}')"
done; done; done
