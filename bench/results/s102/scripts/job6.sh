#!/bin/bash
# steps ABBA: base new new base on each model
. <path>
export CUDA_VISIBLE_DEVICES=0
LL=~/models-local/hf/Llama-3.1-8B-Instruct Q8=~/models-local/hf/Qwen3-8B
for spec in "$LL:16+0,15+10,16+112,16+240,16+250,16+304,16+368,16+496,16+512,16+752,16+762,16+1008,16+2032" "$Q8:16+0,16+240,16+250,16+304,16+496,16+512,16+1008" "$MOSS:16+0,16+240,16+250,16+304,16+496,16+512,16+1008" "$M8:9+0,12+0,16+0,24+0,16+240,16+250,16+496,16+512"; do
  path=${spec%%:*}; mixed=${spec#*:}; m=$(basename $path); i=0
  for lab in base new new base; do i=$((i+1))
    [ $lab = base ] && T=$BASE || T=$TREE
    export XDG_CACHE_HOME=$WORK/cache_bb_$lab
    f=$OUT/steps/ab_${m}_${lab}_$i.txt
    timeout 1200 dotnet $T/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $path --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed $mixed > $f 2>&1
    echo "$m $lab $i rc=$? $(grep -E "^\| [0-9]+ \| [0-9]+ \(" $f | awk -F"|" "{split(\$6,a,\",\"); printf \"%s+%s:%s \", \$2+0, \$3+0, a[2]}")"
  done
done
