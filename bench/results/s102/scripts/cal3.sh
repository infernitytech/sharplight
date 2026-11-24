#!/bin/bash
. <path>
export CUDA_VISIBLE_DEVICES=0
LL=~/models-local/hf/Llama-3.1-8B-Instruct
i=0; for cfg in base new new base base new; do i=$((i+1))
  T=$TREE; export XDG_CACHE_HOME=$WORK/cache_bb_new
  [ $cfg = base ] && { T=$BASE; export XDG_CACHE_HOME=$WORK/cache_bb_base; }
  f=$OUT/steps/cal3_${cfg}_$i.txt
  timeout 1200 dotnet $T/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $LL --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed 16+240,16+250,16+368,16+496,16+1008,16+2032 > $f 2>&1
  echo "$cfg $i rc=$? $(grep -E "^\| [0-9]+ \| [0-9]+ \(" $f | awk -F"|" "{split(\$6,a,\",\"); printf \"%s+%s:%s \", \$2+0, \$3+0, a[2]}")"
done
