#!/bin/bash
. <path>
export CUDA_VISIBLE_DEVICES=0
LL=~/models-local/hf/Llama-3.1-8B-Instruct
for r in 1 2; do for cfg in base new p0; do
  unset SHARPLIGHT_NATIVE_F16_PARTIAL; T=$TREE; export XDG_CACHE_HOME=$WORK/cache_bb_new
  [ $cfg = base ] && { T=$BASE; export XDG_CACHE_HOME=$WORK/cache_bb_base; }
  [ $cfg = p0 ] && export SHARPLIGHT_NATIVE_F16_PARTIAL=0
  f=$OUT/steps/cal2_${cfg}_$r.txt
  timeout 1200 dotnet $T/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $LL --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed 16+0,16+240,16+250,16+304,16+368,16+496,16+512,16+1008 > $f 2>&1
  echo "$cfg r$r rc=$? $(grep -E "^\| [0-9]+ \| [0-9]+ \(" $f | awk -F"|" "{split(\$6,a,\",\"); printf \"%s+%s:%s \", \$2+0, \$3+0, a[2]}")"
done; done
f=$OUT/steps/rec_new.txt
timeout 1200 dotnet $TREE/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $LL --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed 15+10,14+10,13+11,12+12,16+16,15+17,16+240,15+241 > $f 2>&1
grep -E "^\| [0-9]+ \| [0-9]+ \(" $f
