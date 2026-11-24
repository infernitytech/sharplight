#!/bin/bash
. <path>
export CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME=$WORK/cache_bb_new
LL=~/models-local/hf/Llama-3.1-8B-Instruct
for r in 1 2; do for cfg in 0 3 auto; do
  if [ $cfg = auto ]; then unset SHARPLIGHT_NATIVE_F16_CONFIG; else export SHARPLIGHT_NATIVE_F16_CONFIG=$cfg; fi
  f=$OUT/steps/cal_cfg${cfg}_$r.txt
  timeout 1200 dotnet $TREE/bench/BatchBench/bin/Release/net11.0/BatchBench.dll $LL --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed 16+0,16+48,16+112,16+176,16+240,16+250,16+304,16+368,16+496,16+512,16+1008,16+2032 > $f 2>&1
  echo "cfg$cfg r$r rc=$? $(grep -E "^\| [0-9]+ \| [0-9]+ \(" $f | awk -F"|" "{split(\$6,a,\",\"); printf \"%s+%s:%s \", \$2+0, \$3+0, a[2]}")"
done; done
