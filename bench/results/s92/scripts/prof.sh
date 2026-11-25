#!/bin/bash
. $ENV_SH
O=$OUT/steps; BB=$TREE/bench/BatchBench/bin/Release/net11.0/BatchBench
M=${M:-$HOME/models/hf/Qwen3-8B-FP8}; N=$(basename $M)
for f in drv p130; do
  if [ $f = p130 ]; then export SHARPLIGHT_CUDA_PTXAS=$PTXAS_13_0 SHARPLIGHT_CUDA_PTXAS_ISA=9.0; fi
  export XDG_CACHE_HOME=$WORK/cache_$f
  $BB $M --provider cuda --slots 16 --ctx 640 --depth 256 --batch 2048 --modes greedy --reps 5 --mixed "0+250x8,16+0" --profile 0,1 --profile-top 30 > $O/prof_${N}_$f.txt 2>&1
done
