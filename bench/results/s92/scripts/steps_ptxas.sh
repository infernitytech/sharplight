#!/bin/bash
# BatchBench steps with the driver's JIT vs external ptxas 13.0 / 12.9 (alternated), own cache dir per form.
. $ENV_SH
O=$OUT/steps; mkdir -p $O
BB=$TREE/bench/BatchBench/bin/Release/net11.0/BatchBench
M=${M:-$HOME/models/hf/Qwen3-8B-FP8}; N=$(basename $M)
form() {
  case $1 in
    drv) unset SHARPLIGHT_CUDA_PTXAS SHARPLIGHT_CUDA_PTXAS_ISA;;
    p130) export SHARPLIGHT_CUDA_PTXAS=$PTXAS_13_0 SHARPLIGHT_CUDA_PTXAS_ISA=9.0;;
    p129) export SHARPLIGHT_CUDA_PTXAS=$PTXAS_12_9 SHARPLIGHT_CUDA_PTXAS_ISA=8.8;;
  esac
  export XDG_CACHE_HOME=$WORK/cache_$1
}
for r in 1 2; do for f in ${FORMS:-drv p130 p129}; do form $f
  echo "== $N $f round $r"
  $BB $M --provider cuda --slots 64 --ctx 640 --depth 256 --batch 2048 --modes greedy --reps 5 \
    --mixed "${MIXED:-16+0,64+0,0+250x3,0+250x8,15+250x1,13+250x3,48+250x4}" 2>&1 | grep -E "^\|" 
done; done
