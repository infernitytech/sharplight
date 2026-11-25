#!/bin/bash
source /workspace/tools/env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME=/workspace/work/fp4/cache
BB=<tree>/bench/BatchBench/bin/Release/net11.0/BatchBench
M=/workspace/work/fp4/slm/nvidia_Qwen3-8B-FP4.slm
for r in 1 2; do
  for v in on off; do
    E=""; [ $v = off ] && E="SHARPLIGHT_MULTI_GEMM_TALL=0"
    env $E timeout 1500 $BB $M --provider cuda --slots 64 --ctx 512 --depth 256 --sizes 1,16,32,64 --modes greedy --reps 20 --text > /workspace/work/fp4/steps/multi_${v}_$r.txt 2>&1
  done
done
timeout 1500 $BB $M --provider cuda --slots 64 --ctx 512 --depth 256 --sizes 64 --modes greedy --reps 5 --text --profile 64 --profile-mode greedy --profile-top 20 > /workspace/work/fp4/steps/prof_multi_decode.txt 2>&1
echo "ab multi done"
