#!/bin/bash
source /workspace/tools/env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME=/workspace/work/fp4/cache
BB=<tree>/bench/BatchBench/bin/Release/net11.0/BatchBench
lab=$1; M=$2; shift 2
env "$@" timeout 1500 $BB $M --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 5 --text --mixed 16+240 --profile 0 --profile-top 30 > /workspace/work/fp4/steps/prof_${lab}_mixed.txt 2>&1
env "$@" timeout 1500 $BB $M --provider cuda --slots 64 --ctx 512 --depth 256 --sizes 16,64 --modes greedy --reps 5 --text --profile 16,64 --profile-mode greedy --profile-top 30 > /workspace/work/fp4/steps/prof_${lab}_decode.txt 2>&1
echo "prof $lab done"
