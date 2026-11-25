#!/bin/bash
# usage: bb.sh <label> <model> [env...]: decode steps at 1/16/32/64 sequences, then mixed / prompt steps
source /workspace/tools/env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME=/workspace/work/fp4/cache
BB=<tree>/bench/BatchBench/bin/Release/net11.0/BatchBench
lab=$1; M=$2; shift 2
env "$@" timeout 1500 $BB $M --provider cuda --slots 64 --ctx 512 --depth 256 --sizes 1,16,32,64 --modes greedy --reps 20 --text ${PROFILE:+--profile $PROFILE --profile-mode greedy} > /workspace/work/fp4/steps/${lab}_decode.txt 2>&1
echo "$lab decode rc=$?"
env "$@" timeout 1500 $BB $M --provider cuda --slots 24 --ctx 2560 --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed 16+240,16+1008,0+2048 > /workspace/work/fp4/steps/${lab}_mixed.txt 2>&1
echo "$lab mixed rc=$?"
