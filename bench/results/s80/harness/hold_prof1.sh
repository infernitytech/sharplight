#!/bin/bash
. <path>
cd ~/sl-worktrees/perf-r27
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0
export VLLM_VENV=~/venvs/vllm-r12 QUIET_GPU=1 ROUNDS=1 O=<path> OUT_DIR=<path>
M=~/models-local/hf/gpt-oss-20b
echo "sl $(date +%T)"; SHARPLIGHT_STEP_PROFILE=$O/steps64.csv bench/s16_serve.sh sharplight gptoss $M "64" --provider cuda
SHARPLIGHT_SCHED_TRACE=1 bench/s16_serve.sh sharplight gptoss_trace $M "64" --provider cuda
echo "mixed $(date +%T)"
BB=bench/BatchBench/bin/Release/net11.0/BatchBench
timeout 900 $BB $M --provider cuda --slots 72 --ctx 640 --depth 256 --batch 2048 --modes greedy --reps 10 --mixed "64+0,8+250x8,0+256x8" --profile 0,1 --profile-top 30 > $O/mixed.txt 2>&1
echo "done $(date +%T)"
