#!/bin/bash
# Hold 1: sharplight develop serving vs vLLM serving, gpt-oss-20b, 16/64 streams.
. <path>
cd ~/sl-worktrees/perf-r27
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0
export VLLM_VENV=~/venvs/vllm-r12 QUIET_GPU=1 ROUNDS=2 OUT_DIR=<path>
M=~/models-local/hf/gpt-oss-20b
echo "sl $(date +%T)"; bench/s16_serve.sh sharplight gptoss $M "16 64" --provider cuda
echo "vllm $(date +%T)"; bench/s16_serve.sh vllm gptoss $M "16 64" --gpu-memory-utilization 0.6
echo "done $(date +%T)"
