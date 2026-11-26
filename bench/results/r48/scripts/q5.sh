#!/bin/bash
# Queue 5: vLLM 0.33 references in the same session, 64-stream A/B (Llama-3.1-8B), and the branch on gpt-oss
B=~/sl-worktrees/perf-r48-base; H=~/models/hf
./serve.sh vllm vllm llama $H/Llama-3.1-8B-Instruct 16 --gpu-memory-utilization 0.6
./serve.sh vllm64 vllm llama $H/Llama-3.1-8B-Instruct 64 --gpu-memory-utilization 0.8
./serve.sh vllm vllm gptoss $H/gpt-oss-20b 16 --gpu-memory-utilization 0.6
REPO=$B ./serve.sh c64/base sharplight llama $H/Llama-3.1-8B-Instruct 64 --provider cuda
./serve.sh c64/br sharplight llama $H/Llama-3.1-8B-Instruct 64 --provider cuda
echo q5-done
