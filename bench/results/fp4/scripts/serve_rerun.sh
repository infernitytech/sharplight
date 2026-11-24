#!/bin/bash
cd /workspace/work/fp4
export QUIET_GPU=1
flock ~/.sl-bench.lock ./serve.sh sharplight q8nvfp4b /workspace/work/fp4/slm/nvidia_Qwen3-8B-FP4.slm
flock ~/.sl-bench.lock ./serve.sh sharplight q8fp8 ~/models/hf/Qwen3-8B-FP8
