#!/bin/bash
cd /workspace/work/fp4
export QUIET_GPU=1
flock ~/.sl-bench.lock ./serve.sh vllm q8rh /workspace/work/fp4/models/RedHatAI_Qwen3-8B-NVFP4 --linear-backend cutlass
flock ~/.sl-bench.lock ./serve.sh sharplight q8rh /workspace/work/fp4/slm/RedHatAI_Qwen3-8B-NVFP4.slm
