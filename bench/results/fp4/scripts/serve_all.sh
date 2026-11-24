#!/bin/bash
# Each serving run in its own bench-lock hold (rounds of the perf round interleave).
cd /workspace/work/fp4
S8=/workspace/work/fp4/slm/nvidia_Qwen3-8B-FP4.slm; H8=/workspace/work/fp4/models/nvidia_Qwen3-8B-FP4
SM=/workspace/work/fp4/slm/RedHatAI_Qwen3-30B-A3B-NVFP4.slm; HM=/workspace/work/fp4/models/RedHatAI_Qwen3-30B-A3B-NVFP4
flock ~/.sl-bench.lock ./serve.sh vllm q8nvfp4 $H8 --linear-backend cutlass
flock ~/.sl-bench.lock ./serve.sh sharplight q8nvfp4 $S8
VMEM=0.85 flock ~/.sl-bench.lock ./serve.sh vllm q30nvfp4 $HM --linear-backend cutlass
grep -q "ready=1" <tree>/bench/results/fp4/serve/serve_vllm_q30nvfp4.server.log.ready || VMEM=0.85 flock ~/.sl-bench.lock ./serve.sh vllm q30nvfp4 $HM --linear-backend cutlass --moe-backend cutlass
flock ~/.sl-bench.lock ./serve.sh sharplight q30nvfp4 $SM
