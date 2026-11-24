#!/bin/bash
# usage: serve.sh <engine sharplight|vllm> <name> <model> [extra...]; one hold per call (bench lock taken by the caller)
source /workspace/tools/env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME=/workspace/work/fp4/cache
export VLLM_VENV=<venvs>/vllm-r12 OUT_DIR=<tree>/bench/results/fp4/serve ROUNDS=${ROUNDS:-2} FIRST=${FIRST:-0}
cd <tree>
eng=$1; name=$2; m=$3; shift 3
if [ $eng = sharplight ]; then
  bench/s16_serve.sh sharplight $name $m "${CONCS:-1 16 64}" --provider cuda "$@"
else
  bench/s16_serve.sh vllm $name $m "${CONCS:-1 16 64}" --gpu-memory-utilization ${VMEM:-0.6} "$@"
fi
echo "serve $eng $name rc=$?"
