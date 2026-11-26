#!/bin/bash
# usage: r49_serve.sh <name> <model> [sharplight args]
cd <checkout>
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 VLLM_VENV=$HOME/.venvs/vllm VLLM_USE_DEEP_GEMM=0 OUT_DIR=$PWD/bench/results/r49/serve QUIET_GPU=0
mkdir -p $OUT_DIR
N=$1; M=$2; shift 2
for E in ${ENGINES:-sharplight vllm}; do
  if [ $E = sharplight ]; then A=(--provider cuda "$@"); else A=(); fi
  uptime > $OUT_DIR/run_${E}_$N.load
  timeout 2400 bench/s16_serve.sh $E $N $M "${CONCS:-16 64}" "${A[@]}" > $OUT_DIR/run_${E}_$N.log 2>&1; echo "$E rc=$?"
done
