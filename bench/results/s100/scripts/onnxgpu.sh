#!/bin/bash
# usage: onnxgpu.sh <out.jsonl> <rounds> <engines> <model>...   one GPU-lock hold per model (§98 harness)
. /workspace/tools/env.sh
cd /root/sl-worktrees/perf-r40
export ORT_VENV=/root/venvs/ort; . bench/onnx/env.sh
export TRT_LIB=/root/venvs/ort/lib/python3.12/site-packages/tensorrt_libs
export LD_LIBRARY_PATH=$TRT_LIB:$LD_LIBRARY_PATH
export SHARPLIGHT_ONNX_MODELS=/root/models-local/onnx SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME=/root/perf-r40-work/cache_onnx
export ONNX_BENCH_DLL_BASE=/root/sl-worktrees/perf-r40-base/bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll
out=$1; rounds=$2; engines=$3; shift 3
for m in "$@"; do
  exec 9>/root/.sl-verify.lock
  flock 9
  echo "$(date +%T) lock held ($m); apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
  python3 bench/onnx/parity_bench.py run "$out" --rounds "$rounds" --runs 30 --engines "$engines" --models "$m" 9>&-
  flock -u 9; exec 9>&-
  echo "$(date +%T) released ($m)"
done
