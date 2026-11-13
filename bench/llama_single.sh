#!/bin/bash
# usage: llama_single.sh <cuda13|vulkan> <name> <path>   -> bench/results/single_llama-<backend>_<name>.jsonl
. "$(dirname "$0")/models.sh"
B=$1; N=$2; P=$3
OUT="$(dirname "$0")/results/single_llama-${B}_${N}.jsonl"
"$(dirname "$0")/gpu_check.sh" > "${OUT%.jsonl}.gpu.txt"
LD_LIBRARY_PATH=$LLAMA_ROOT/runtime.llama-b11177-$B $LLAMA_ROOT/runtime.llama-b11177-$B/llama-bench \
  -m "$P" -ngl 99 -fa 1 -p 512,2048 -n 128 -d 0,4096 -r 5 -o jsonl > "$OUT" 2> "${OUT%.jsonl}.err"
