#!/bin/bash
. <path>
export CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME=$WORK/cache_serve VLLM_USE_FLASHINFER_SAMPLER=0
flock ~/.sl-bench.lock $PY -I $TREE/bench/s54_vllm_decode_profile.py $ML --seqs 16 --depth 256 --steps 32 --words --ctx 1024 --gpu-util 0.85 --top 60 > $OUT/vprof/llama_dec16.txt 2>$OUT/vprof/llama_dec16.err
echo "llama dec16 redo rc=$?"
