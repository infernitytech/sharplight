#!/bin/bash
# vprof.sh <qwen|gptoss|llama>: vLLM's decode step kernels at 1/16/64/128/256 sequences (depth 256, word prompts) and a 2048-token
# prefill (bench/s54_vllm_decode_profile.py, torch profiler), each in its own bench-lock hold.
. <path>
m=$1; case $m in qwen) M=$M8 ;; gptoss) M=$MOSS ;; llama) M=$ML ;; esac
O=$OUT/vprof; mkdir -p $O
export CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME=$WORK/cache_serve VLLM_USE_FLASHINFER_SAMPLER=0
for s in ${SEQS:-1 16 64 128 256}; do
  flock ~/.sl-bench.lock $PY -I $TREE/bench/s54_vllm_decode_profile.py $M --seqs $s --depth 256 --steps 32 --words --ctx 1024 --gpu-util 0.85 --top 60 > $O/${m}_dec$s.txt 2>$O/${m}_dec$s.err
  echo "$m dec$s rc=$?"
done
flock ~/.sl-bench.lock $PY -I $TREE/bench/s54_vllm_decode_profile.py $M --seqs 1 --prefill 2048 --ctx 4096 --gpu-util 0.85 --top 60 > $O/${m}_pre2048.txt 2>$O/${m}_pre2048.err
echo "$m pre2048 rc=$?"
