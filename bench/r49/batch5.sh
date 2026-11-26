#!/bin/bash
M=${SL_MODELS:-models}/Qwen3-8B-FP8
O=${SL_LOGS:-logs}/pp5; mkdir -p $O
echo "start $(uptime)"
for N in 512 2048 8192; do
  for v in off on; do
    if [ $v = off ]; then export SHARPLIGHT_MULTI_GEMM_TALL=0; else unset SHARPLIGHT_MULTI_GEMM_TALL; fi
    NOLOCK=1 ./r49-pp.sh $O $v vulkan $M $N
  done
  unset SHARPLIGHT_MULTI_GEMM_TALL
  NOLOCK=1 ./r49-pp.sh $O on cuda $M $N
done
# second round, reversed order
for N in 512 2048 8192; do for v in on off; do
    if [ $v = off ]; then export SHARPLIGHT_MULTI_GEMM_TALL=0; else unset SHARPLIGHT_MULTI_GEMM_TALL; fi
    NOLOCK=1 ./r49-pp.sh $O ${v}2 vulkan $M $N
done; done
unset SHARPLIGHT_MULTI_GEMM_TALL
./prof.sh $O/prof_on.txt $M 2048
echo "end $(uptime)"
