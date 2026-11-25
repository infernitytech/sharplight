#!/bin/bash
# FP8 KV measurements: BatchBench decode / prompt / mixed steps f16 against f8e4m3_b32 (order f16 f8 f8 f16), then PplBench.
D=$(dirname "$0")
MIX="1+0,64+0,128+0,256+0,0+2048,256+256x7"; export CTX=2560
for m in qwen llama; do
  i=0; for kt in f16 f8e4m3_b32 f8e4m3_b32 f16; do i=$((i+1)); $D/bb.sh $m $kt r$i "$MIX"; done
done
for m in qwen llama; do
  for kt in f16 f8e4m3_b32 q8_0; do $D/ppl.sh $m $kt 1024; done
  for kt in f16 f8e4m3_b32; do TOKENS=6144 $D/ppl.sh $m $kt 8; done
done
