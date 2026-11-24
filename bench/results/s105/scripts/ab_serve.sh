#!/bin/bash
# Serving A/B develop (base) vs this branch (new): length mix, 128 and 256 streams, 8 requests per stream, order base new new base.
S=<path>
for m in ${MODELS:-qwen llama gptoss}; do
  case $m in qwen) KV=127632 ;; gptoss) KV=263424 ;; llama) KV=100480 ;; esac
  i=0
  for v in base new new base; do i=$((i+1))
    V=$v SUFFIX=_${v}$i PER_OVERRIDE=8 $S ab sharplight $m "128 256" mix --kv-tokens $KV
  done
done
