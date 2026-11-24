#!/bin/bash
S=<path>
for m in qwen gptoss llama; do
  case $m in qwen) KV=127632; R=15 ;; gptoss) KV=263424; R=19 ;; llama) KV=100480; R=9.5 ;; esac
  RATES="256:$R" PER_OVERRIDE=3 V=new $S open sharplight $m "256" poisson --kv-tokens $KV
  RATES="256:$R" PER_OVERRIDE=3 $S open vllm $m "256" poisson --max-num-batched-tokens 2048
done
