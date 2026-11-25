#!/bin/bash
# Saturation matrix r2: per model, develop's build and vLLM; fixed lengths (6 requests per stream) and the length mix (8 per stream).
S=<path>
for m in ${MODELS:-qwen gptoss llama}; do
  case $m in qwen) KV=127632 ;; gptoss) KV=263424 ;; llama) KV=100480 ;; esac
  PER_OVERRIDE=6 $S r2 sharplight $m "16 64 128 256" fixed --kv-tokens $KV
  PER_OVERRIDE=6 $S r2 vllm $m "16 64 128 256" fixed --max-num-batched-tokens 2048
  PER_OVERRIDE=8 $S r2 sharplight $m "64 128 256" mix --kv-tokens $KV
  PER_OVERRIDE=8 $S r2 vllm $m "64 128 256" mix --max-num-batched-tokens 2048
done
