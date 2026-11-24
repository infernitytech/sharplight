#!/bin/bash
# The saturation matrix: per model ours / vLLM, fixed lengths at 16-256 streams, then the length mix at 64-256.
S=<path>
for m in ${MODELS:-qwen gptoss llama}; do
  case $m in qwen) KV=127632 ;; gptoss) KV=263424 ;; llama) KV=100480 ;; esac
  $S r1 sharplight $m "16 64 128 256" fixed --kv-tokens $KV
  $S r1 vllm $m "16 64 128 256" fixed --max-num-batched-tokens 2048
  $S r1 sharplight $m "64 128 256" mix --kv-tokens $KV
  $S r1 vllm $m "64 128 256" mix --max-num-batched-tokens 2048
done
