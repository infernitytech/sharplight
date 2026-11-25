#!/bin/bash
# Queue against Preempt (order queue preempt preempt queue) at 256 streams, length mix, 8 requests per stream, the KV tokens vLLM has.
S=$(dirname "$0")/sat.sh
for m in ${MODELS:-llama qwen}; do
  case $m in qwen) KV=127632 ;; llama) KV=100480 ;; esac
  i=0
  for p in ${ORDER:-queue preempt preempt queue}; do i=$((i+1))
    SUFFIX=_${p}$i PER_OVERRIDE=8 $S kvadm sharplight $m "256" mix --kv-tokens $KV --kv-admission $p
  done
done
