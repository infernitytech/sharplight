#!/bin/bash
# Follow-up of 109: Qwen perplexities, FP8 KV serving against vLLM's fp8 KV cache (length mix at 256 streams; long context: 32 streams
# of ~8K-token prompts), the Qwen3-8B-FP8 preempt rerun, clock windows of the Llama decode steps.
D=$(dirname "$0"); . $D/env.sh
vkv() { grep -h "GPU KV cache size" $OUT/sat/$1/*/serve_vllm_*.server.log 2>/dev/null | tail -1 | grep -oE "size: [0-9,]+" | tr -dc 0-9; }
# 1. perplexities
TOKENS=6144 $D/ppl.sh qwen f16 8
for kt in f8e4m3_b32 q8_0; do $D/ppl.sh qwen $kt 1024; done
# 2. FP8 KV serving, length mix, 256 streams: vLLM fp8 first (its KV tokens size ours)
for m in llama qwen; do
  PER_OVERRIDE=8 SUFFIX=_fp8 $D/sat.sh f8mix vllm $m 256 mix --max-num-batched-tokens 2048 --kv-cache-dtype fp8
  KV=$(vkv f8mix/${m}_vllm_mix_fp8 ); echo "vLLM $m fp8 KV tokens $KV"
  [ -n "$KV" ] && PER_OVERRIDE=8 SUFFIX=_f8 $D/sat.sh f8mix sharplight $m 256 mix --kv-tokens $KV --kv-type f8e4m3_b32
done
# 3. long context: Llama, 32 streams, ~8K-token prompts, 256 output tokens, 3 requests per stream
export SLOTS=32 CTX=9216 PER_OVERRIDE=3 LA_EXTRA="--prompt-words 6000 --max-tokens 256"
for kd in auto fp8; do
  SUFFIX=_$kd $D/sat.sh long vllm llama 32 fixed --max-num-batched-tokens 2048 --kv-cache-dtype $kd
done
KV16=$(vkv long/llama_vllm_fixed_auto); KV8=$(vkv long/llama_vllm_fixed_fp8); echo "long KV $KV16 $KV8"
SUFFIX=_f16 $D/sat.sh long sharplight llama 32 fixed --kv-tokens $KV16
SUFFIX=_f8 $D/sat.sh long sharplight llama 32 fixed --kv-tokens $KV8 --kv-type f8e4m3_b32
unset SLOTS CTX PER_OVERRIDE LA_EXTRA
# 4. Qwen3-8B-FP8 queue / preempt rerun
MODELS=qwen ORDER="preempt queue" $D/kvadm.sh
# 5. clock windows of the Llama decode steps
for kt in f16 f8e4m3_b32; do
  flock ~/.sl-bench.lock env CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME=$WORK/cache_bb $PY $TREE/bench/s73_clock_sampler.py --gpu 0 --interval 0.01 \
    --out $OUT/bb/llama_${kt}_clk.csv -- $TREE/bench/BatchBench/bin/Release/net11.0/BatchBench $ML --provider cuda --slots 264 --ctx 2560 \
    --kv-tokens 90112 --kv-type $kt --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed 128+0,256+0 --hold-ms 2000 \
    > $OUT/bb/llama_${kt}_clk.txt 2> $OUT/bb/llama_${kt}_clk.err
  echo "clk $kt rc=$?"
done
