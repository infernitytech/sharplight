#!/bin/bash
W=<path> N=~/sl-worktrees/perf-r26; B=~/sl-worktrees/r26-base; O=$W/final; mkdir -p $O/vllm
Q25="~/models-local/gguf/Qwen2.5-1.5B-Instruct/qwen2.5-1.5b-instruct-q8_0.gguf ~/models-local/gguf/Qwen2.5-7B-Instruct/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
echo "greedy $(date +%T)"
for t in base:$B new:$N; do IFS=: read l tr <<< "$t"; $W/greedy.sh $tr $l $O/greedy ~/models/hf/gpt-oss-20b ~/models/hf/Qwen3-8B-FP8; PROVIDERS=cuda $W/greedy.sh $tr $l $O/greedy $Q25; done
echo "steps1 $(date +%T)"
JOBS="gptoss_text gptoss_feedback qwen_text q25" $W/steps.sh $O/steps1 1 new:$N base:$B
echo "vllm $(date +%T)"; cd $N
timeout 1200 ~/venvs/vllm-r12/bin/python bench/s54_vllm_decode_profile.py ~/models/hf/gpt-oss-20b --seqs 64 --words --skip 5 > $O/vllm/gptoss.txt 2>&1
timeout 1200 ~/venvs/vllm-r12/bin/python bench/s54_vllm_decode_profile.py ~/models/hf/Qwen3-8B-FP8 --seqs 64 --words --skip 5 > $O/vllm/qwen.txt 2>&1
grep "decode steps" $O/vllm/*.txt
echo "steps2 $(date +%T)"
JOBS="gptoss_text gptoss_feedback qwen_text q25" $W/steps.sh $O/steps2 1 base:$B new:$N
echo "vulkan $(date +%T)"
PROV=vulkan SIZES=1,64 JOBS="gptoss_text qwen_text" $W/steps.sh $O/vulkan 1 base:$B new:$N
echo "done $(date +%T)"
