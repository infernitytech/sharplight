#!/bin/bash
W=<path> N=~/sl-worktrees/perf-r26; B=~/sl-worktrees/r26-base; O=$W/final/long; mkdir -p $O
echo "long $(date +%T)"
for r in 1 2; do $W/long.sh $B base$r $O 130048 X=1; $W/long.sh $N new$r $O 130048 X=1; done
SIZES=1,4 $W/long.sh $N new32k $O 32768 X=1; $W/long.sh $B base32k $O 32768 X=1
echo "vllm profile $(date +%T)"; cd $N
timeout 1500 ~/venvs/vllm-r12/bin/python bench/s54_vllm_decode_profile.py ~/models/hf/gpt-oss-20b --seqs 4 --depth 130048 --ctx 131072 --steps 64 --skip 5 --gpu-util 0.92 --top 30 > $O/vllm_prof_d130048_x4.txt 2>&1
grep "decode steps" $O/vllm_prof_d130048_x4.txt
echo "vllm tok/s $(date +%T)"
timeout 2400 ~/venvs/vllm-r12/bin/python -I bench/s58_long_vllm.py ~/models/hf/gpt-oss-20b gptoss --lens "" --depths 32768,130048 --streams 1,4 --max-len 130560 --gpu-util 0.92 --out $O/vllm_tg.json > $O/vllm_tg.log 2>&1
grep tg128 $O/vllm_tg.log
echo "vulkan qwen $(date +%T)"
PROV=vulkan SIZES=1,64 JOBS="qwen_text" $W/steps.sh $W/final/vulkan2 1 new:$N base:$B
SHARPLIGHT_MULTI_DEBUG=1 timeout 300 $N/bench/BatchBench/bin/Release/net11.0/BatchBench ~/models-local/gguf/Qwen2.5-7B-Instruct/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --provider cuda --slots 16 --ctx 640 --depth 256 --sizes 16 --modes greedy --reps 3 --text > $W/final/q25_multi_debug.txt 2>&1
grep -c "bias True" $W/final/q25_multi_debug.txt
echo "done $(date +%T)"
