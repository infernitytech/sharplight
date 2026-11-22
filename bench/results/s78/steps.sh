#!/bin/bash
# steps.sh <outdir> <pairs> <spec...>: inside the lock; spec = label:tree[:ENV=V,ENV2=V]; runs gpt-oss text/feedback and Qwen3 text per spec, alternated
. <path>
O=$1; N=$2; shift 2; mkdir -p $O
GO=~/models/hf/gpt-oss-20b; QW=~/models/hf/Qwen3-8B-FP8
A="--provider ${PROV:-cuda} --slots 64 --ctx 640 --depth 256 --sizes ${SIZES:-1,16,64} --modes greedy --reps 20"
for r in $(seq 1 $N); do for spec in "$@"; do
  IFS=: read lab tree envs <<< "$spec"; E=$(echo "$envs" | tr ',' ' ')
  BB="dotnet $tree/bench/BatchBench/bin/Release/net11.0/BatchBench.dll"
  for job in ${JOBS:-gptoss_text gptoss_feedback qwen_text}; do
    case $job in gptoss_text) m=$GO; f=--text;; gptoss_feedback) m=$GO; f=--feedback;; qwen_text) m=$QW; f=--text;; q25) m=~/models-local/gguf/Qwen2.5-7B-Instruct/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf; f=--text;; esac
    env $E timeout 900 nice -n 5 $BB $m $A $f > $O/${job}_${lab}_$r.txt 2>&1
    echo "$job $lab r$r: $(grep -E '^\| [0-9]+ ' $O/${job}_${lab}_$r.txt | awk -F'|' '{printf "%s:%s  ", $2, $5}')"
  done
done; done
