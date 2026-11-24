#!/bin/bash
. <path>
cd $OUT
LL=~/models-local/hf/Llama-3.1-8B-Instruct Q8=~/models-local/hf/Qwen3-8B
./greedy.sh "gpt-oss-20b:$MOSS" "Qwen3-8B-FP8:$M8" "Llama-3.1-8B:$LL"
for c in 1000 1024; do for lab in base new; do CHUNK=$c ./ppl.sh $lab $LL llama 1; done; done
for lab in base new; do CHUNK=1000 ./ppl.sh $lab $Q8 qwen3-8b 1; done
for lab in base new; do CHUNK=1000 ./ppl.sh $lab $MOSS gptoss 1; done
