#!/bin/bash
cd <path>
./hold.sh ./tests.sh > job5.log 2>&1
./hold.sh ./job6.sh > job6.log 2>&1
. <path> export ROUNDS=1 CONCS="16 64"
./hold.sh ./serve_ab.sh $OUT/ab "base new vllm new base" qwen gptoss llama > job7.log 2>&1
