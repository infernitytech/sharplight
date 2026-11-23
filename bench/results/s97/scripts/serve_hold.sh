#!/bin/bash
. <env.sh>
cd $OUT
ROUNDS=2 ./serve_ab.sh $OUT/serve "new base vllm base new" gptoss
ROUNDS=2 ./serve_ab.sh $OUT/serve "new base vllm" qwen
