#!/bin/bash
# after the FP8 decode fix: decode steps develop vs r41 (Qwen3-8B-FP8), serving r41 vs vLLM with correct text
. $OUT/env.sh
cd $OUT
SIZES=1,8,9,12,16,24,32 REPS=2 ./steps.sh dec $M8 "base|base|X=1" "new|new|X=1"
export CFG_xn48="X=1"
CONCS="16 64" ROUNDS=1 ./hold.sh ./serve_ab.sh $OUT/abfix "xn48 vllm xn48" qwen
