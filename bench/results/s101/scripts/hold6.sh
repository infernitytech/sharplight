#!/bin/bash
# open-loop (Poisson) A/B: develop-48, r41-48, r41-48 tile-aware off, vLLM
. $OUT/env.sh
cd $OUT
B=$BASE/samples/Sharplight.Cli/bin/Release/net11.0/sharplight
export CFG_xb48="SHARPLIGHT_SERVER=$B SHARPLIGHT_STEP_ROWS_PER_SEQUENCE=48"
export CFG_xn48="X=1"
export CFG_xn48off="SHARPLIGHT_STEP_TILE_AWARE=0"
for m in qwen gptoss; do
  ARRIVAL=poisson CONCS="16 64" ROUNDS=1 ./hold.sh ./serve_ab.sh $OUT/abp "xb48 xn48 xn48off vllm xn48 xb48 xn48off" $m
done
