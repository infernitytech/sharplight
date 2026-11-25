#!/bin/bash
# closed-loop A/B: develop-32, develop-48, r41-48 (tile-aware), vLLM, alternated
. $OUT/env.sh
cd $OUT
B=$BASE/samples/Sharplight.Cli/bin/Release/net11.0/sharplight
export CFG_xb32="SHARPLIGHT_SERVER=$B"
export CFG_xb48="SHARPLIGHT_SERVER=$B SHARPLIGHT_STEP_ROWS_PER_SEQUENCE=48"
export CFG_xn48="X=1"
export CFG_xn48off="SHARPLIGHT_STEP_TILE_AWARE=0"
for m in qwen gptoss; do
  CONCS="16 64" ROUNDS=1 ./hold.sh ./serve_ab.sh $OUT/ab "xb32 xb48 xn48 xn48off vllm xn48 xb48 xn48off xb32" $m
done
