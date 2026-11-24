#!/bin/bash
# additional model: Llama-3.1-8B-Instruct (bf16 dense): step costs at a tile boundary, closed and open loop develop-48 vs r41-48
. $OUT/env.sh
cd $OUT
L=$MODELS/hf/Llama-3.1-8B-Instruct
SLOTS=24 REPS=1 MIXED=16+0,15+10,16+112,16+240,16+250,16+496,16+512 ./steps.sh tail $L "new|new|X=1"
B=$BASE/samples/Sharplight.Cli/bin/Release/net11.0/sharplight
export CFG_xb48="SHARPLIGHT_SERVER=$B SHARPLIGHT_STEP_ROWS_PER_SEQUENCE=48"
export CFG_xn48="X=1"
export CFG_xn48off="SHARPLIGHT_STEP_TILE_AWARE=0"
CONCS="16 64" ROUNDS=1 ./hold.sh ./serve_ab2.sh $OUT/abl "xb48 xn48 xn48off xn48 xb48" llama
ARRIVAL=poisson CONCS="16 64" ROUNDS=1 ./hold.sh ./serve_ab2.sh $OUT/ablp "xb48 xn48 xn48off xn48 xb48" llama
