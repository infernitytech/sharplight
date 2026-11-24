#!/bin/bash
. $OUT/env.sh
cd $OUT
SLOTS=24 REPS=1 MIXED=16+0,15+10,16+16,16+32,16+112,16+240,16+250,16+496,16+512,16+752,16+762,16+784,0+10,0+128 PROF="--profile 1,0" ./steps.sh tail $M8 "new|new|"
export CFG_xd48="SHARPLIGHT_SCHED_TRACE=1 SHARPLIGHT_STEP_ROWS_PER_SEQUENCE=48"
export SHARPLIGHT_SERVER=$BASE/samples/Sharplight.Cli/bin/Release/net11.0/sharplight
CONCS="16 64" ROUNDS=1 ./hold.sh ./serve_ab.sh $OUT/trace48 "xd48" qwen gptoss
