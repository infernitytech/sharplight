#!/bin/bash
. $OUT/env.sh
cd $OUT
mkdir -p $OUT/prof48
export CFG_xpq="SHARPLIGHT_SCHED_TRACE=1 SHARPLIGHT_STEP_PROFILE=$OUT/prof48/qwen.csv"
export CFG_xpo="SHARPLIGHT_SCHED_TRACE=1 SHARPLIGHT_STEP_PROFILE=$OUT/prof48/gptoss.csv"
CONCS="16 64" ROUNDS=1 ./hold.sh ./serve_ab.sh $OUT/prof48 "xpq" qwen
CONCS="16 64" ROUNDS=1 ./hold.sh ./serve_ab.sh $OUT/prof48 "xpo" gptoss
