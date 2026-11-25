#!/bin/bash
. $OUT/env.sh
cd $OUT
export CFG_xt="SHARPLIGHT_SCHED_TRACE=1"
CONCS="16 64" ROUNDS=1 ./hold.sh ./serve_ab.sh $OUT/tr41 "xt" qwen gptoss
ARRIVAL=poisson CONCS="16 64" ROUNDS=1 ./hold.sh ./serve_ab.sh $OUT/tr41p "xt" qwen gptoss
