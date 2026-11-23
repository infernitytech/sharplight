#!/bin/bash
. /workspace/work/perf-r40/env.sh
cd $OUT
export CFG_xw64="SHARPLIGHT_STEP_ROWS_PER_SEQUENCE=64" CFG_xw48="SHARPLIGHT_STEP_ROWS_PER_SEQUENCE=48"
CONCS=16 ROUNDS=2 PER_OVERRIDE=8 ./serve_ab.sh $OUT/serve_pol "new xw64 xw48 wfull vllm" gptoss qwen
