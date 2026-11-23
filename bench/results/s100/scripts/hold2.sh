#!/bin/bash
. /workspace/work/perf-r40/env.sh
cd $OUT
./pdl_sweep.sh > pdl_sweep.log 2>&1
MIX="16+0,16+128,16+250,16+256,16+368,16+496,0+250,0+368,64+192,64+448,32+250,16+250x2"
REPS=2 MIXED=$MIX ./steps.sh mr $MOSS "base|base|X=1" "mr1024|new|SHARPLIGHT_MULTI_GEMM_MAX_ROWS=1024"
SLOTS=20 REPS=1 MIXED=$MIX ./steps.sh mr $M8 "base|base|X=1" "mr1024|new|SHARPLIGHT_MULTI_GEMM_MAX_ROWS=1024"
