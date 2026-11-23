#!/bin/bash
. /workspace/work/perf-r40/env.sh
cd $OUT
for M in $MOSS $M8; do
REPS=2 SIZES=1,16,64 ./steps.sh dec $M "base|base|X=1" "new|new|X=1" "nofloor|new|SHARPLIGHT_CUDA_PDL_SHARED_FLOOR=0"
done
