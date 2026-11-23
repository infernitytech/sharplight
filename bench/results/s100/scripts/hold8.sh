#!/bin/bash
. /workspace/work/perf-r40/env.sh
cd $OUT
./correct.sh > correct.log 2>&1
MIX="16+0,16+250,16+368,16+496,16+250x2,0+512,16+256"
for M in $MOSS; do REPS=2 SIZES=1,16,64 MIXED=$MIX ./steps.sh fin $M "base|base|X=1" "new|new|X=1"; done
SLOTS=20 REPS=2 SIZES=1,16 MIXED=$MIX ./steps.sh fin $M8 "base|base|X=1" "new|new|X=1"
