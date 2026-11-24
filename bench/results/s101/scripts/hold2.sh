#!/bin/bash
. $OUT/env.sh
cd $OUT
REPS=2 MIXED=16+0,15+10,16+16,16+32,16+64,16+112,16+240,16+496,64+0,64+16,64+64 ./steps.sh tok $MOSS "tr1|new|X=1" "tr0|new|SHARPLIGHT_MOE_TOKEN_ROWS=0"
REPS=1 MIXED=15+10 PROF="--profile 0" ./steps.sh tokp $MOSS "tr0|new|SHARPLIGHT_MOE_TOKEN_ROWS=0"
