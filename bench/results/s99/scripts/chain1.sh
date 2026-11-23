#!/bin/bash
cd $OUT; . ./env.sh
while kill -0 1642199 2>/dev/null; do sleep 5; done
./re.sh > re1.log 2>&1
for m in $M8 $MOSS; do
  MIXED="0+512x2,16+256x1" ./steps.sh trig $m "entry|new|X=1" "notrig|new|SHARPLIGHT_CUDA_PDL_TRIGGER=0" "store|new|SHARPLIGHT_CUDA_PDL_TRIGGER=store"
done
