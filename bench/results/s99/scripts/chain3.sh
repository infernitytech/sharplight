#!/bin/bash
cd $OUT; . ./env.sh
while kill -0 1664194 2>/dev/null; do sleep 5; done
for m in survey/bert-base-uncased.model.json efficientnet-lite4-11.json survey/phi35-mini-q4f16.json survey/qwen2.5-0.5b.model_q4.json; do
  ./var.sh trig3.log cuda 1 $m "entry|new|X=1" "notrig|new|SHARPLIGHT_CUDA_PDL_TRIGGER=0" "fill1|new|SHARPLIGHT_CUDA_PDL_MIN_FILL=1" "fill05|new|SHARPLIGHT_CUDA_PDL_MIN_FILL=0.5" "fill2|new|SHARPLIGHT_CUDA_PDL_MIN_FILL=2"
done
for m in $M8 $MOSS; do
  MIXED="0+512x1,16+256x1" ./steps.sh trig3 $m "entry|new|X=1" "fill1|new|SHARPLIGHT_CUDA_PDL_MIN_FILL=1" "fill2|new|SHARPLIGHT_CUDA_PDL_MIN_FILL=2"
done
