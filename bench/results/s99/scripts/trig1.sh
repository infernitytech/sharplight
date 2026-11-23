#!/bin/bash
cd $OUT
for m in survey/bert-base-uncased.model.json efficientnet-lite4-11.json survey/phi35-mini-q4f16.json survey/qwen2.5-0.5b.model_q4.json zoo/yolov8n.json mobilenetv2-12.json zoo/convnext-tiny.json; do
  ./var.sh trig1.log cuda 1 $m "base|base|X=1" "entry|new|X=1" "notrig|new|SHARPLIGHT_CUDA_PDL_TRIGGER=0" "store|new|SHARPLIGHT_CUDA_PDL_TRIGGER=store"
done
