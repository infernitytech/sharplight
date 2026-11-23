#!/bin/bash
cd $OUT; . ./env.sh
for m in survey/bert-base-uncased.model.json efficientnet-lite4-11.json survey/phi35-mini-q4f16.json survey/qwen2.5-0.5b.model_q4.json zoo/yolov8n.json mobilenetv2-12.json zoo/convnext-tiny.json resnet50-v1-12.json vit-base-patch16-224.json whisper-tiny-encoder.json segformer-b0.json; do
  ./var.sh rule1.log cuda 1 $m "base|base|X=1" "rule|new|X=1" "any|new|SHARPLIGHT_CUDA_PDL_SHARED=any" "notrig|new|SHARPLIGHT_CUDA_PDL_TRIGGER=0"
done
for m in $M8 $MOSS; do
  MIXED="0+512x1,16+256x1" ./steps.sh rule1 $m "base|base|X=1" "rule|new|X=1" "notrig|new|SHARPLIGHT_CUDA_PDL_TRIGGER=0"
done
