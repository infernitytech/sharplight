#!/bin/bash
cd $OUT; . ./env.sh
SIZES=1,16,64 MIXED="0+512x1,16+256x1" ./steps.sh fin2 $MOSS "base|base|X=1" "fin2|exp|X=1"
SIZES=1,16,64 MIXED="0+512x1,16+256x1" ./steps.sh fin2 $M8 "base|base|X=1" "fin2|exp|X=1"
for m in survey/bert-base-uncased.model.json survey/phi35-mini-q4f16.json survey/qwen2.5-0.5b.model_q4.json efficientnet-lite4-11.json zoo/convnext-tiny.json resnet50-v1-12.json; do
  ./var.sh fin2.log cuda 1 $m "base|base|X=1" "fin2|exp|X=1"
done
