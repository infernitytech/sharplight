#!/bin/bash
cd $OUT; . ./env.sh
SIZES=1,16,64 MIXED="0+512x1,16+256x1" ./steps.sh pro $M8 "base|base|X=1" "pro|exp|X=1"
./nsys_llm.sh $MOSS n16 "base|base|X=1" "pro|exp|X=1"
SIZES=1,16,64 MIXED="0+512x1,16+256x1" ./steps.sh pro $MOSS "base|base|X=1" "pro|exp|X=1"
for m in survey/bert-base-uncased.model.json survey/phi35-mini-q4f16.json survey/qwen2.5-0.5b.model_q4.json efficientnet-lite4-11.json; do
  ./var.sh pro.log cuda 1 $m "base|base|X=1" "pro|exp|X=1"
done
