#!/bin/bash
cd $OUT; . ./env.sh
MIXED="0+512x1,16+256x1" SIZES=16,64 ./steps.sh fit $M8 "base|base|X=1" "rule|exp|X=1" "fit1|exp|SHARPLIGHT_CUDA_PDL_FIT=1" "fit0|exp|SHARPLIGHT_CUDA_PDL_FIT=0"
for m in survey/bert-base-uncased.model.json efficientnet-lite4-11.json survey/phi35-mini-q4f16.json survey/qwen2.5-0.5b.model_q4.json; do
  ./var.sh fit.log cuda 1 $m "base|base|X=1" "rule|exp|X=1" "fit1|exp|SHARPLIGHT_CUDA_PDL_FIT=1" "fit0|exp|SHARPLIGHT_CUDA_PDL_FIT=0"
done
SIZES=16,64 ./steps.sh fit $MOSS "base|base|X=1" "rule|exp|X=1" "fit1|exp|SHARPLIGHT_CUDA_PDL_FIT=1" "fit0|exp|SHARPLIGHT_CUDA_PDL_FIT=0"
