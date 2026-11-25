#!/bin/bash
cd $OUT; . ./env.sh
until grep -q release kqe1.log; do sleep 5; done
for m in $M8 $MOSS; do
  MIXED="0+512x1,16+256x1" ./steps.sh trig2 $m "entry|new|X=1" "notrig|new|SHARPLIGHT_CUDA_PDL_TRIGGER=0" "store|new|SHARPLIGHT_CUDA_PDL_TRIGGER=store"
done
./nsys_onnx.sh survey/phi35-mini-q4f16.json 1 "entry|X=1" "notrig|SHARPLIGHT_CUDA_PDL_TRIGGER=0" "store|SHARPLIGHT_CUDA_PDL_TRIGGER=store"
rm -rf $WORK/dump; mkdir -p $WORK/dump
SHARPLIGHT_CUDA_DUMP_SOURCE=$WORK/dump XDG_CACHE_HOME=$WORK/cache_dump REPS=1 ./var.sh dump.log cuda 1 survey/phi35-mini-q4f16.json "store|new|SHARPLIGHT_CUDA_PDL_TRIGGER=store"
