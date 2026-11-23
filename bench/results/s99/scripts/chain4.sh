#!/bin/bash
cd $OUT
./moe16.sh > moe16.out 2>&1
./nsys_onnx.sh survey/qwen2.5-0.5b.model_q4.json 1 "entry|X=1" "fill1|SHARPLIGHT_CUDA_PDL_MIN_FILL=1" > nsys2.out 2>&1
./nsys_onnx.sh survey/bert-base-uncased.model.json 1 "entry|X=1" "fill1|SHARPLIGHT_CUDA_PDL_MIN_FILL=1" "notrig|SHARPLIGHT_CUDA_PDL_TRIGGER=0" >> nsys2.out 2>&1
