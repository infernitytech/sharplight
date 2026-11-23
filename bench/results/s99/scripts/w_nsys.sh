#!/bin/bash
cd $OUT
while kill -0 1742504 2>/dev/null; do sleep 5; done
./nsys_llm.sh ~/models/hf/Qwen3-8B-FP8 n16 "base|base|X=1" "carve|exp|X=1"
