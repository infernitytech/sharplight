#!/bin/bash
# multi-output q/k/v GEMM rows: Qwen2.5-1.5B (bf16, biased q/k/v) mixed steps; then ONNX for the dependent-launch floor
. /workspace/work/perf-r40/env.sh
cd $OUT
Q25=$MODELS/hf/Qwen2.5-1.5B-Instruct
MIX="16+0,16+128,16+250,16+256,16+368,16+496,0+250,0+368,16+250x2"
SLOTS=32 REPS=2 MIXED=$MIX ./steps.sh mr $Q25 "base|base|X=1" "mr1024|new|SHARPLIGHT_MULTI_GEMM_MAX_ROWS=1024"
./onnxgpu.sh $OUT/onnx_floor.jsonl 2 sl-cuda,sl-cuda@base "bert-base (16 tok)" "efficientnet-lite4" "phi3.5-mini q4f16 (GQA)" "qwen2.5-0.5b q4 (MatMulNBits)" "resnet50" "mobilenetv2" "yolov10n" "whisper-tiny-encoder"
