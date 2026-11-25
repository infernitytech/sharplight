#!/bin/bash
cd $OUT; . ./env.sh
./hold.sh ./serve_hold.sh > serve_hold.log 2>&1
./correct.sh > correct.log 2>&1
./tests.sh > tests.log 2>&1
./onnxgpu.sh $PWD/onnx_gpu_final.jsonl 2 sl-cuda,sl-cuda@base,sl-cuda-fast,sl-cuda-fast@base,ort-cuda,ort-cuda-strict "resnet50" "convnext-tiny" "efficientnet-lite4" "mobilenetv2" "yolov8n" "yolov10n" "ssd-10 (NMS)" "bert-base (16 tok)" "vit-base" "whisper-tiny-encoder" "qwen2.5-0.5b q4 (MatMulNBits)" "phi3.5-mini q4f16 (GQA)" "segformer-b0" "mobilesam-encoder" "raft-large" "fasterrcnn-12 (counts)" "maskrcnn-12 (counts)" > onnx_final.out 2>&1
