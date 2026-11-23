#!/bin/bash
. /workspace/work/perf-r40/env.sh
cd $OUT
MIX="16+0,16+250,16+368,16+496,16+250x2"
PROVIDER=vulkan SLOTS=20 REPS=3 MIXED=$MIX ./steps.sh vk2 $MODELS/hf/Qwen2.5-1.5B-Instruct "base|base|X=1" "new|new|X=1" "mr256|new|SHARPLIGHT_MULTI_GEMM_MAX_ROWS=256"
