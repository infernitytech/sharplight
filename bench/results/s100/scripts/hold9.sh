#!/bin/bash
. /workspace/work/perf-r40/env.sh
cd $OUT
MIX="16+0,16+250,16+368,16+496,16+250x2"
PROVIDER=vulkan SLOTS=20 REPS=1 MIXED=$MIX ./steps.sh vk $M8 "base|base|X=1" "new|new|X=1"
PROVIDER=vulkan SLOTS=20 REPS=1 MIXED=$MIX ./steps.sh vk /root/models-local/hf/Qwen2.5-1.5B-Instruct "base|base|X=1" "new|new|X=1"
