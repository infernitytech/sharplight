#!/bin/bash
# step costs of mixed decode + prompt steps (gpt-oss-20b, Qwen3-8B-FP8), then per-kernel profiles of 16+0 / 0+256 / 16+256
. /workspace/work/perf-r40/env.sh
cd $OUT
MIX="16+0,0+128,0+256,16+128,16+256,0+512,16+496,16+1008,0+1024,16+2032,0+2048,0+250x2,16+250x2,0+250x4,16+250x4,16+250x8"
for M in $MOSS $M8; do REPS=2 MIXED=$MIX ./steps.sh mix $M "new|new|X=1"; done
for M in $MOSS $M8; do REPS=1 MIXED=16+0,0+256,16+256,0+512 PROF="--profile 0,1,2,3 --profile-top 30" ./steps.sh prof $M "new|new|X=1"; done
