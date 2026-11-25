#!/bin/bash
. <path>
cd $OUT
./tc.sh bug ~/sl-worktrees/perf-r42-bug Qwen3-8B-FP8 $M8 0
./tc.sh base $BASE Qwen3-8B-FP8 $M8 1
./tc.sh base $BASE gpt-oss-20b $MOSS 1
./tc.sh base $BASE Llama-3.1-8B ~/models-local/hf/Llama-3.1-8B-Instruct 1
./tc.sh base $BASE Qwen3-30B-A3B-FP8 ~/models-local/hf/Qwen3-30B-A3B-FP8 1
