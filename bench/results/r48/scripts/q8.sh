#!/bin/bash
# Queue 8: Llama-3.1-8B 64-stream repeat, base / branch
B=~/sl-worktrees/perf-r48-base; H=~/models/hf
REPO=$B ./serve.sh c64/base2 sharplight llama $H/Llama-3.1-8B-Instruct 64 --provider cuda
./serve.sh c64/br2 sharplight llama $H/Llama-3.1-8B-Instruct 64 --provider cuda
REPO=$B ./serve.sh ab/base3 sharplight gptoss $H/gpt-oss-20b 16 --provider cuda
./serve.sh ab/br3 sharplight gptoss $H/gpt-oss-20b 16 --provider cuda
P=~/sl-worktrees/perf-r48/bench/results/r48/serve
SHARPLIGHT_STEP_PROFILE=$P/prof/fp8_br3.csv ./serve.sh ab/br3 sharplight fp8 $H/Qwen3-8B-FP8 16 --provider cuda
SHARPLIGHT_STEP_PROFILE=$P/prof/llama_br3.csv ./serve.sh ab/br3 sharplight llama $H/Llama-3.1-8B-Instruct 16 --provider cuda
echo q8-done
