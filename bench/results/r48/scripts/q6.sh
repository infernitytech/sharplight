#!/bin/bash
# Queue 6: repeats. Llama-3.1-8B 16 streams base / branch alternated; gpt-oss-20b and Qwen3-8B-FP8 base / branch; pad on / off (x tree) for gpt-oss
B=~/sl-worktrees/perf-r48-base; X=~/sl-worktrees/perf-r48-x; H=~/models/hf; P=~/sl-worktrees/perf-r48/bench/results/r48/serve
REPO=$B ./serve.sh ab/base2 sharplight llama $H/Llama-3.1-8B-Instruct 16 --provider cuda
SHARPLIGHT_STEP_PROFILE=$P/prof/llama_br2.csv ./serve.sh ab/br2 sharplight llama $H/Llama-3.1-8B-Instruct 16 --provider cuda
REPO=$X SL_AB_PAD=0 SHARPLIGHT_STEP_PROFILE=$P/prof/llama_off2.csv ./serve.sh ab/off2 sharplight llama $H/Llama-3.1-8B-Instruct 16 --provider cuda
SHARPLIGHT_STEP_PROFILE=$P/prof/gptoss_br2.csv ./serve.sh ab/br2 sharplight gptoss $H/gpt-oss-20b 16 --provider cuda
REPO=$B ./serve.sh ab/base2 sharplight gptoss $H/gpt-oss-20b 16 --provider cuda
REPO=$B ./serve.sh ab/base2 sharplight fp8 $H/Qwen3-8B-FP8 16 --provider cuda
SHARPLIGHT_STEP_PROFILE=$P/prof/fp8_br2.csv ./serve.sh ab/br2 sharplight fp8 $H/Qwen3-8B-FP8 16 --provider cuda
REPO=$B ./serve.sh c64/base2 sharplight llama $H/Llama-3.1-8B-Instruct 64 --provider cuda
./serve.sh c64/br2 sharplight llama $H/Llama-3.1-8B-Instruct 64 --provider cuda
echo q6-done
