#!/bin/bash
# Queue 4: serving with step shape buckets (experiment tree), on / off, Llama-3.1-8B and gpt-oss-20b at 16 streams
X=~/sl-worktrees/perf-r48-x; H=~/models/hf; P=~/sl-worktrees/perf-r48/bench/results/r48/serve
REPO=$X SL_AB_PAD=1 SHARPLIGHT_STEP_PROFILE=$P/prof/llama_pad1.csv ./serve.sh pad/on1 sharplight llama $H/Llama-3.1-8B-Instruct 16 --provider cuda
REPO=$X SL_AB_PAD=0 SHARPLIGHT_STEP_PROFILE=$P/prof/llama_pad0.csv ./serve.sh pad/off1 sharplight llama $H/Llama-3.1-8B-Instruct 16 --provider cuda
REPO=$X SL_AB_PAD=1 SHARPLIGHT_STEP_PROFILE=$P/prof/gptoss_pad1.csv ./serve.sh pad/on1 sharplight gptoss $H/gpt-oss-20b 16 --provider cuda
REPO=$X SL_AB_PAD=0 SHARPLIGHT_STEP_PROFILE=$P/prof/gptoss_pad0.csv ./serve.sh pad/off1 sharplight gptoss $H/gpt-oss-20b 16 --provider cuda
echo q4-done
