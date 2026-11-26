#!/bin/bash
# Queue 10: 64 streams, padding on / off (step profiles), Llama-3.1-8B
X=~/sl-worktrees/perf-r48-x; H=~/models/hf; P=~/sl-worktrees/perf-r48/bench/results/r48/serve
SHARPLIGHT_STEP_PROFILE=$P/prof/c64_br3.csv ./serve.sh c64/br3 sharplight llama $H/Llama-3.1-8B-Instruct 64 --provider cuda
REPO=$X SL_AB_PAD=0 SHARPLIGHT_STEP_PROFILE=$P/prof/c64_off3.csv ./serve.sh c64/off3 sharplight llama $H/Llama-3.1-8B-Instruct 64 --provider cuda
REPO=~/sl-worktrees/perf-r48-base SHARPLIGHT_STEP_PROFILE=$P/prof/c64_base3.csv ./serve.sh c64/base3 sharplight llama $H/Llama-3.1-8B-Instruct 64 --provider cuda
echo q10-done
