#!/bin/bash
# Queue 11: CUDA Prepare (pre-recorded decode shapes): 64 and 16 streams, then the GPU test suites again
H=~/models/hf; P=~/sl-worktrees/perf-r48/bench/results/r48/serve
SHARPLIGHT_STEP_PROFILE=$P/prof/c64_br4.csv ./serve.sh c64/br4 sharplight llama $H/Llama-3.1-8B-Instruct 64 --provider cuda
SHARPLIGHT_STEP_PROFILE=$P/prof/llama_br4.csv ./serve.sh ab/br4 sharplight llama $H/Llama-3.1-8B-Instruct 16 --provider cuda
./q9.sh
echo q11-done
