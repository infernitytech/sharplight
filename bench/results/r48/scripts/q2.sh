#!/bin/bash
# Queue 2: Llama-3.1-8B serving at 16 streams, base vs branch (GEMM threshold), with step profiles
B=~/sl-worktrees/perf-r48-base; H=~/models/hf; P=~/sl-worktrees/perf-r48/bench/results/r48/serve
mkdir -p $P/prof
REPO=$B SHARPLIGHT_STEP_PROFILE=$P/prof/llama_base.csv ./serve.sh gemm/base sharplight llama $H/Llama-3.1-8B-Instruct 16 --provider cuda
SHARPLIGHT_STEP_PROFILE=$P/prof/llama_br.csv ./serve.sh gemm/br sharplight llama $H/Llama-3.1-8B-Instruct 16 --provider cuda
echo q2-done
