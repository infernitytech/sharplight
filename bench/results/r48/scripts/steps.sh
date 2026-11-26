#!/bin/bash
# usage: steps.sh <tag> <model dir> <mixed list> [extra]   (REPO overrides the checkout)
. <path>
export HOME=~/test-home NUGET_PACKAGES=~/.nuget/packages DOTNET_CLI_HOME=~ SHARPLIGHT_VK_DEVICES=0 CUDA_VISIBLE_DEVICES=0
R=${REPO:-~/sl-worktrees/perf-r48}; OUT=~/sl-worktrees/perf-r48/bench/results/r48/steps; TAG=$1; M=$2; MIX=$3; shift 3
mkdir -p $OUT
flock ~/.sl-bench.lock bash -c "uptime > $OUT/$TAG.gpu; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> $OUT/$TAG.gpu; timeout 1500 $R/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider cuda --slots ${SLOTS:-32} --ctx ${CTX:-1280} --depth 256 --batch 2048 --modes greedy --reps 10 --text --mixed $MIX $* > $OUT/$TAG.txt 2>&1"
grep -E "^\||\[first\]|\[run" $OUT/$TAG.txt | tail -60
