#!/bin/bash
# usage: sizes.sh <tag> <model dir> <sizes> [extra]   decode-only steps
. <path>
export HOME=~/test-home NUGET_PACKAGES=~/.nuget/packages DOTNET_CLI_HOME=~ SHARPLIGHT_VK_DEVICES=0 CUDA_VISIBLE_DEVICES=0
R=${REPO:-~/sl-worktrees/perf-r48}; OUT=~/sl-worktrees/perf-r48/bench/results/r48/steps; TAG=$1; M=$2; SZ=$3; shift 3
mkdir -p $OUT
flock ~/.sl-bench.lock bash -c "uptime > $OUT/$TAG.gpu; timeout 1500 $R/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider cuda --slots ${SLOTS:-256} --ctx ${CTX:-512} --depth ${DEPTH:-256} --batch 2048 --modes ${MODES:-greedy} --reps ${REPS:-10} --sizes $SZ $* > $OUT/$TAG.txt 2>&1"
grep -E "^\| [0-9]" $OUT/$TAG.txt
