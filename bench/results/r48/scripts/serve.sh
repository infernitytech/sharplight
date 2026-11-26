#!/bin/bash
# usage: serve.sh <outsub> <engine> <name> <model> <concs> [server args]  (env passes through to the server; REPO picks the checkout)
. <path>
export HOME=~/test-home NUGET_PACKAGES=~/.nuget/packages DOTNET_CLI_HOME=~ SHARPLIGHT_VK_DEVICES=0 CUDA_VISIBLE_DEVICES=0
R=${REPO:-~/sl-worktrees/perf-r48}; SUB=$1; shift
export OUT_DIR=~/sl-worktrees/perf-r48/bench/results/r48/serve/$SUB VLLM_VENV=~/venvs/vllm-r12 QUIET_GPU=1 FIRST=${FIRST:-0}
export SHARPLIGHT_SERVER=$R/samples/Sharplight.Cli/bin/Release/net11.0/sharplight
mkdir -p $OUT_DIR
LOGF=$OUT_DIR/run_$2_$(date +%s).log
flock ~/.sl-bench.lock bash -c "uptime > $LOGF.load; grep throttled_usec /sys/fs/cgroup/cpu.stat > $LOGF.thr0; $R/bench/s16_serve.sh $* > $LOGF 2>&1; rc=\$?; grep throttled_usec /sys/fs/cgroup/cpu.stat > $LOGF.thr1; uptime >> $LOGF.load; exit \$rc"
echo "rc=$?"
