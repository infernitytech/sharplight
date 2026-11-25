. <dev-env.sh>
export TREE=~/sl-worktrees/perf-r35 BASE=~/sl-worktrees/perf-r35-base OUT=$OUT WORK=$WORK
export TEXT=$WORK
export M27=~/models/hf/Qwen3.8-27B-FP8 M8=~/models/hf/Qwen3-8B-FP8 MOSS=~/models/hf/gpt-oss-20b
export PPL=$TREE/bench/PplBench/bin/Release/net11.0/PplBench PPLB=$BASE/bench/PplBench/bin/Release/net11.0/PplBench
mkdir -p $WORK
