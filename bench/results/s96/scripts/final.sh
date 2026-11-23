#!/bin/bash
# final hold: gpt-oss decode steps under the clock sampler, then serving ours / vLLM / ours for both models
. <env.sh>
mkdir -p $OUT/steps $OUT/serve
BB=$TREE/bench/BatchBench/bin/Release/net11.0/BatchBench
export XDG_CACHE_HOME=$WORK/cache_bb
$PY $TREE/bench/s73_clock_sampler.py --gpu 0 --interval 0.005 --out $OUT/steps/gptoss_text.csv -- $BB $MOSS --provider cuda --slots 64 --ctx 640 --depth 256 --sizes 16,64 --modes greedy --reps 40 --text > $OUT/steps/gptoss_text.txt 2>&1
grep -E "sampler|rows|ms" $OUT/steps/gptoss_text.txt | tail -8
ROUNDS=2 ./serve_ab.sh $OUT/serve "new vllm new" gptoss qwen
