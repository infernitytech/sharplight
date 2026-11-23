#!/bin/bash
. /workspace/work/perf-r40/env.sh
cd $OUT
CONCS=16 ROUNDS=1 ./serve_ab.sh $OUT/serve_trace "strace vtrace" gptoss qwen
