#!/bin/bash
X=~/sl-worktrees/perf-r48-x; L=~/models/hf/Llama-3.1-8B-Instruct
REPO=$X SL_DBG_RECORD=1 ./steps.sh diff_llama $L "16+240,10+246,12+244,8+240x2,16+0,15+0,16+224,16+232"
echo q3-done
