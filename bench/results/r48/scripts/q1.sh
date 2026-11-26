#!/bin/bash
# Queue 1: decode-only A/B, no-lanes steady state, record breakdown (base = develop, br = branch)
B=~/sl-worktrees/perf-r48-base; L=~/models/hf/Llama-3.1-8B-Instruct
REPO=$B DEPTH=128 ./sizes.sh dec_base_llama $L 64,65,80,96,128,160,192,224,255,256 --kv-tokens 60000
DEPTH=128 ./sizes.sh dec_br_llama $L 64,65,80,96,128,160,192,224,255,256 --kv-tokens 60000
REPO=$B SHARPLIGHT_CUDA_CONCURRENT=0 ./steps.sh noconc_base_llama $L "16+0,13+240,12+250,15+200,16+240,14+100"
SL_DBG_RECORD=1 ./steps.sh dbg_br_llama $L "16+0,13+240,12+250,15+200,16+240,10+246,14+100"
echo q1-done
