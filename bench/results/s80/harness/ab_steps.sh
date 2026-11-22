#!/bin/bash
# usage: ab_steps.sh <outdir> <label> <worktree> [extra BatchBench args]; env M, SIZES, ENVV (env assignments)
. <path>
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0
O=$1; l=$2; w=$3; shift 3; mkdir -p $O
M=${M:-~/models-local/hf/gpt-oss-20b}
env $ENVV timeout 900 $w/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider ${PROV:-cuda} --slots 64 --ctx 640 --depth 256 --sizes ${SIZES:-1,4,16,64} --modes greedy --reps ${REPS:-20} "$@" > $O/steps_$l.txt 2>&1
echo "== $l"; grep "^| [0-9]" $O/steps_$l.txt
