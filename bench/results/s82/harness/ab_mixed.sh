#!/bin/bash
# usage: ab_mixed.sh <outdir> label:worktree ... ; env MIXED (specs), PROF (profile entries), M (model), EXTRA
. <env.sh>
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0
O=$1; shift; mkdir -p $O
M=${M:-~/models-local/hf/gpt-oss-20b}
for t in "$@"; do IFS=: read l w <<< "$t"
  timeout 900 $w/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider ${PROV:-cuda} --slots 72 --ctx 640 --depth 256 --batch 2048 --modes greedy --reps ${REPS:-10} \
    --mixed "${MIXED:-64+0,8+250x8,40+250x7,0+256x8}" ${PROF:+--profile $PROF --profile-top 40} $EXTRA > $O/mixed_$l.txt 2>&1
  echo "== $l"; grep "^| [0-9]" $O/mixed_$l.txt
done
