#!/bin/bash
. <path>
export CUDA_VISIBLE_DEVICES=0
W=~/sl-worktrees/perf-r23; B=<path> O=<path> mkdir -p $O
PY=~/venvs/vllm-r12/bin/python
samp() { local name=$1; shift; echo "== $name"; $PY $W/bench/s73_clock_sampler.py --gpu 0 --out $O/$name.csv -- "$@" 2>&1 | grep -v Warning; }
for v in ${CHECK:-}; do echo "== check $v"; $B/$v | grep check | sed 's/  check //' | tr '\n' ' '; echo; done
for round in 1 2; do
 for rn in vllm ours; do
  [ $rn = vllm ] && R=<path> || R=$W/bench/results/s67/harness/routes64_text.txt
  [ $rn = vllm ] && P=12 || P=80
  for v in ${VARIANTS}; do
    for p in gu dn; do samp ${rn}_${p}_$v env ROUTES=$R LAYERS=24 PASSES=$P SUSTAINED_ONLY=1 $B/${p}_$v; done
  done
 done
done
