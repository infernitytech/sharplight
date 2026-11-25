#!/bin/bash
# nsys_llm.sh <model> <tag> "<label>|<tree>|<env>" ... : nsys of BatchBench 16-sequence decode, one hold
. $OUT/env.sh
M=$1; tag=$2; shift 2; n=$(basename $M)
exec 9><gpu lock>; flock 9
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0
echo "# hold $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
for v in "$@"; do IFS='|' read -r label tr envs <<< "$v"
  T=$TREE; [ $tr = base ] && T=$BASE; [ $tr = exp ] && T=$EXP
  f=$OUT/prof/${tag}_${n}_$label; rm -f $f.nsys-rep $f.sqlite
  env $envs XDG_CACHE_HOME=$WORK/cache_bb_$tr nsys profile -t cuda -s none --cpuctxsw=none --cuda-graph-trace=node -o $f $T/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider cuda --slots 64 --ctx 640 --depth 256 --sizes 16 --modes greedy --reps 40 --text 9>&- > $f.txt 2>&1
  grep "^| 16" $f.txt
done
echo "# release $(date +%T)"
