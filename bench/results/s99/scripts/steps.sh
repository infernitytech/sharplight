#!/bin/bash
# steps.sh <tag> <model path> "<label>|<tree base|new>|<env>" ...  : decode steps 1/16/64 + mixed prompt steps, REPS alternations, one hold
. $OUT/env.sh
tag=$1; M=$2; shift 2; n=$(basename $M)
mkdir -p $OUT/steps
exec 9><gpu lock>; flock 9
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0
echo "# hold $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
for i in $(seq ${REPS:-2}); do for v in "$@"; do
  IFS='|' read -r label tr envs <<< "$v"
  T=$TREE; [ $tr = base ] && T=$BASE; [ $tr = exp ] && T=$EXP
  export XDG_CACHE_HOME=$WORK/cache_bb_$tr
  BB=$T/bench/BatchBench/bin/Release/net11.0/BatchBench
  f=$OUT/steps/${tag}_${n}_${label}_$i
  env $envs $BB $M --provider cuda --slots 64 --ctx 640 --depth 256 --sizes ${SIZES:-1,16,64} --modes greedy --reps 30 --text 9>&- > $f.txt 2>&1
  [ -n "$MIXED" ] && env $envs $BB $M --provider cuda --slots 64 --ctx 1024 --depth 256 --mixed $MIXED --modes greedy --reps 20 --text 9>&- > $f.mixed.txt 2>&1
  echo "== $n $label $i"; grep -E "^\| [0-9]" $f.txt | cut -c1-80; [ -n "$MIXED" ] && grep -E "^\| [0-9]" $f.mixed.txt | cut -c1-120
done; done
echo "# release $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
