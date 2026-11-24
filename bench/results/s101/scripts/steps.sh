#!/bin/bash
# steps.sh <tag> <model path> "<label>|<tree base|new>|<env>" ... : mixed prompt steps (MIXED), REPS alternations, one hold
. $OUT/env.sh
tag=$1; M=$2; shift 2; n=$(basename $M)
mkdir -p $OUT/steps
exec 9>$VERIFY_LOCK; flock 9
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0
echo "# hold $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
for i in $(seq ${REPS:-2}); do for v in "$@"; do
  IFS='|' read -r label tr envs <<< "$v"
  T=$TREE; [ $tr = base ] && T=$BASE
  export XDG_CACHE_HOME=$WORK/cache_bb_$tr
  BB=$T/bench/BatchBench/bin/Release/net11.0/BatchBench
  f=$OUT/steps/${tag}_${n}_${label}_$i
  [ -n "$SIZES" ] && { env $envs $BB $M --provider ${PROVIDER:-cuda} --slots 64 --ctx 640 --depth 256 --sizes $SIZES --modes greedy --reps 30 --text 9>&- > $f.txt 2>&1; }
  [ -n "$MIXED" ] && env $envs $BB $M --provider ${PROVIDER:-cuda} --slots ${SLOTS:-64} --ctx 2560 --depth 256 --batch ${BATCH:-2048} --mixed $MIXED --modes greedy --reps ${MREPS:-10} --text $PROF 9>&- > $f.mixed.txt 2>&1
  echo "== $n $label $i"; [ -n "$SIZES" ] && grep -E "^\| [0-9]" $f.txt | cut -c1-80; [ -n "$MIXED" ] && grep -E "^\| [0-9]" $f.mixed.txt | cut -c1-120
done; done
echo "# release $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
