#!/bin/bash
# rl.sh <qwen|gptoss|llama> <tag> "<mixed entries>" [extra BatchBench args]: one BatchBench run of mixed entries with per-entry profiles,
# a 2 s sustained window per entry and NVML samples (power / clocks), inside the bench lock.
. <path>
m=$1; tag=$2; MIX=$3; shift 3
case $m in qwen) M=$M8; KV=131072 ;; gptoss) M=$MOSS; KV=262144 ;; llama) M=$ML; KV=90112 ;; esac
O=$OUT/roof; mkdir -p $O
n=$(echo $MIX | tr ',' '\n' | wc -l); P=$(seq -s, 0 $((n-1)))
export CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME=$WORK/cache_bb
[ "${V:-base}" = base ] && BB=$BASE/bench/BatchBench/bin/Release/net11.0/BatchBench || BB=$TREE/bench/BatchBench/bin/Release/net11.0/BatchBench
flock ~/.sl-bench.lock $PY $TREE/bench/s73_clock_sampler.py --gpu 0 --interval 0.01 --out $O/${m}_${V:-base}_$tag.csv -- \
  $BB $M --provider cuda --slots ${SLOTS:-264} --ctx ${CTX:-1024} --kv-tokens $KV --depth ${DEPTH:-256} --batch 2048 --modes greedy --reps 10 --text \
  --mixed "$MIX" --profile $P --hold-ms ${HOLD:-2000} "$@" > $O/${m}_${V:-base}_$tag.txt 2> $O/${m}_${V:-base}_$tag.err
echo "$m $tag rc=$?"
