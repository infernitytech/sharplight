#!/bin/bash
# bb.sh <qwen|llama> <kv type> <tag> "<mixed entries>": one BatchBench run (greedy, text printed, per-entry profiles) inside the bench lock.
. $(dirname "$0")/env.sh
m=$1; kt=$2; tag=$3; MIX=$4; shift 4
case $m in qwen) M=$M8; KV=131072 ;; llama) M=$ML; KV=90112 ;; esac
O=$OUT/bb; mkdir -p $O
n=$(echo $MIX | tr ',' '\n' | wc -l); P=$(seq -s, 0 $((n-1)))
export CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME=$WORK/cache_bb
flock ~/.sl-bench.lock $TREE/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider ${PROV:-cuda} --slots 264 --ctx ${CTX:-1024} --kv-tokens $KV \
  --kv-type $kt --depth ${DEPTH:-256} --batch 2048 --modes greedy --reps 10 --text --mixed "$MIX" --profile $P "$@" > $O/${m}_${kt}_$tag.txt 2> $O/${m}_${kt}_$tag.err
echo "$m $kt $tag rc=$?"
