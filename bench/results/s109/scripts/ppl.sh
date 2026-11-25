#!/bin/bash
# ppl.sh <qwen|llama> <kv type> <chunk> [provider]: PplBench on wikitext-2 (4 x 3,072 tokens) inside the bench lock.
. $(dirname "$0")/env.sh
m=$1; kt=$2; ch=$3; prov=${4:-cuda}
case $m in qwen) M=$M8 ;; llama) M=$ML ;; esac
O=$OUT/ppl; mkdir -p $O
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 XDG_CACHE_HOME=$WORK/cache_bb
flock ~/.sl-bench.lock $TREE/bench/PplBench/bin/Release/net11.0/PplBench $M --provider $prov --text $WORK/wikitext2_test.txt --seq 3072 \
  --chunk $ch --tokens ${TOKENS:-12288} --kv-type $kt > $O/${m}_${kt}_c${ch}_$prov.txt 2>&1
echo "$m $kt c$ch $prov rc=$?"; tail -2 $O/${m}_${kt}_c${ch}_$prov.txt
