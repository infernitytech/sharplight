#!/bin/bash
# PplBench (wikitext-2 12,288 tokens, 4 x 3,072); usage: ppl.sh <label> <model path> <name> <reps>
. <path>
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0 XDG_CACHE_HOME=$WORK/cache
lab=$1; M=$2; name=$3; n=$4; [ $lab = base ] && W=$BASE || W=$TREE
O=$OUT/correct; D=$WORK/ppl; mkdir -p $O $D
export XDG_CACHE_HOME=$WORK/cache_bb_$lab
for r in $(seq 1 $n); do
  $W/bench/PplBench/bin/Release/net11.0/PplBench $M --provider cuda --text $WORK/wikitext2_test.txt --seq 3072 --chunk ${CHUNK:-1024} --tokens 12288 --dump $D/${name}_${lab}_c${CHUNK:-1024}_$r.f16 > $O/ppl_${name}_${lab}_c${CHUNK:-1024}_$r.txt 2>&1
  echo "ppl $name $lab $r rc=$? $(md5sum < $D/${name}_${lab}_c${CHUNK:-1024}_$r.f16 | cut -c1-8) $(grep -io 'ppl[ =:]*[0-9.]*' $O/ppl_${name}_${lab}_c${CHUNK:-1024}_$r.txt | tail -1)"
done
