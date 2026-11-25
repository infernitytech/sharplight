#!/bin/bash
. $ENV_SH
D=$WORK/bis; rm -rf $D; mkdir -p $D; O=$OUT/correct/bisect; mkdir -p $O
T=$TEXT/wikitext2_test.txt
P=$TREE/bench/PplBench/bin/Release/net11.0/PplBench
i=0
for f in "0" "" "if:#define QOUT 1" "if:#define MULTI 2" "if:#define MULTI 3" "if:#define N 17408\n#define SG" ; do i=$((i+1))
  if [ -z "$f" ]; then unset SHARPLIGHT_CUDA_NVRTC_SASS; else export SHARPLIGHT_CUDA_NVRTC_SASS="$f"; fi
  export XDG_CACHE_HOME=$WORK/cache_bis_$i
  $P ~/models/hf/Qwen3.8-27B-FP8 --provider cuda --text $T --seq 1024 --chunk 512 --tokens 2048 --dump $D/f$i.f16 > $O/f$i.txt 2>&1
  echo "form $i [$f]: $(cmp -s $D/f1.f16 $D/f$i.f16 && echo same-as-off || echo DIFFERS) $(grep -o 'PPL [0-9.]*$' $O/f$i.txt | tail -1)"
done
