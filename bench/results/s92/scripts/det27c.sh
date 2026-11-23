#!/bin/bash
. $ENV_SH
D=$WORK/bis; O=$OUT/correct/bisect
T=$TEXT/wikitext2_test.txt
P=$TREE/bench/PplBench/bin/Release/net11.0/PplBench
export SHARPLIGHT_CUDA_NVRTC_SASS="if:#define MULTI 2" XDG_CACHE_HOME=$WORK/cache_bis_12 SHARPLIGHT_CUDA_CONCURRENT=0
for r in d e; do
  $P ~/models/hf/Qwen3.8-27B-FP8 --provider cuda --text $T --seq 1024 --chunk 512 --tokens 2048 --dump $D/r$r.f16 > $O/r$r.txt 2>&1
  echo "run $r (MULTI 2 SASS, concurrent 0): vs f12 $(cmp -s $D/f12.f16 $D/r$r.f16 && echo same || echo DIFFERS) vs off $(cmp -s $D/f1.f16 $D/r$r.f16 && echo same || echo DIFFERS) $(grep -o 'PPL [0-9.]*$' $O/r$r.txt | tail -1)"
done
