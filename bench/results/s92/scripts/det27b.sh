#!/bin/bash
. $ENV_SH
D=$WORK/bis; O=$OUT/correct/bisect
T=$TEXT/wikitext2_test.txt
P=$TREE/bench/PplBench/bin/Release/net11.0/PplBench
export SHARPLIGHT_CUDA_NVRTC_SASS="if:#define MULTI 2" XDG_CACHE_HOME=$WORK/cache_bis_4
for r in a b; do
  $P ~/models/hf/Qwen3.8-27B-FP8 --provider cuda --text $T --seq 1024 --chunk 512 --tokens 2048 --dump $D/r$r.f16 > $O/r$r.txt 2>&1
  echo "run $r: vs f4 $(cmp -s $D/f4.f16 $D/r$r.f16 && echo same || echo DIFFERS) $(grep -o 'PPL [0-9.]*$' $O/r$r.txt | tail -1)"
done
unset SHARPLIGHT_CUDA_NVRTC_SASS; export XDG_CACHE_HOME=$WORK/cache_bis_2
$P ~/models/hf/Qwen3.8-27B-FP8 --provider cuda --text $T --seq 1024 --chunk 512 --tokens 2048 --dump $D/rc.f16 > $O/rc.txt 2>&1
echo "run c (default SASS): vs f2 $(cmp -s $D/f2.f16 $D/rc.f16 && echo same || echo DIFFERS) $(grep -o 'PPL [0-9.]*$' $O/rc.txt | tail -1)"
