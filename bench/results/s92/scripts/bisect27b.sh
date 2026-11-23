#!/bin/bash
. $ENV_SH
D=$WORK/bis; O=$OUT/correct/bisect
T=$TEXT/wikitext2_test.txt
P=$TREE/bench/PplBench/bin/Release/net11.0/PplBench
export SHARPLIGHT_CUDA_CONCURRENT=0
i=10
for f in "0" "if:#define MULTI 2"; do i=$((i+1))
  export SHARPLIGHT_CUDA_NVRTC_SASS="$f" XDG_CACHE_HOME=$WORK/cache_bis_$i
  $P ~/models/hf/Qwen3.8-27B-FP8 --provider cuda --text $T --seq 1024 --chunk 512 --tokens 2048 --dump $D/f$i.f16 > $O/f$i.txt 2>&1
  echo "form $i [$f, concurrent 0]: vs off: $(cmp -s $D/f1.f16 $D/f$i.f16 && echo same || echo DIFFERS) vs f11: $(cmp -s $D/f11.f16 $D/f$i.f16 && echo same || echo DIFFERS) $(grep -o 'PPL [0-9.]*$' $O/f$i.txt | tail -1)"
done
