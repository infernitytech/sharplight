#!/bin/bash
. $ENV_SH
D=$WORK/bis; O=$OUT/correct/bisect
T=$TEXT/wikitext2_test.txt
P=$TREE/bench/PplBench/bin/Release/net11.0/PplBench
export SHARPLIGHT_FP8_WG=0
i=20
for f in "0" "" ""; do i=$((i+1))
  if [ -z "$f" ]; then unset SHARPLIGHT_CUDA_NVRTC_SASS; else export SHARPLIGHT_CUDA_NVRTC_SASS="$f"; fi
  export XDG_CACHE_HOME=$WORK/cache_wg0_${f:-sass}
  $P ~/models/hf/Qwen3.8-27B-FP8 --provider cuda --text $T --seq 1024 --chunk 512 --tokens 2048 --dump $D/w$i.f16 > $O/w$i.txt 2>&1
  echo "WG=0 form $i [sass=${f:-default}]: vs w21 $(cmp -s $D/w21.f16 $D/w$i.f16 && echo same || echo DIFFERS) vs off(WG on) $(cmp -s $D/f1.f16 $D/w$i.f16 && echo same || echo DIFFERS) $(grep -o 'PPL [0-9.]*$' $O/w$i.txt | tail -1)"
done
