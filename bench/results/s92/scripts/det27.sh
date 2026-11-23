#!/bin/bash
. $ENV_SH
O=$OUT/correct/det; mkdir -p $O
P1000="$(cat $OUT/p1000.txt)"
for r in 1 2; do for lab in base new off; do [ $lab = base ] && W=$BASE || W=$TREE
  [ $lab = off ] && export SHARPLIGHT_CUDA_NVRTC_SASS=0 || unset SHARPLIGHT_CUDA_NVRTC_SASS
  timeout 900 dotnet $W/samples/Sharplight.Cli/bin/Release/net11.0/Sharplight.Cli.dll ~/models/hf/Qwen3.8-27B-FP8 --provider cuda -n 120 -p "$P1000" > $O/q27_p1000_${lab}_$r.txt 2>/dev/null
  echo "$lab $r $(md5sum < $O/q27_p1000_${lab}_$r.txt | cut -c1-8)"
done; done
