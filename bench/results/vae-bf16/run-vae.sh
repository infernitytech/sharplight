#!/bin/bash
# VAE decoders under reduced precision, f16 activations (half, the old behaviour) and the measured choice (auto); --check against f32 products.
# Environment: REPO (the repository, built), MODELS (folders of diffusers pipelines), OUT. Under the GPU lock.
set -u
REPO=${REPO:-.}
D=dotnet
B=$REPO/bench/ConvBench/bin/Release/net11.0/ConvBench.dll
OUT=${OUT:-raw}
M=${MODELS:-models}
for prov in ${PROVS:-cuda vulkan}; do
  for spec in ${SPECS:-sd/sdxl-diffusers:1024 flux/FLUX.1-schnell:1024 sd/sd15-diffusers:512}; do
    m=${spec%:*}; size=${spec#*:}
    for mode in ${MODES:-half auto}; do
      timeout 1500 $D $B --provider $prov --vae $M/$m --size $size --runs 5 --check --activations $mode 2>&1 | grep -v "^Compiled\|^  \|^segment\|^\[" > $OUT/$(basename $m)-$size-$prov-$mode.txt
      tail -3 $OUT/$(basename $m)-$size-$prov-$mode.txt
    done
  done
done
