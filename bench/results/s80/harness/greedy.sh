#!/bin/bash
# greedy.sh <tree> <label> <outdir> [models...]: greedy 120 tokens, CUDA and Vulkan, prompts p40 and p1000 (inside the lock)
. <path>
T=$1; L=$2; O=$3; shift 3; mkdir -p $O
P40="Write a detailed essay about the role of water and power in modern history. Explain the history, give concrete examples and discuss open questions."
P1000="$(cat <path>)"
MODELS=${@:-~/models-local/hf/gpt-oss-20b ~/models-local/hf/Qwen3-8B-FP8}
for m in $MODELS; do b=$(basename $m .gguf); for prov in ${PROVIDERS:-cuda vulkan}; do for pn in p40 p1000; do
  [ $pn = p40 ] && P="$P40" || P="$P1000"
  timeout 900 nice -n 5 dotnet $T/samples/Sharplight.Cli/bin/Release/net11.0/Sharplight.Cli.dll $m --provider $prov -n 120 -p "$P" > $O/${b}_${prov}_${pn}_$L.txt 2>$O/${b}_${prov}_${pn}_$L.err
  echo "$b $prov $pn $L rc=$? $(md5sum < $O/${b}_${prov}_${pn}_$L.txt | cut -c1-8)"
done; done; done
