#!/bin/bash
# greedy text base vs new (CUDA, GPU 0); usage: greedy.sh <models...>; labels: base, new, new_off (new with --fp8-block-scaled off)
. <env.sh>
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0 XDG_CACHE_HOME=$WORK/cache
O=$OUT/correct; mkdir -p $O
P40="Write a detailed essay about the role of water and power in modern history. Explain the history, give concrete examples and discuss open questions."
P1000="$(cat $OUT/p1000.txt)"
for spec in "$@"; do
  m=${spec%%:*}; path=${spec#*:}; labs="base new"; case $m in *FP8*) labs="base new new_off";; esac
  for lab in $labs; do export XDG_CACHE_HOME=$WORK/cache_bb_${lab%%_*}
    [ $lab = base ] && CLI=$BASE/samples/Sharplight.Cli/bin/Release/net11.0/Sharplight.Cli.dll || CLI=$TREE/samples/Sharplight.Cli/bin/Release/net11.0/Sharplight.Cli.dll
    X=(); [ $lab = new_off ] && X=(--fp8-block-scaled off)
    for pn in p40 p1000; do [ $pn = p40 ] && P="$P40" || P="$P1000"
      timeout 900 dotnet $CLI "$path" --provider ${PROV:-cuda} -n 120 -p "$P" "${X[@]}" > $O/${m}_${PROV:-cuda}_${pn}_$lab.txt 2>$O/${m}_${PROV:-cuda}_${pn}_$lab.err
      echo "$m ${PROV:-cuda} $pn $lab rc=$? $(md5sum < $O/${m}_${PROV:-cuda}_${pn}_$lab.txt | cut -c1-8)"
    done
  done
done
