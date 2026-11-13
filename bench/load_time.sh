#!/bin/bash
# usage: load_time.sh  -> wall-clock seconds from process start to the first generated token (load + compile + 1-token prefill + 1 decode).
# Sharplight cold = empty kernel cache (XDG_DATA_HOME) and empty NVIDIA shader cache; warm = default caches; .slm = compiled-model package.
# OS page cache is NOT dropped (no root): files read are mostly page-cache warm; the same applies to every engine.
HERE="$(cd "$(dirname "$0")" && pwd)"; . "$HERE/models.sh"
CLI=$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight
SCR=${SCRATCH:-/tmp}/load_time_$$; mkdir -p $SCR
t() { local s=$(date +%s.%N); "$@" >/dev/null 2>$SCR/err; local rc=$?; echo "$(echo "$(date +%s.%N)-$s" | bc) (rc=$rc)"; }
for e in "$@"; do n=${e%%|*}; p=${e##*|}
  echo "## $n"
  echo -n "sharplight GGUF cold-cache:  "; XDG_DATA_HOME=$SCR/xdg __GL_SHADER_DISK_CACHE_PATH=$SCR/gl SHARPLIGHT_LOAD_TIMING=1 t $CLI "$p" -p hi -n 1; tail -3 $SCR/err | cut -c1-200
  for i in 1 2; do echo -n "sharplight GGUF warm-cache #$i: "; t $CLI "$p" -p hi -n 1; done
  if [ $n != qwen27b-q8 ]; then $CLI "$p" -p hi -n 1 --save $SCR/$n.slm >/dev/null 2>&1; SLM=$SCR/$n.slm; else SLM=$HOME/models/sharplight/Qwen3.8-27B-Q8_0.slm; fi
  for i in 1 2; do echo -n "sharplight .slm package #$i:   "; t $CLI "$SLM" -p hi -n 1; done
  for b in cuda13 vulkan; do for i in 1 2; do echo -n "llama.cpp $b #$i:           "; LD_LIBRARY_PATH=$LLAMA_ROOT/runtime.llama-b11177-$b t $LLAMA_ROOT/runtime.llama-b11177-$b/llama-completion -m "$p" -ngl 99 -p hi -n 1 --no-warmup; done; done
  rm -f $SCR/$n.slm
done
