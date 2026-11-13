#!/bin/bash
# usage: load_time_coldfile.sh -> like load_time.sh but the model file is evicted from the page cache before each run (kernel caches warm).
HERE="$(cd "$(dirname "$0")" && pwd)"; . "$HERE/models.sh"
CLI=$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight
t() { local s=$(date +%s.%N); "$@" >/dev/null 2>&1; echo "$(echo "$(date +%s.%N)-$s" | bc)"; }
G27=$HOME/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf; S27=$HOME/models/sharplight/Qwen3.8-27B-Q8_0.slm
M7=$HOME/models/gguf/Mistral-7B-Instruct-v0.3/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
for f in "$G27" "$M7"; do
  echo "## $(basename $f) cold page cache"
  for i in 1 2; do python3 $HERE/evict.py "$f"; echo "sharplight GGUF #$i: $(t $CLI "$f" -p hi -n 1)"; done
  for b in cuda13 vulkan; do for i in 1 2; do python3 $HERE/evict.py "$f"; echo "llama.cpp $b #$i: $(LD_LIBRARY_PATH=$LLAMA_ROOT/runtime.llama-b11177-$b t $LLAMA_ROOT/runtime.llama-b11177-$b/llama-completion -m "$f" -ngl 99 -p hi -n 1 --no-warmup)"; done; done
done
echo "## Qwen3.8-27B .slm cold page cache"
for i in 1 2; do python3 $HERE/evict.py "$S27"; echo "sharplight .slm #$i: $(t $CLI "$S27" -p hi -n 1)"; done
