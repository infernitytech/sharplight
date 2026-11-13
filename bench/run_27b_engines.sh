#!/bin/bash
# Qwen3.8-27B decode on Sharplight (Vulkan, CPU) and llama.cpp (CUDA, Vulkan, CPU); docs/benchmarks.md section 15.
# LLAMA_DIR: folder with the runtime.llama-<build>-{cuda13,vulkan,cpu} builds; Q4KM / Q8: model paths.
cd "$(dirname "$0")/.."
SL="dotnet samples/Sharplight.Cli/bin/Release/net11.0/sharplight.dll"
L=${LLAMA_DIR:?set LLAMA_DIR}
declare -A M=([q4km]=${Q4KM:?set Q4KM} [q8]=${Q8:?set Q8})
for q in q4km q8; do m=${M[$q]}
  for r in 1 2 3; do echo "== sl-vulkan $q run $r"; $SL $m --bench 16 --ctx 512 2>&1 | grep '^prefill'; done
  for b in cuda13 vulkan; do echo "== llama-$b $q"; LD_LIBRARY_PATH=$L/runtime.llama-b11177-$b $L/runtime.llama-b11177-$b/llama-bench -m $m -p 0 -n 128 -r 3 -ngl 99 -fa 1 2>&1 | grep -E '\| *tg'; done
  for r in 1 2 3; do echo "== sl-cpu $q run $r"; taskset -c 0-15 $SL $m --cpu --bench 16 --ctx 512 2>&1 | grep '^prefill'; done
  echo "== llama-cpu $q"; LD_LIBRARY_PATH=$L/runtime.llama-b11177-cpu taskset -c 0-15 $L/runtime.llama-b11177-cpu/llama-bench -m $m -p 0 -n 64 -r 3 -t 16 -fa 1 2>&1 | grep -E '\| *tg'
done
echo DONE
