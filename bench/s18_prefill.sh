#!/bin/bash
# Section 18: gated-delta-net prefill of Qwen3.8-27B before / after the chunked delta rule and the token-parallel conv1d.
#   FP8=<dir of Qwen3.8-27B-FP8> Q8=<Qwen3.8-27B-Q8_0.gguf> Q4KM=<Qwen3.8-27B-Q4_K_M.gguf> LLAMA_DIR=<folder with
#   runtime.llama-b11177-*> bench/s18_prefill.sh [prefill|profile|llama]
# "before" runs the same build with the new kernels switched off (SHARPLIGHT_DELTA_CHUNK=0 SHARPLIGHT_CONV1D_PARALLEL=0),
# which is exactly the kernel set of main before this change. One prefill chunk of N rows (--batch N).
SL=samples/Sharplight.Cli/bin/Release/net11.0/sharplight
declare -A M=([fp8]=${FP8:?set FP8} [q8]=${Q8:?set Q8} [q4km]=${Q4KM:?set Q4KM})
declare -A CFG=([before]="SHARPLIGHT_DELTA_CHUNK=0 SHARPLIGHT_CONV1D_PARALLEL=0" [p2]="SHARPLIGHT_DELTA_CHUNK_PARTS=2" [p1]="SHARPLIGHT_DELTA_CHUNK_PARTS=1")
what=${1:-prefill}
others() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -v -E "kwin|chrome|rider" | awk -F, '{print $2 $3}' | tr '\n' ';'; }
case $what in
  prefill)
    for q in fp8 q8 q4km; do for t in 512 2048 8192; do for c in before p2 p1; do
      r=$(env ${CFG[$c]} $SL ${M[$q]} --bench $t --ctx $((t + 256)) --batch $t 2>&1 | grep '^prefill')
      echo "$q pp$t $c | $r | gpu: $(others)"
    done; done; done ;;
  profile)
    for t in 512 2048 8192; do for c in before p2 p1; do
      echo "== fp8 pp$t $c | gpu: $(others)"
      env ${CFG[$c]} SHARPLIGHT_VK_PROFILE=1 $SL ${M[fp8]} --bench $t --ctx $((t + 256)) --batch $t 2>&1 \
        | awk '/prefill T=/{p=1} /decode x/{p=0} p' | grep -E "GPU time|delta|conv1d"
    done; done ;;
  llama)
    for q in q8 q4km; do
      echo "== llama-cuda13 $q | gpu: $(others)"
      LD_LIBRARY_PATH=$LLAMA_DIR/runtime.llama-b11177-cuda13 $LLAMA_DIR/runtime.llama-b11177-cuda13/llama-bench -m ${M[$q]} \
        -p 512,2048,8192 -n 0 -r 3 -ngl 99 -fa 1 2>&1 | grep -E '\| *pp'
    done ;;
esac
