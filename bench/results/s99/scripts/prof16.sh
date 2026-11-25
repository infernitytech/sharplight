#!/bin/bash
# per-kernel decode-step profiles at 16 sequences: vLLM (torch profiler) and ours (nsys graph nodes); one hold
. $OUT/env.sh
M=${M:-$MOSS}; n=$(basename $M)
exec 9><gpu lock>; flock 9
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0
echo "# hold $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
cd $TREE
[ -z "$SKIP_VLLM" ] && $PY bench/s54_vllm_decode_profile.py $M --seqs 16 --words --skip 5 --gpu-util 0.6 --top 60 9>&- > $OUT/prof/vllm16_$n.txt 2>$OUT/prof/vllm16_$n.err
for v in ${VARIANTS:-entry}; do
  case $v in entry) E=X=1;; notrig) E=SHARPLIGHT_CUDA_PDL_TRIGGER=0;; store) E=SHARPLIGHT_CUDA_PDL_TRIGGER=store;; *) E=$v;; esac
  rm -f $OUT/prof/sl16_${n}_$v.nsys-rep $OUT/prof/sl16_${n}_$v.sqlite
  env $E XDG_CACHE_HOME=$WORK/cache_bb_new nsys profile -t cuda -s none --cpuctxsw=none --cuda-graph-trace=node -o $OUT/prof/sl16_${n}_$v $TREE/bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider cuda --slots 64 --ctx 640 --depth 256 --sizes 16 --modes greedy --reps 40 --text 9>&- > $OUT/prof/sl16_${n}_$v.txt 2>&1
done
echo "# release $(date +%T)"
