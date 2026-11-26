#!/bin/bash
LK="flock ${SL_BENCH_LOCK:-/tmp/sl-bench.lock}"; [ -n "${NOLOCK:-}" ] && LK=""
# usage: r47-pp.sh <outdir> <tag> <provider> <model> <lens...>  -- Sharplight --bench per length (median of 5 inside) under the bench lock
OUT=$1; TAG=$2; PROV=$3; MODEL=$4; shift 4
CLI=${SL_WORKTREE:-.}/samples/Sharplight.Cli/bin/Release/net11.0/sharplight
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$OUT"
export SHARPLIGHT_VK_DEVICES=0 CUDA_VISIBLE_DEVICES=0
for N in "$@"; do
  $LK bash -c "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > '$OUT/${TAG}_${PROV}_$N.gpu'; '$CLI' '$MODEL' --bench $N --ctx $((N+256)) --provider $PROV \${SL_ARGS:-} 2>&1 | grep -E '^prefill|^decode|rror|GPU' > '$OUT/${TAG}_${PROV}_$N.txt'"
  echo "$TAG $PROV $N: $(head -2 $OUT/${TAG}_${PROV}_$N.txt | tr '\n' ' ')"
done
