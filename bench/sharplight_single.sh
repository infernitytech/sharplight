#!/bin/bash
# usage: sharplight_single.sh <name> <path> [reps=3] [extra CLI args...]  -> results/single_sharplight_<name>.txt
# --bench N = warm prefill of N tokens (best of 4) + 64 decode steps at context depth N. N=16 approximates "tg128 at depth 0".
HERE="$(cd "$(dirname "$0")" && pwd)"; NAME=$1; MP=$2; REPS=${3:-3}; shift 3
OUT=$HERE/results/single_sharplight_$NAME.txt; : > $OUT
CLI=$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight
for N in 16 512 2048 4096; do for r in $(seq $REPS); do
  "$HERE/gpu_check.sh" | head -1 | sed "s/^/[N=$N run=$r] /" >> $OUT
  $CLI "$MP" --bench $N --ctx $((N+256)) --batch ${BATCH:-512} "$@" 2>/dev/null | grep '^prefill' | sed "s/^/[N=$N run=$r] /" >> $OUT
done; done
