#!/bin/bash
# usage: run.sh "<K,N,DUAL,TT[,extra -D flags separated by :]> ..."; builds then runs each configuration (build outside a GPU hold with BUILD_ONLY=1).
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT=${BIN_DIR:-$HERE/bin}; mkdir -p "$OUT"
for s in $1; do IFS=, read k n d t ex <<< "$s"; fl=$(echo "$ex" | tr ':' ' '); tag=$(echo "$s" | tr ',:=' '__-')
  b=$OUT/b_$tag
  [ -x "$b" ] || nice -n 15 nvcc -O3 -arch=sm_120 -std=c++17 -DK=$k -DN=$n -DDUAL=$d -DTT=$t $fl "$HERE/bench_dec.cu" -o "$b" || continue
  [ "${BUILD_ONLY:-0}" = 1 ] || "$b"
done
