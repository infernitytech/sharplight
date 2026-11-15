#!/bin/sh
# Builds libeetq_pre.so: EETQ's own int8 quantizer / weight preprocessing (csrc/cutlass_kernels/cutlass_preprocessors.cc of
# https://github.com/NetEase-FuXi/EETQ, Apache-2.0) with stand-ins for the CUDA / CUTLASS headers (the Sm75 / Sm80 int8 layout:
# ColumnMajorTileInterleave<64, 2>). Needs g++ >= 12. usage: build.sh <EETQ checkout> [output dir]
set -e
here=$(cd "$(dirname "$0")" && pwd)
src="$1/csrc/cutlass_kernels"
out="${2:-$here}"
g++ -O2 -std=c++17 -shared -fPIC -I"$here/shim" -I"$src" -o "$out/libeetq_pre.so" "$here/harness.cc" "$src/cutlass_preprocessors.cc"
