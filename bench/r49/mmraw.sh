#!/bin/bash
# usage: mmraw.sh <type> <rows> <shapes> <dual> -- full output, caller holds the lock
export TMPDIR=${TMPDIR:-/tmp}
cd ${SL_WORKTREE:-.}
export SHARPLIGHT_VK_DEVICES=0 CUDA_VISIBLE_DEVICES=0
SHARPLIGHT_BENCH=1 SHARPLIGHT_BENCH_TYPES=$1 SHARPLIGHT_BENCH_ROWS=$2 SHARPLIGHT_BENCH_SHAPES="$3" SHARPLIGHT_BENCH_DUAL=${4:-0} \
  timeout 900 dotnet run --project tests/Sharplight.Vulkan.Tests -c Release --no-build -- -method "Sharplight.Vulkan.Tests.BenchTests.MatMulPrefill" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E "BENCH|FAIL|rror|Register Count|Total:" | sed 's/, Binary.*Shared/ Shared/'
