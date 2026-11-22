#!/bin/bash
. <path>
cd ~/sl-worktrees/perf-r26
echo "== Cuda.Tests $(nice -n 5 dotnet run --project tests/Sharplight.Cuda.Tests -c Release --no-build 2>&1 | grep -E 'Total:|\[FAIL\]' | tail -3)"
<path> ~/sl-worktrees/perf-r26 MatMulEpilogueTests FusionTests RopeTests NormHalfCopyTests NormTests Fp8GemmTests QuantFormatTests RowInvarianceTests MatMulTests SkinnyGemmTests MoeTests MmqTests DualGemmTests SplitHalfGemmTests StreamedExpertsTests FullAttentionTests PagedAttentionTests SparseAttentionTests StatefulTests StateHandoffTests CompressedKvTests
echo "done $(date +%T)"
