#!/bin/bash
# Queue 9: GPU test suites on the branch (CUDA)
. <path>
export HOME=~/test-home NUGET_PACKAGES=~/.nuget/packages DOTNET_CLI_HOME=~ SHARPLIGHT_VK_DEVICES=0 CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_TEST_PROVIDER=cuda
T=~/sl-worktrees/perf-r48/bench/results/r48/tests; mkdir -p $T
cd ~/sl-worktrees/perf-r48
flock ~/.sl-bench.lock bash -c "timeout 3600 dotnet run --project tests/Sharplight.Cuda.Tests -c Release --no-build > $T/cuda.txt 2>&1; timeout 3600 dotnet run --project tests/Sharplight.Serving.Tests -c Release --no-build > $T/serving_cuda.txt 2>&1"
VK_ICD_FILENAMES=/nonexistent.json CUDA_VISIBLE_DEVICES=-1 timeout 3000 nice dotnet run --project tests/Sharplight.Core.Tests -c Release --no-build > $T/core.txt 2>&1
echo q9-done
