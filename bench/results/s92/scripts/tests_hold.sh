#!/bin/bash
cd $TREE
O=${1:-$OUT/tests}; mkdir -p $O
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0
. $ENV_SH
export NUGET_PACKAGES=${NUGET_PACKAGES:-$NUGET} DOTNET_CLI_HOME=$HOME
for prov in cuda vulkan; do
  for c in Fp8UniformScales Fp8BlockScaled Fp8Gemm SkinnyGemm DualGemm MatMul MatMulEpilogue Moe Fusion Norm NormHalfCopy Rope RowInvariance FullAttention PagedAttention PackedSequence; do
    SHARPLIGHT_TEST_PROVIDER=$prov HOME=$TEST_HOME nice -n 15 dotnet run --project tests/Sharplight.Vulkan.Tests -c Release --no-build -- -class "Sharplight.Vulkan.Tests.${c}Tests" > $O/vk_${prov}_$c.txt 2>&1
    echo "$prov $c: $(grep -o 'Total: [0-9,]*.*Not Run: [0-9]*' $O/vk_${prov}_$c.txt | tail -1)"
  done
done
HOME=$TEST_HOME nice -n 15 dotnet run --project tests/Sharplight.Cuda.Tests -c Release --no-build > $O/cuda.txt 2>&1; echo "Cuda.Tests: $(grep -o 'Total: .*' $O/cuda.txt | tail -1)"
HOME=$TEST_HOME nice -n 15 dotnet run --project tests/Sharplight.Serving.Tests -c Release --no-build > $O/serving.txt 2>&1; echo "Serving.Tests: $(grep -o 'Total: .*' $O/serving.txt | tail -1)"
