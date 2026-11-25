#!/bin/bash
# tests.sh: Cuda / Serving / Import.Onnx tests on CUDA, one class per process, inside one GPU hold
. /workspace/work/perf-r40/env.sh
O=$OUT/tests; mkdir -p $O; cd $TREE
exec 9>$VERIFY_LOCK; flock 9
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0 SHARPLIGHT_TEST_PROVIDER=cuda HOME=$TEST_HOME
echo "# hold $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
for p in Sharplight.Cuda.Tests Sharplight.Serving.Tests Sharplight.Import.Onnx.Tests; do
  for f in $(grep -rlE "\[(Fact|Theory)" tests/$p --include=*.cs); do
    ns=$(grep -m1 -oP '^namespace \K[\w.]+' $f); for c in $(grep -oP '(public|internal) (sealed |static |partial )*class \K\w+' $f); do
      nice -n 15 dotnet run --project tests/$p -c Release --no-build -- -class "$ns.$c" 9>&- > $O/${p}_$c.txt 2>&1
      echo "$p $c rc=$? $(grep -oE 'Total: [0-9]+, Errors: [0-9]+, Failed: [0-9]+, Skipped: [0-9]+' $O/${p}_$c.txt | tail -1)"
    done
  done
done
for prov in cuda vulkan; do for c in FusionTests MatMulEpilogueTests NormTests NormHalfCopyTests MoeTests Fp8GemmTests F16RangeTests; do
  SHARPLIGHT_TEST_PROVIDER=$prov nice -n 15 dotnet run --project tests/Sharplight.Vulkan.Tests -c Release --no-build -- -class "Sharplight.Vulkan.Tests.$c" 9>&- > $O/Vulkan.Tests_${prov}_$c.txt 2>&1
  echo "Sharplight.Vulkan.Tests[$prov] $c rc=$? $(grep -oE 'Total: [0-9]+, Errors: [0-9]+, Failed: [0-9]+, Skipped: [0-9]+' $O/Vulkan.Tests_${prov}_$c.txt | tail -1)"
done; done
echo "# release $(date +%T)"
