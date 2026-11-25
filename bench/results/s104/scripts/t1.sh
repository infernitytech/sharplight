#!/bin/bash
# t1.sh <class...>: Vulkan.Tests classes on CUDA, one per process
. <path>
O=$OUT/tests; mkdir -p $O; cd $TREE
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0 SHARPLIGHT_TEST_PROVIDER=${SHARPLIGHT_TEST_PROVIDER:-cuda} HOME=~/test-home XDG_CACHE_HOME=$WORK/cache_tests
for c in "$@"; do
  nice -n 15 dotnet run --project tests/${SUITE:-Sharplight.Vulkan.Tests} -c Release --no-build -- -class "${NS:-Sharplight.Vulkan.Tests}.$c" 9>&- > $O/${TAG:-t}_$c.txt 2>&1
  echo "$c rc=$? $(grep -oE 'Total: [0-9]+, Errors: [0-9]+, Failed: [0-9]+, Skipped: [0-9]+' $O/${TAG:-t}_$c.txt | tail -1)"
done
