#!/bin/bash
. $OUT/env.sh
cd $TREE
exec 9><gpu lock>; flock 9
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0 SHARPLIGHT_TEST_PROVIDER=cuda HOME=~/test-home
nice -n 15 dotnet run --project tests/Sharplight.Cuda.Tests -c Release --no-build -- -class Sharplight.Cuda.Tests.ManySegmentPlacementTests 9>&- > $OUT/tests/Sharplight.Cuda.Tests_ManySegmentPlacementTests.txt 2>&1
echo "Sharplight.Cuda.Tests ManySegmentPlacementTests rc=$? $(grep -oE 'Total: [0-9]+, Errors: [0-9]+, Failed: [0-9]+, Skipped: [0-9]+' $OUT/tests/Sharplight.Cuda.Tests_ManySegmentPlacementTests.txt | tail -1)"
