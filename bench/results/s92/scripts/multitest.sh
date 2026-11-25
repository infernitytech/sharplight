#!/bin/bash
. $ENV_SH
cd $TREE
for f in default 0; do
  if [ $f = 0 ]; then export SHARPLIGHT_CUDA_NVRTC_SASS=0; else unset SHARPLIGHT_CUDA_NVRTC_SASS; fi
  export XDG_CACHE_HOME=$WORK/cache_mt_$f
  echo "== SASS $f"
  timeout 1200 dotnet run --project tests/Sharplight.Cuda.Tests -c Release --no-build -- -method "*NativeFp8MultiGemmPromptRows*" 2>&1 | grep -E "FAIL|Total:|y[0-9]\[" | head -12
done
