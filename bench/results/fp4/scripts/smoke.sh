#!/bin/bash
source /workspace/tools/env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME=/workspace/work/fp4/cache
CLI=<tree>/samples/Sharplight.Cli/bin/Release/net11.0/sharplight
M=${1:-/workspace/work/fp4/slm/nvidia_Qwen3-8B-FP4.slm}
O=/workspace/work/fp4/smoke
for v in 1 0; do
  SHARPLIGHT_NATIVE_FP4=$v timeout 900 $CLI $M --provider cuda --chat -p "Explain in three sentences why the sky is blue." -n 96 --temp 0 > $O/$(basename $M)_fp4$v.txt 2>&1
  echo "fp4=$v rc=$?"
done
