#!/bin/bash
# usage: hold.sh <cmd...>  -- runs cmd inside the GPU lock on GPU 0
exec 9><gpu-lock>
flock 9
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0
echo "[hold $(date +%T)] $*" >&2
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader >&2
"$@" 9>&-
rc=$?
echo "[release $(date +%T)]" >&2
exit $rc
