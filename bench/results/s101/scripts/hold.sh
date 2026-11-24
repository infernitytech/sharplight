#!/bin/bash
# usage: hold.sh <cmd...>  -- runs cmd inside the GPU lock on GPU 0; prints the GPU's processes before and after
exec 9>$VERIFY_LOCK
flock 9
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0
echo "[hold $(date +%T)] $*" >&2
procs() { nvidia-smi | awk '/Processes:/{p=1} p && /[0-9]+ +[CG] /{print "  gpu-proc:", $5, $6, $7}' >&2; }
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader >&2; procs
"$@" 9>&-
rc=$?
procs
echo "[release $(date +%T)]" >&2
exit $rc
