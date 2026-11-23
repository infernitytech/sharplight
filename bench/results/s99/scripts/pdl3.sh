#!/bin/bash
exec 9><gpu lock>; flock 9
export CUDA_VISIBLE_DEVICES=0
echo "# hold $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
for cfg in "0 4224" "4224 4224" "1024 4224" "16384 4224" "4224 0" "0 1024" "0 16384"; do set -- $cfg
  for m in 0 1 2; do NA=108 NB=32 A_SMEM=$1 B_SMEM=$2 A_US=3 B_US=40 MODE=$m $WORK/pdl/pdl 9>&- | sed "s/^/A_SMEM=$1 B_SMEM=$2 /"; done
done
echo "# release $(date +%T)"
