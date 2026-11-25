#!/bin/bash
exec 9><gpu lock>; flock 9
export CUDA_VISIBLE_DEVICES=0
echo "# hold $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
for cfg in "108 32 0 256 0" "108 32 4224 256 0" "108 32 4224 0 40" "2 32 4224 0 40" "108 32 0 0 40" "16 192 11264 64 0" "16 192 11264 0 8"; do set -- $cfg
  for m in 0 1 2; do NA=$1 NB=$2 B_SMEM=$3 MEM_KB=$4 A_US=3 B_US=$5 MODE=$m $WORK/pdl/pdl 9>&- | sed "s/^/smem=$3 memkb=$4 /"; done
done
echo "# release $(date +%T)"
