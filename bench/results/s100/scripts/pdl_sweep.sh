#!/bin/bash
# placement threshold: B's shared memory after A with no shared memory, A at 1-4 blocks per SM
exec 9>/root/.sl-verify.lock; flock 9
export CUDA_VISIBLE_DEVICES=0
cd /root/perf-r40-work/pdl
for na in 108 180 328; do for bs in 0 256 512 1024 1536 2048 3072 4096 6144; do
  printf "NA=%s B_SMEM=%s: " $na $bs; NA=$na NB=16 A_SMEM=0 B_SMEM=$bs A_US=3 B_US=40 MODE=1 ./pdl 9>&- | tr '\n' ' ' | cut -c1-220; echo
done; done
