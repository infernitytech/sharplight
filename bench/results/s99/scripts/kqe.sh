#!/bin/bash
# K-quant decode GEMM: develop's layout vs the sector-parity layout (s97 rejected diff), 3 s energy windows, one hold
exec 9><gpu lock>; flock 9
export CUDA_VISIBLE_DEVICES=0
echo "# hold $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
for r in 1 2; do for f in 4 6; do for shape in 4096_12288 12288_4096; do for bm in 16 32; do for lay in base new; do
  T=$bm; [ $bm = 16 ] && T=9
  printf "r$r %-5s " $lay; ENERGY=3 $WORK/kq/kq_${f}_${shape}_${bm}_$lay $T 9>&- | tail -1
done; done; done; done; done
echo "# release $(date +%T)"
