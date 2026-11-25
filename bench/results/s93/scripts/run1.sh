#!/bin/bash
cd $OUT/det
for c in nvrtc nvcc; do
 for pdl in 0 1; do for nz in 0 1; do
  if [ $c = nvrtc ]; then export CUBIN=$PWD/m2_nvrtc.cubin; else unset CUBIN; fi
  echo "== $c"; REPS=100 PDL=$pdl NOISE=$nz timeout 300 ./det_m2 | tail -4
 done; done
done
