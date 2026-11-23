#!/bin/bash
# usage: mk.sh <name> <kernel.cu> K N DUAL [nvcc flags...] -> binary b_<name>
# HB: the harness (bench/s86_bs/bench_fp8.cu with bench_fp8_harness_cluster_loadtag.diff applied; default bench_mc.cu here),
# H: its include directory. Env: MSA / MSB (row counts), CHECK_T, SUSTAIN (ms), PERSIST (persistent blocks, 82), WG (copy warpgroup, 1).
n=$1; kf=$2; k=$3; nn=$4; d=$5; shift 5
H=${H:-bench/s86_bs}
nice -n 15 nvcc -O3 -gencode arch=compute_120f,code=sm_120f -std=c++17 -diag-suppress 177 -DK=$k -DN=$nn -DDUAL=$d -DKFILE="\"$kf\"" \
  -DMS_A=${MSA:-2048} -DMS_B=${MSB:-1920} -DCHECK_T=${CHECK_T:-2048} -DSUSTAIN=${SUSTAIN:-1500} -DTMA -DPERSIST=${PERSIST:-82} \
  -DKENTRY=gemm_fp8_tma -DWUNI -DF8N_WG=${WG:-1} -DF8N_WUNI=1 "$@" -I$H ${HB:-bench_mc.cu} -o b_$n -lcuda -lnvidia-ml
