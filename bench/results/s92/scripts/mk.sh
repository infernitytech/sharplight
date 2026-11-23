#!/bin/bash
# mk.sh name kernel.cu K N DUAL [flags]  -> binary b_name
. $ENV_SH
n=$1; kf=$2; k=$3; nn=$4; d=$5; shift 5
H=~/sl-worktrees/perf-r33/bench/s86_bs
nice -n 15 nvcc -O3 -gencode arch=compute_120f,code=sm_120f -std=c++17 -diag-suppress 177 -DK=$k -DN=$nn -DDUAL=$d -DKFILE="\"$kf\"" -DMS_A=${MSA:-2048} -DMS_B=${MSB:-1920} -DCHECK_T=${CHECK_T:-2048} -DSUSTAIN=${SUSTAIN:-1500} -DTMA -DPERSIST=${PERSIST:-82} -DKENTRY=gemm_fp8_tma -DWUNI -DF8N_WG=${WG:-1} -DF8N_WUNI=1 "$@" -I$H ${HB:-$R33/harness/bench_mc.cu} -o b_$n -lcuda -lnvidia-ml 2>&1 | grep -v "^$" | head
