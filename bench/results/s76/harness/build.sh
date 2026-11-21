#!/bin/bash
# build.sh <suffix> [flags]
. $ENV
cd $WORK
n=$1; shift
GU="-DK=2880 -DN=2880 -DNFULL=5760 -DDUAL=1 -DBIAS=1 -DNM_BM=32 -DNM_WM=1 -DNM_WN=8 -DNM_BN=128 -DNM_STAGES=4 -DNM_TABLE=1 -DNM_MINB=2"
DN="-DK=2880 -DN=2880 -DNFULL=2880 -DDUAL=0 -DBIAS=1 -DNM_BM=32 -DNM_WM=1 -DNM_WN=8 -DNM_BN=192 -DNM_STAGES=4 -DNM_TABLE=1 -DNM_MINB=2"
nice -n 15 nvcc -O3 -arch=sm_120 -std=c++17 $GU "$@" -DKFILE='"k.cu"' bench_moe.cu -o bin/gu_$n -Xptxas -v 2>&1 | grep -iE "error|registers|spill" | grep -v "0 bytes spill" | sed "s/^/gu_$n: /" &
nice -n 15 nvcc -O3 -arch=sm_120 -std=c++17 $DN "$@" -DKFILE='"k.cu"' bench_moe.cu -o bin/dn_$n -Xptxas -v 2>&1 | grep -iE "error|registers|spill" | grep -v "0 bytes spill" | sed "s/^/dn_$n: /" &
wait
