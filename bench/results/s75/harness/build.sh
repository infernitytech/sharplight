#!/bin/bash
# build.sh <kernel file> <suffix> [extra flags]: gate/up and down harness binaries with the engine's decode tile configs.
# BM=64 env: 64-row tile configs (gu: no table, 4 stages; dn: 3 stages) unless overridden by flags.
. $ENV
cd $REPO/bench/s56_moe
B=$WORK/bin; KF="\"$1\""; n=$2; shift 2
BM=${BM:-32}
GUS=${GUS:-4}; DNS=${DNS:-4}; GUT=${GUT:-1}; DNT=${DNT:-1}
GU="-DK=2880 -DN=2880 -DNFULL=5760 -DDUAL=1 -DBIAS=1 -DNM_BM=$BM -DNM_WM=1 -DNM_WN=8 -DNM_BN=128 -DNM_STAGES=$GUS -DNM_TABLE=$GUT -DNM_MINB=2"
DN="-DK=2880 -DN=2880 -DNFULL=2880 -DDUAL=0 -DBIAS=1 -DNM_BM=$BM -DNM_WM=1 -DNM_WN=8 -DNM_BN=${DNBN:-192} -DNM_STAGES=$DNS -DNM_TABLE=$DNT -DNM_MINB=2"
nice -n 15 nvcc -O3 -arch=sm_120 -std=c++17 $GU "$@" -DKFILE="$KF" bench_moe.cu -o $B/gu_$n -Xptxas -v 2>&1 | grep -iE "error|registers|spill" | grep -v "0 bytes spill" | sed "s/^/gu_$n: /" &
nice -n 15 nvcc -O3 -arch=sm_120 -std=c++17 $DN "$@" -DKFILE="$KF" bench_moe.cu -o $B/dn_$n -Xptxas -v 2>&1 | grep -iE "error|registers|spill" | grep -v "0 bytes spill"| sed "s/^/dn_$n: /" &
wait
