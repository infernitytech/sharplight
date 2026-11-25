#!/bin/bash
# hb.sh <kernel file> <suffix> [extra nvcc flags]: gate/up harness binary into $WORK/bin (ONLY=dn: down; BOTH=1 both)
. <env.sh>
cd $TREE/bench/s56_moe
KF="\"$1\""; n=$2; shift 2
GU="-DK=2880 -DN=2880 -DNFULL=5760 -DDUAL=1 -DBIAS=1 -DNM_BM=32 -DNM_WM=1 -DNM_WN=8 -DNM_BN=${GU_BN:-128} -DNM_STAGES=${GU_ST:-4} -DNM_MINB=${GU_MB:-2}"
DN="-DK=2880 -DN=2880 -DNFULL=2880 -DDUAL=0 -DBIAS=1 -DNM_BM=32 -DNM_WM=1 -DNM_WN=8 -DNM_BN=${DN_BN:-192} -DNM_STAGES=${DN_ST:-4} -DNM_MINB=${DN_MB:-2}"
[ "${ONLY:-gu}" != dn ] && { nice -n 15 nvcc -O3 -arch=sm_120 -std=c++17 $GU "$@" -DKFILE="$KF" bench_moe.cu -o $WORK/bin/gu_$n -ldl 2>&1 | grep -iE "error" & }
[ -n "$BOTH" -o "${ONLY}" = dn ] && { nice -n 15 nvcc -O3 -arch=sm_120 -std=c++17 $DN "$@" -DKFILE="$KF" bench_moe.cu -o $WORK/bin/dn_$n -ldl 2>&1 | grep -iE "error" & }
wait
