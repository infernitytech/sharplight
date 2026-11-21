#!/bin/bash
# build_harness.sh <kernel file> <suffix> [extra flags]: gate/up and down harness binaries with the engine's decode tile configs.
. <path>
cd ~/sl-worktrees/perf-r23/bench/s56_moe
B=<path> KF="\"$1\""; n=$2; shift 2
GU="-DK=2880 -DN=2880 -DNFULL=5760 -DDUAL=1 -DBIAS=1 -DNM_BM=32 -DNM_WM=1 -DNM_WN=8 -DNM_BN=128 -DNM_STAGES=4 -DNM_MINB=2"
DN="-DK=2880 -DN=2880 -DNFULL=2880 -DDUAL=0 -DBIAS=1 -DNM_BM=32 -DNM_WM=1 -DNM_WN=8 -DNM_BN=192 -DNM_STAGES=4 -DNM_MINB=2"
nice -n 15 nvcc -O3 -arch=sm_120 -std=c++17 $GU "$@" -DKFILE="$KF" bench_moe.cu -o $B/gu_$n 2>&1 | grep -i error &
nice -n 15 nvcc -O3 -arch=sm_120 -std=c++17 $DN "$@" -DKFILE="$KF" bench_moe.cu -o $B/dn_$n 2>&1 | grep -i error &
wait
