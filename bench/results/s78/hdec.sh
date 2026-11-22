#!/bin/bash
# hdec.sh <kernel file> <label> <extra nvcc flags...>: builds the decode harness (outside the lock)
K=$1; L=$2; shift 2
cd ~/sl-worktrees/perf-r26/bench/s78_decode && CUDA_VISIBLE_DEVICES="" nvcc -O3 -arch=sm_120 -std=c++17 -DKFILE="\"$K\"" "$@" bench_decode.cu -o <path>
