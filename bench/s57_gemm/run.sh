#!/bin/bash
# usage: run.sh kernel.cu [extra nvcc flags]; env: SHAPES ("K,N,DUAL ..."), MSA / MSB (row counts), CHECK_T, SUSTAIN (ms), ARCH
K=$1; shift; EXTRA="$@"
SHAPES=${SHAPES:-"4096,4096,0 4096,1024,0 4096,12288,1 12288,4096,0"}
for s in $SHAPES; do IFS=, read k n d <<< "$s"
  nice -n 15 nvcc -O3 ${ARCH:--arch=sm_120} -std=c++17 -diag-suppress 177 -DK=$k -DN=$n -DDUAL=$d -DKFILE="\"$K\"" -DMS_A=${MSA:-2048} -DMS_B=${MSB:-8192} -DCHECK_T=${CHECK_T:-2048} -DSUSTAIN=${SUSTAIN:-1500} $EXTRA bench_fp8.cu -o b_$$ -lcuda -lnvidia-ml && ./b_$$ | grep -v "^$"
done
rm -f b_$$
