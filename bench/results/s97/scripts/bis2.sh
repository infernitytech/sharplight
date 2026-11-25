#!/bin/bash
. <env.sh>
R=$TREE/bench/results/s73/routing/vllm64_words.txt
e() { printf "%-14s " "$1"; shift; env "$@" ROUTES=$R LAYERS=24 PASSES=2 SUSTAINED_ONLY=1 ENERGY=3 $BIN | grep energy | cut -c1-200; }
for i in 1 2; do
BIN=$WORK/bin/gu_base; for m in 1 2 3 4 5; do e RD=$m RD=$m; done
for v in dmin1 dmin1noa dmin1nos dmin1noas; do BIN=$WORK/bin/gu_$v; e $v X=1; done
printf "%-14s " rk; SECONDS_RUN=3 SMEM_KB=34 $WORK/bin/rk
done
