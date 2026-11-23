#!/bin/bash
. <env.sh>
R=$TREE/bench/results/s73/routing/vllm64_words.txt
e() { printf "%-14s " "$1"; shift; env "$@" ROUTES=$R LAYERS=24 PASSES=2 SUSTAINED_ONLY=1 ENERGY=3 $BIN | grep energy | cut -c1-200; }
for i in 1 2; do
BIN=$WORK/bin/gu_base; for m in 1 11 12 14 17; do e RD=$m RD=$m; done
BIN=$WORK/bin/gu_dmin1noas; e dmin1noas X=1
done
