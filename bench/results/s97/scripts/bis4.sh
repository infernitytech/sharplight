#!/bin/bash
. <env.sh>
R=$TREE/bench/results/s73/routing/vllm64_words.txt
e() { printf "%-14s " "$1"; shift; env "$@" ROUTES=$R LAYERS=24 PASSES=2 SUSTAINED_ONLY=1 ENERGY=3 $BIN | grep energy | cut -c1-200; }
BIN=$WORK/bin/gu_rdx
for i in 1 2; do for m in 1 18 116 132 11; do e RD=$m RD=$m; done; done
