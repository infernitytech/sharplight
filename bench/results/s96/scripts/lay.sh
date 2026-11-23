#!/bin/bash
. <env.sh>
R1=$TREE/bench/results/s73/routing/vllm64_words.txt
for v in "$@"; do for L in 1 2 4 8 24; do printf "%-10s L=%-3s " $v $L; env ROUTES=$R1 LAYERS=$L PASSES=2 SUSTAINED_ONLY=1 ENERGY=4 $WORK/bin/$v | grep energy; done; done
