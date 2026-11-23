#!/bin/bash
. <env.sh>
R1=$TREE/bench/results/s73/routing/vllm64_words.txt; R2=$TREE/bench/results/s67/harness/routes64_text.txt
$WORK/bin/gu_cl64t0 | grep -E "check|T= *64 " | head -3
e() { printf "%-28s " "$1"; shift; env "$@" LAYERS=24 PASSES=2 SUSTAINED_ONLY=1 ENERGY=3 $B | grep energy | cut -c1-175; }
for i in 1 2; do for R in $R1 $R2; do echo "-- $(basename $R)"; for p in gu dn; do
B=$WORK/bin/${p}_cl;    e "$p 32-row all" ROUTES=$R
B=$WORK/bin/${p}_cl64t0; e "$p 64-row all" ROUTES=$R ROUTE_M=64
B=$WORK/bin/${p}_cl;    e "$p 32-row experts<=32" ROUTES=$R ROUTE_M=64 MAX_ROWS=32
B=$WORK/bin/${p}_cl64t0; e "$p 64-row experts>32" ROUTES=$R ROUTE_M=64 MIN_ROWS=33
done; done; done
