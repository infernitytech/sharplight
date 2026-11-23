#!/bin/bash
. <env.sh>
R=$TREE/bench/results/s73/routing/vllm64_words.txt
for i in 1 2; do
printf "%-12s " gu_base;  env ROUTES=$R LAYERS=24 PASSES=2 SUSTAINED_ONLY=1 ENERGY=3 $WORK/bin/gu_base | grep energy | cut -c1-220
printf "%-12s " gu_dmin1; env ROUTES=$R LAYERS=24 PASSES=2 SUSTAINED_ONLY=1 ENERGY=3 $WORK/bin/gu_dmin1 | grep energy | cut -c1-220
printf "%-12s " gu_RD;    env RD=1 ROUTES=$R LAYERS=24 PASSES=2 SUSTAINED_ONLY=1 ENERGY=3 $WORK/bin/gu_base | grep energy | cut -c1-220
printf "%-12s " rk;       SECONDS_RUN=3 SMEM_KB=34 $WORK/bin/rk
printf "%-12s " rg_base;  SECONDS_RUN=3 $WORK/bin/rg_base
printf "%-12s " rg_dmin1; SECONDS_RUN=3 $WORK/bin/rg_dmin1
done
