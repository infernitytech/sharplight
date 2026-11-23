#!/bin/bash
. <env.sh>
R1=$TREE/bench/results/s73/routing/vllm64_words.txt
for i in 1 2; do for cfg in "rk_0_32 SMEM_KB=34" "rk_s4_a1 SMEM_KB=34" "rk_s5_a1 SMEM_KB=41" "rk_s8_a1 SMEM_KB=34"; do set -- $cfg; b=$1; shift; printf "%-24s " "$cfg"; env SECONDS_RUN=3 "$@" $WORK/bin/$b; done
for v in gu_dbare gu_dfull gu_s3b3 gu_s3b2 dn_dfull dn_s3b3 dn_s4b3; do printf "%-24s " $v; env ROUTES=$R1 LAYERS=24 PASSES=2 SUSTAINED_ONLY=1 ENERGY=3 $WORK/bin/$v | grep energy | cut -c1-200; done; done
