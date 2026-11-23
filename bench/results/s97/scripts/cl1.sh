#!/bin/bash
. <env.sh>
R1=$TREE/bench/results/s73/routing/vllm64_words.txt; R2=$TREE/bench/results/s67/harness/routes64_text.txt
for b in gu_base gu_cl dn_base dn_cl; do echo "== check $b"; $WORK/bin/$b | grep -E "check|error" | head -8; done
for i in 1 2; do for R in $R1 $R2; do for b in gu_base gu_cl dn_base dn_cl; do
printf "%-8s %-22s " $b $(basename $R); env ROUTES=$R LAYERS=24 PASSES=2 SUSTAINED_ONLY=1 ENERGY=3 $WORK/bin/$b | grep energy | cut -c1-200; done; done; done
