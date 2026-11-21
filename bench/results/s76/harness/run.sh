#!/bin/bash
# run.sh (inside the lock): CHECK="suffixes" RUNS="suffixes" ROUNDS=n
cd $WORK
R1=$REPO/bench/results/s73/routing/vllm64_words.txt; R2=$REPO/bench/results/s67/harness/routes64_text.txt
for v in ${CHECK:-}; do for p in gu dn; do echo "== check ${p}_$v: $(timeout 300 bin/${p}_$v | grep check | sed 's/  check //' | awk '{print $4}' | tr '\n' ' ')"; done; done
for round in $(seq 1 ${ROUNDS:-2}); do
 for rn in vllm ours; do
  [ $rn = vllm ] && R=$R1 || R=$R2; [ $rn = vllm ] && P=12 || P=60
  for suf in ${RUNS}; do
    for p in gu dn; do
      t=$(ROUTE_M=32 ROUTES=$R LAYERS=24 PASSES=$P SUSTAINED_ONLY=1 timeout 300 bin/${p}_$suf | grep "back to back" | awk '{print $(NF-3)}')
      echo "r$round $rn ${p} $suf $t"
    done
  done
 done
done
