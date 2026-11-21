#!/bin/bash
# run.sh: inside the GPU lock. RUNS="name:binary_suffix:ROUTE_M ..." ; CHECK="suffix ..."; ROUNDS
. $ENV
export CUDA_VISIBLE_DEVICES=0
B=$WORK/bin; W=$WORK
for v in ${CHECK:-}; do for p in gu dn; do echo "== check ${p}_$v: $(timeout 300 $B/${p}_$v | grep check | sed 's/  check //' | awk '{print $4}' | tr '\n' ' ')"; done; done
for round in $(seq 1 ${ROUNDS:-2}); do
 for rn in ${ROUTINGS:-vllm ours}; do
  R=$W/routes_$rn.txt; [ $rn = vllm ] && P=12 || P=60
  for spec in ${RUNS}; do
    IFS=: read name suf rm <<< "$spec"
    for p in gu dn; do
      t=$(ROUTE_M=$rm ROUTES=$R LAYERS=24 PASSES=$P SUSTAINED_ONLY=1 timeout 300 $B/${p}_$suf | grep "back to back" | awk '{print $(NF-3)}')
      echo "r$round $rn ${p} $name $t"
    done
  done
 done
done
