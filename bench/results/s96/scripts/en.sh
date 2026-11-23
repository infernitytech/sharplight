#!/bin/bash
# en.sh <outfile> <variants...>: energy windows (NVML counter) of harness variants and Marlin on both routings, alternated, 2 rounds.
# MARLIN=0 skips Marlin. ROUNDS, SEC.
. <env.sh>
O=$1; shift
R1=$TREE/bench/results/s73/routing/vllm64_words.txt; R2=$TREE/bench/results/s67/harness/routes64_text.txt
SEC=${SEC:-5}
for round in $(seq 1 ${ROUNDS:-2}); do
 for rn in vllm ours; do
  [ $rn = vllm ] && R=$R1 || R=$R2
  for p in gu dn; do
   for v in "$@"; do [ -x $WORK/bin/${p}_$v ] || continue
     printf "%s r%s %-6s %-3s %-14s " "$(date +%T)" $round $rn $p $v
     env ROUTES=$R LAYERS=24 PASSES=3 SUSTAINED_ONLY=1 ENERGY=$SEC $WORK/bin/${p}_$v | grep -E "^energy|error|check" | tr '\n' ' '; echo
   done
   if [ "${MARLIN:-1}" = 1 ]; then
     [ $p = gu ] && prod=gate_up || prod=down
     printf "%s r%s %-6s %-3s %-14s " "$(date +%T)" $round $rn $p marlin
     $PY $TREE/bench/s73_marlin_replay.py $R --product $prod --seconds $SEC --energy --reps 3 2>&1 | grep "energy" | tr '\n' ' '; echo
   fi
  done
 done
done
