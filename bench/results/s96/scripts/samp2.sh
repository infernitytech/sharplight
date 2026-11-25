#!/bin/bash
. <env.sh>
R1=$TREE/bench/results/s73/routing/vllm64_words.txt
S="$PY $TREE/bench/s73_clock_sampler.py --gpu 0 --interval 0.005"
for v in "$@"; do
 case $v in
  re_*) $S --out $OUT/samp/$v.csv -- env SECONDS_RUN=5 $WORK/bin/$v | grep -vE "instant" ;;
  *) $S --out $OUT/samp/$v.csv -- env ROUTES=$R1 LAYERS=24 PASSES=2 SUSTAINED_ONLY=1 ENERGY=5 $WORK/bin/$v | grep -E "sampler: [0-9]|energy" | grep -v instant ;;
 esac
done
