#!/bin/bash
# steps.sh <tag>: gpt-oss-20b decode steps (16, 64 sequences, --text) under the clock sampler, base (r37 = develop) and new alternated
. <env.sh>
BASE=$BASE
mkdir -p $OUT/steps
for i in 1 2; do for v in base new; do
  T=$TREE; [ $v = base ] && T=$BASE
  export XDG_CACHE_HOME=$WORK/cache_bb_$v
  BB=$T/bench/BatchBench/bin/Release/net11.0/BatchBench
  M=${MODEL:-$MOSS}; n=$(basename $M)
  $PY $TREE/bench/s73_clock_sampler.py --gpu 0 --interval 0.005 --out $OUT/steps/${1}_${n}_${v}_$i.csv -- $BB $M --provider cuda --slots 64 --ctx 640 --depth 256 --sizes 1,16,64 --modes greedy --reps 30 --text > $OUT/steps/${1}_${n}_${v}_$i.txt 2>&1
  echo "== $n $v $i"; grep -iE "sampler|greedy" $OUT/steps/${1}_${n}_${v}_$i.txt | tail -5
done; done
