#!/bin/bash
# expert GEMMs on vLLM's 16-sequence routing: record it, then our harness and Marlin's replay, alternated; one hold
. $OUT/env.sh
exec 9><gpu lock>; flock 9
export CUDA_VISIBLE_DEVICES=0
echo "# hold $(date +%T) apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
cd $TREE/bench
R16=$OUT/harness/vllm16_words.txt; R64=$TREE/bench/results/s73/routing/vllm64_words.txt
[ -s $R16 ] || $PY s73_vllm_routing.py $MOSS --out $R16 --seqs 16 --depth 256 --steps 64 9>&- > $OUT/harness/routing16.log 2>&1
wc -l $R16
for r in 1 2; do for R in $R16 $R64; do
  echo "== round $r $(basename $R)"
  ROUTES=$R LAYERS=24 PASSES=2 SUSTAINED_ONLY=1 ENERGY=3 $WORK/bin/gu 9>&- | grep -E "energy|per routing" | sed 's/^/ours gate_up: /'
  ROUTES=$R LAYERS=24 PASSES=2 SUSTAINED_ONLY=1 ENERGY=3 $WORK/bin/dn 9>&- | grep -E "energy|per routing" | sed 's/^/ours down: /'
  $PY s73_marlin_replay.py $R --product gate_up --seconds 3 --energy --gpu 0 9>&- 2>/dev/null | tail -3 | sed 's/^/marlin gate_up: /'
  $PY s73_marlin_replay.py $R --product down --seconds 3 --energy --gpu 0 9>&- 2>/dev/null | tail -3 | sed 's/^/marlin down: /'
done; done
echo "# release $(date +%T)"
