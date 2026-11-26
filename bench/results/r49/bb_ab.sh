#!/bin/bash
# usage: bb_ab.sh <tag> <model> <sizes> <pairs> [extra]
cd <checkout>; export CUDA_VISIBLE_DEVICES=0
T=$1; M=$2; S=$3; P=$4; shift 4
for i in $(seq $P); do for mv in 0 1; do
  SHARPLIGHT_MULTI_MATVEC=$mv timeout 1200 bench/BatchBench/bin/Release/net11.0/BatchBench $M --provider ${PROV:-cuda} --slots ${SLOTS:-1} --ctx 1024 --depth 256 --sizes $S --modes greedy --reps ${REPS:-300} "$@" > bench/results/r49/mv/bb_${T}_mv${mv}_$i.txt 2>&1
  echo "mv=$mv run=$i"; grep -E "^\|" bench/results/r49/mv/bb_${T}_mv${mv}_$i.txt | tail -3
done; done
