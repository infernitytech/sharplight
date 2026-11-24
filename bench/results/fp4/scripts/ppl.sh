#!/bin/bash
# usage: ppl.sh <label> <model> [env...]
source /workspace/tools/env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME=/workspace/work/fp4/cache
P=<tree>/bench/PplBench/bin/Release/net11.0/PplBench
lab=$1; M=$2; shift 2
env "$@" timeout 2400 $P $M --provider cuda --text /workspace/work/fp4/wikitext2_test.txt --seq 3072 --chunk 1024 --tokens 12288 > /workspace/work/fp4/ppl/$lab.txt 2>&1
echo "ppl $lab rc=$? $(grep -io 'ppl[ =:]*[0-9.]*' /workspace/work/fp4/ppl/$lab.txt | tail -1)"
