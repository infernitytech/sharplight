#!/bin/bash
# sat.sh <outdir-name> <sharplight|vllm> <qwen|llama> "<concs>" <mix|fixed> [extra server args]
# One server sized for 256 sequences, 2,560-token contexts, closed loop, gauges sampled, length mix; holds the bench lock.
. $(dirname "$0")/env.sh
cd $TREE
N=$1; ENG=$2; m=$3; CONCS=$4; LOAD=$5; shift 5
case $m in qwen) M=$M8 ;; llama) M=$ML ;; esac
export QUIET_GPU=1 ROUNDS=${ROUNDS:-1} SLOTS=${SLOTS:-256} CTX_PER_SLOT=${CTX:-2560} CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME=$WORK/cache_serve
LA="--gauges"
[ $LOAD = mix ] && LA="$LA --length-mix"
export LOADGEN_ARGS="$LA"
O=$OUT/sat/${N}/${m}_${ENG}_${LOAD}${SUFFIX:-}; mkdir -p $O
if [ $ENG = sharplight ]; then X=(--provider cuda --max-queue 4096 "$@"); else X=(--gpu-memory-utilization ${U:-0.9} "$@"); fi
echo "$(date +%T) start $m $ENG $LOAD concs=$CONCS args=${X[*]}" >> $OUT/sat/${N}/runs.log
flock ~/.sl-bench.lock env OUT_DIR=$O bench/s16_serve.sh $ENG $m $M "$CONCS" "${X[@]}" > $O/driver.log 2>&1
echo "$(date +%T) end $m $ENG $LOAD rc=$?" >> $OUT/sat/${N}/runs.log
