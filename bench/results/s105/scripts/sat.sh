#!/bin/bash
# sat.sh <outdir-name> <sharplight|vllm> <qwen|gptoss|llama> "<concs>" <fixed|mix|poisson> [extra server args]
# One server sized for 256 sequences (ours: --slots 256, the pool sized by what fits; vLLM: --max-num-seqs 256), 2,560-token
# contexts, closed loop at the given concurrencies, gauges sampled; mix: log-normal prompt/output lengths. Holds the bench lock.
. <path>
cd $TREE
N=$1; ENG=$2; m=$3; CONCS=$4; LOAD=$5; shift 5
case $m in qwen) M=$M8 ;; gptoss) M=$MOSS ;; llama) M=$ML ;; esac
export QUIET_GPU=1 ROUNDS=${ROUNDS:-1} SLOTS=${SLOTS:-256} CTX_PER_SLOT=${CTX:-2560} CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME=$WORK/cache_serve
LA="--gauges"
case $LOAD in mix) LA="$LA --length-mix" ;; poisson) LA="$LA --length-mix" ;; esac
export LOADGEN_ARGS="$LA"
O=$OUT/sat/${N}/${m}_${ENG}_${LOAD}${SUFFIX:-}; mkdir -p $O
# V=base (default): develop's build in the base worktree; V=new: this branch's build.
[ "${V:-base}" = base ] && export SHARPLIGHT_SERVER=$BASE/samples/Sharplight.Cli/bin/Release/net11.0/sharplight
[ "${V:-base}" = base ] || unset SHARPLIGHT_SERVER
if [ $ENG = sharplight ]; then X=(--provider cuda --max-queue 4096 "$@"); else X=(--gpu-memory-utilization ${U:-0.9} "$@"); fi
echo "$(date +%T) start $m $ENG $LOAD concs=$CONCS rates=${RATES:-} args=${X[*]}" >> $OUT/sat/${N}/runs.log
flock ~/.sl-bench.lock env OUT_DIR=$O bench/s16_serve.sh $ENG $m $M "$CONCS" "${X[@]}" > $O/driver.log 2>&1
echo "$(date +%T) end $m $ENG $LOAD rc=$?" >> $OUT/sat/${N}/runs.log
