#!/bin/bash
# x<name>: new with the env assignments in $CFG_x<name>
# env ARRIVAL=closed|poisson, CONCS, ROUNDS
. <path>
cd $TREE
export FIRST=1 VLLM_VENV=~/venvs/vllm-r12 QUIET_GPU=1 ROUNDS=${ROUNDS:-2}
O=$1; CFGS=$2; shift 2; mkdir -p $O
for m in "$@"; do
  case $m in qwen) M=$M8; R="16:7 64:17"; U=0.6 ;; gptoss) M=$MOSS; R="16:9 64:20"; U=0.6 ;; llama) M=~/models-local/hf/Llama-3.1-8B-Instruct; R="16:4.3 64:11"; U=0.8 ;; esac
  if [ "${ARRIVAL:-closed}" = poisson ]; then export RATES="$R" PER_OVERRIDE=${PER_OVERRIDE:-8}; else unset RATES PER_OVERRIDE; fi
  i=0
  for c in $CFGS; do i=$((i+1))
    export XDG_CACHE_HOME=$WORK/cache_serve_${c%%[0-9]*}
    unset SHARPLIGHT_STEP_FLOOR_WAVES
    case $c in
      base) SHARPLIGHT_SERVER=$BASE/samples/Sharplight.Cli/bin/Release/net11.0/sharplight OUT_DIR=$O/${m}_${c}_$i bench/s16_serve.sh sharplight $m $M "${CONCS:-16 64}" --provider cuda 9>&- ;;
      new) OUT_DIR=$O/${m}_${c}_$i bench/s16_serve.sh sharplight $m $M "${CONCS:-16 64}" --provider cuda 9>&- ;;
      x*) v=CFG_$c; env ${!v} OUT_DIR=$O/${m}_${c}_$i bench/s16_serve.sh sharplight $m $M "${CONCS:-16 64}" --provider cuda 9>&- ;;
      vllm) OUT_DIR=$O/${m}_${c}_$i bench/s16_serve.sh vllm $m $M "${CONCS:-16 64}" --gpu-memory-utilization $U 9>&- ;;
    esac
  done
done
