#!/bin/bash
# usage: serve_ab.sh outdir "configs" models...   config: base | new | wfloor (new, floor only while no sequence decodes) | wfull (new, whole budget while some sequence decodes) | full (new, --min-step-tokens 0) | vllm
# env ARRIVAL=closed|poisson, CONCS, ROUNDS
. $OUT/env.sh
cd $TREE
export VLLM_VENV=<vllm-venv> QUIET_GPU=1 ROUNDS=${ROUNDS:-2}
O=$1; CFGS=$2; shift 2; mkdir -p $O
for m in "$@"; do
  case $m in qwen) M=$M8; R="16:7 64:17" ;; gptoss) M=$MOSS; R="16:9 64:20" ;; esac
  if [ "${ARRIVAL:-closed}" = poisson ]; then export RATES="$R" PER_OVERRIDE=${PER_OVERRIDE:-8}; else unset RATES PER_OVERRIDE; fi
  i=0
  for c in $CFGS; do i=$((i+1))
    export XDG_CACHE_HOME=$WORK/cache_serve_${c%%[0-9]*}
    unset SHARPLIGHT_STEP_FLOOR_WAVES
    case $c in
      base) SHARPLIGHT_SERVER=$BASE/samples/Sharplight.Cli/bin/Release/net11.0/sharplight OUT_DIR=$O/${m}_${c}_$i bench/s16_serve.sh sharplight $m $M "${CONCS:-16 64}" --provider cuda 9>&- ;;
      new) OUT_DIR=$O/${m}_${c}_$i bench/s16_serve.sh sharplight $m $M "${CONCS:-16 64}" --provider cuda 9>&- ;;
      wfloor) SHARPLIGHT_STEP_FLOOR_WAVES=1 OUT_DIR=$O/${m}_${c}_$i bench/s16_serve.sh sharplight $m $M "${CONCS:-16 64}" --provider cuda 9>&- ;;
      wfull) SHARPLIGHT_STEP_FLOOR_WAVES=full OUT_DIR=$O/${m}_${c}_$i bench/s16_serve.sh sharplight $m $M "${CONCS:-16 64}" --provider cuda 9>&- ;;
      full) OUT_DIR=$O/${m}_${c}_$i bench/s16_serve.sh sharplight $m $M "${CONCS:-16 64}" --provider cuda --min-step-tokens 0 9>&- ;;
      vllm) OUT_DIR=$O/${m}_${c}_$i bench/s16_serve.sh vllm $m $M "${CONCS:-16 64}" --gpu-memory-utilization 0.6 9>&- ;;
    esac
  done
done
