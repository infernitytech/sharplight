#!/bin/bash
# usage: serve_ab.sh outdir "configs" models...   config: base | new | full (new, --min-step-tokens 0) | trim | vllm
# env ARRIVAL=closed|poisson (RATES per model below), CONCS, ROUNDS
cd $TREE
export VLLM_VENV=$VLLM_VENV QUIET_GPU=1 ROUNDS=${ROUNDS:-2}
. $ENV_SH
O=$1; CFGS=$2; shift 2; mkdir -p $O
B=~/sl-worktrees
for m in "$@"; do
  case $m in qwen) M=~/models/hf/Qwen3-8B-FP8; R="16:7 64:17" ;; gptoss) M=~/models/hf/gpt-oss-20b; R="16:9 64:20" ;; esac
  if [ "${ARRIVAL:-closed}" = poisson ]; then export RATES="$R" PER_OVERRIDE=${PER_OVERRIDE:-8}; else unset RATES PER_OVERRIDE; fi
  i=0
  for c in $CFGS; do i=$((i+1))
    case $c in
      base) SHARPLIGHT_SERVER=$B/perf-r34-base/samples/Sharplight.Cli/bin/Release/net11.0/sharplight OUT_DIR=$O/${c}_$i bench/s16_serve.sh sharplight $m $M "${CONCS:-16 64}" --provider cuda 9>&- ;;
      new) OUT_DIR=$O/${c}_$i bench/s16_serve.sh sharplight $m $M "${CONCS:-16 64}" --provider cuda 9>&- ;;
      full) OUT_DIR=$O/${c}_$i bench/s16_serve.sh sharplight $m $M "${CONCS:-16 64}" --provider cuda --min-step-tokens 0 9>&- ;;
      trim) SHARPLIGHT_SERVER=$B/perf-r34-trim/samples/Sharplight.Cli/bin/Release/net11.0/sharplight OUT_DIR=$O/${c}_$i bench/s16_serve.sh sharplight $m $M "${CONCS:-16 64}" --provider cuda 9>&- ;;
      vllm) OUT_DIR=$O/${c}_$i bench/s16_serve.sh vllm $m $M "${CONCS:-16 64}" --gpu-memory-utilization 0.6 9>&- ;;
    esac
  done
done
