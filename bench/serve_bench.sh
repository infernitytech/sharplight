#!/bin/bash
# usage: serve_bench.sh <sharplight|llama-cuda13|llama-vulkan|vllm> <name> <model-path-or-hf-dir> "<concurrency list>" [extra server args...]
# Starts ONE server sized for the largest concurrency (SLOTS, default = max of the list; sharplight --slots, llama-server -np,
# vLLM --max-num-seqs), then runs loadgen.py (prompt ~256 tok, max 128 out, closed loop) for each concurrency, then stops the server.
# Results: bench/results/serve_<engine>_<name>_c<N>.json ; server log / ready time in serve_<engine>_<name>.server.log
HERE="$(cd "$(dirname "$0")" && pwd)"; . "$HERE/models.sh"
ENG=$1; NAME=$2; MP=$3; CONCS=$4; shift 4; EXTRA=("$@")
PORT=${PORT:-18080}; CTX=${CTX_PER_SLOT:-512}; 
SLOTS=${SLOTS:-$(echo $CONCS | tr ' ' '\n' | sort -n | tail -1)}
LOG=$HERE/results/serve_${ENG}_${NAME}.server.log
export VLLM_USE_FLASHINFER_SAMPLER=0
case $ENG in
  sharplight) "$HERE/../samples/Sharplight.Server/bin/Release/net11.0/sharplight-server" "$MP" --slots $SLOTS --ctx $CTX --listen http://127.0.0.1:$PORT --temp 0 "${EXTRA[@]}" > $LOG 2>&1 & ;;
  llama-*) B=${ENG#llama-}; LD_LIBRARY_PATH=$LLAMA_ROOT/runtime.llama-b11177-$B $LLAMA_ROOT/runtime.llama-b11177-$B/llama-server -m "$MP" -ngl 99 -fa on -np $SLOTS -c $((SLOTS*CTX)) --port $PORT --no-webui "${EXTRA[@]}" > $LOG 2>&1 & ;;
  vllm) $HOME/.venvs/vllm/bin/vllm serve "$MP" --served-model-name m --max-model-len $CTX --max-num-seqs $SLOTS --port $PORT --no-enable-prefix-caching "${EXTRA[@]}" > $LOG 2>&1 & ;;
esac
SP=$!; T0=$(date +%s.%N); OK=0
while kill -0 $SP 2>/dev/null; do
  curl -sf localhost:$PORT/health >/dev/null 2>&1 && { OK=1; break; }
  [ $(echo "$(date +%s.%N)-$T0>1500"|bc) = 1 ] && break; sleep 1
done
echo "ready=$OK after $(echo "$(date +%s.%N)-$T0"|bc) s" | tee -a $LOG.ready
if [ $OK = 1 ]; then
  for N in $CONCS; do
    OUT=$HERE/results/serve_${ENG}_${NAME}_c$N; "$HERE/gpu_check.sh" > $OUT.gpu.txt
    case $N in 1) PER=8;; 4) PER=6;; 16) PER=4;; *) PER=3;; esac
    IE=""; [ $ENG != sharplight ] && IE="--ignore-eos"
    timeout 900 $HOME/.venvs/vllm/bin/python -I "$HERE/loadgen.py" --url http://127.0.0.1:$PORT --model m -c $N --per-worker $PER $IE --json $OUT.json | tee $OUT.out
  done
fi
kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null; sleep 3
