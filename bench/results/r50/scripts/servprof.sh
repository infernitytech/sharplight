#!/bin/bash
# usage: servprof.sh <outdir> <sl|vllm> <name> <model> <conc> [server args]
# Rounds: r1 warm, r2 measured (step profile on for sl), r3 nsys capture window (GPU idle gaps), r4 dotnet-trace (sl only).
# Env: SRV (sharplight binary), PER (requests per worker), NSYS=0 to skip, TRACE=0 to skip.
. <path>/env.sh
export HOME=<home>/test-home NUGET_PACKAGES=<path>/packages DOTNET_CLI_HOME=<home> SHARPLIGHT_VK_DEVICES=0 CUDA_VISIBLE_DEVICES=0
OUT=$1; ENG=$2; NAME=$3; MP=$4; N=$5; shift 5
mkdir -p $OUT; PORT=18090; PY=<path>/python; LG=<path>/loadgen.py
SLOTS=${SLOTS:-16}; PER=${PER:-4}; TAG=$OUT/${ENG}_${NAME}_c$N
uptime > $TAG.load; grep throttled_usec /sys/fs/cgroup/cpu.stat > $TAG.thr0
NS=(); [ "${NSYS:-1}" = 1 ] && NS=(nsys launch --session-new=r50$ENG -t cuda --cuda-graph-trace=graph)
set -m
if [ $ENG = sl ]; then
  SHARPLIGHT_STEP_PROFILE=$TAG.steps.csv "${NS[@]}" ${SRV:-<path>/sharplight} serve "$MP" --provider cuda --slots $SLOTS --ctx ${CTX:-512} --listen http://127.0.0.1:$PORT --temp 0 "$@" > $TAG.server.log 2>&1 &
else
  VLLM_USE_FLASHINFER_SAMPLER=0 "${NS[@]}" <path>/vllm serve "$MP" --served-model-name m --max-model-len ${CTX:-512} --max-num-seqs $SLOTS --port $PORT --no-enable-prefix-caching "$@" > $TAG.server.log 2>&1 &
fi
SP=$!; trap 'kill -- -$SP 2>/dev/null; sleep 3; kill -9 -- -$SP 2>/dev/null' EXIT
T0=$SECONDS; OK=0
while kill -0 $SP 2>/dev/null; do curl -sf localhost:$PORT/health >/dev/null 2>&1 && { OK=1; break; }; [ $((SECONDS-T0)) -gt 1500 ] && break; sleep 1; done
echo "ready=$OK after $((SECONDS-T0)) s" | tee $TAG.ready; [ $OK = 1 ] || exit 2
run() { timeout 900 $PY -I $LG --url http://127.0.0.1:$PORT --model m -c $N --per-worker $PER --ignore-eos --metrics --seed $1 --json $TAG.r$1.json | cut -c1-300; }
run 1; run 2
if [ "${NSYS:-1}" = 1 ]; then
  run 3 & L=$!; sleep ${NSYS_DELAY:-6}; nsys start --session=r50$ENG -o $TAG.nsys --force-overwrite=true; sleep ${NSYS_DUR:-6}; nsys stop --session=r50$ENG; wait $L
fi
if [ $ENG = sl ] && [ "${TRACE:-1}" = 1 ]; then
  PID=$(pgrep -x sharplight | head -1)
  run 4 & L=$!; sleep 6; DOTNET_ROLL_FORWARD=Major timeout 60 <path>/dotnet-trace collect -p $PID --profile dotnet-sampled-thread-time --duration 00:00:00:08 -o $TAG.nettrace --format Speedscope > $TAG.trace.log 2>&1; wait $L
fi
grep throttled_usec /sys/fs/cgroup/cpu.stat > $TAG.thr1; uptime >> $TAG.load
