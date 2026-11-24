#!/bin/bash
# bisect the batched Qwen3-8B-FP8 garbage over older builds; a bf16 dense model on develop
. $OUT/env.sh
cd $OUT; mkdir -p diag
exec 9>$VERIFY_LOCK; flock 9
export CUDA_VISIBLE_DEVICES=0
run() { lab=$1; srv=$2; model=$3; shift 3
  export XDG_CACHE_HOME=$WORK/cache_diag_$lab
  env SHARPLIGHT_SCHED_TRACE=1 setsid $srv serve $model --slots 16 --ctx 1024 --listen http://127.0.0.1:18090 --temp 0 --provider cuda "$@" > diag/srv_$lab.log 2>&1 9>&- &
  sp=$!
  for i in $(seq 900); do curl -sf localhost:18090/health >/dev/null && break; sleep 1; done
  $PY -I $OUT/scripts/batch_greedy.py http://127.0.0.1:18090 diag/batch_$lab.jsonl 32 48 9>&-
  echo "$lab: $(python3 -c "import json;a=[json.loads(l)['text'] for l in open('diag/batch_$lab.jsonl')];print(sum('!!' in t for t in a),'of',len(a),'with !!;',repr(a[1][:70]))")"
  kill -- -$sp; sleep 5; kill -9 -- -$sp 2>/dev/null; wait $sp 2>/dev/null
}
S=samples/Sharplight.Cli/bin/Release/net11.0/sharplight
run r23 $TREES/perf-r23/$S $M8
run r37 $TREES/perf-r37/$S $M8
run r39 $TREES/perf-r39/$S $M8
run q8bf16 $BASE/$S $MODELS/hf/Qwen3-8B
run llama $BASE/$S $MODELS/hf/Llama-3.1-8B-Instruct
echo "# release $(date +%T)"
