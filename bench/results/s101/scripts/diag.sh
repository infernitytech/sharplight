#!/bin/bash
# Qwen3-8B-FP8 garbage in batched serving: single request, batch with block-scaled FP8 off, batch without prefix cache
. $OUT/env.sh
cd $OUT; mkdir -p diag
exec 9>$VERIFY_LOCK; flock 9
export CUDA_VISIBLE_DEVICES=0
B=$BASE/samples/Sharplight.Cli/bin/Release/net11.0/sharplight
export XDG_CACHE_HOME=$WORK/cache_serve_xb
run() { lab=$1; n=$2; shift 2
  env SHARPLIGHT_SCHED_TRACE=1 setsid $B serve $M8 --slots 16 --ctx 1024 --listen http://127.0.0.1:18090 --temp 0 --provider cuda "$@" > diag/srv_$lab.log 2>&1 9>&- &
  sp=$!
  for i in $(seq 600); do curl -sf localhost:18090/health >/dev/null && break; sleep 1; done
  $PY -I $OUT/scripts/batch_greedy.py http://127.0.0.1:18090 diag/batch_$lab.jsonl $n 48 9>&-
  echo "$lab: $(python3 -c "import json;a=[json.loads(l)['text'] for l in open('diag/batch_$lab.jsonl')];print(sum('!!' in t for t in a),'of',len(a),'with !!;',repr(a[0][:70]))")"
  kill -- -$sp; sleep 5; kill -9 -- -$sp 2>/dev/null; wait $sp 2>/dev/null
}
run single 1
run batch 32
run nobs 32 --fp8-block-scaled off
run nocache 32 --no-prefix-cache
run slots4 32 --slots 4
echo "# release $(date +%T)"
