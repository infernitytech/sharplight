#!/bin/bash
# greedy CLI text base vs new, and batched greedy through the server (develop-48 vs r41-48, tile-aware on / off)
. $OUT/env.sh
cd $OUT; mkdir -p correct
exec 9>$VERIFY_LOCK; flock 9
export CUDA_VISIBLE_DEVICES=0
./greedy.sh "gpt-oss-20b:$MOSS" "Qwen3-8B-FP8:$M8" 9>&-
B=$BASE/samples/Sharplight.Cli/bin/Release/net11.0/sharplight N=$TREE/samples/Sharplight.Cli/bin/Release/net11.0/sharplight
for spec in "qwen:$M8" "gptoss:$MOSS"; do m=${spec%%:*}; p=${spec#*:}
  for cfg in "b48:$B:SHARPLIGHT_STEP_ROWS_PER_SEQUENCE=48" "n48:$N:X=1" "n48off:$N:SHARPLIGHT_STEP_TILE_AWARE=0" "b48r2:$B:SHARPLIGHT_STEP_ROWS_PER_SEQUENCE=48"; do
    IFS=: read -r lab srv envs <<< "$cfg"
    export XDG_CACHE_HOME=$WORK/cache_serve_x${lab:0:1}
    env $envs SHARPLIGHT_SCHED_TRACE=1 setsid $srv serve $p --slots 16 --ctx 1024 --listen http://127.0.0.1:18090 --temp 0 --provider cuda > correct/srv_${m}_$lab.log 2>&1 9>&- &
    sp=$!
    for i in $(seq 600); do curl -sf localhost:18090/health >/dev/null && break; sleep 1; done
    $PY -I $OUT/scripts/batch_greedy.py http://127.0.0.1:18090 correct/batch_${m}_$lab.jsonl 32 64 9>&-
    echo "$m $lab $(md5sum < correct/batch_${m}_$lab.jsonl | cut -c1-8) held=$(grep -c 'held<=' correct/srv_${m}_$lab.log) whole=$(grep -c ' whole' correct/srv_${m}_$lab.log)"
    kill -- -$sp; sleep 5; kill -9 -- -$sp 2>/dev/null; wait $sp 2>/dev/null
  done
done
echo "# release $(date +%T)"
