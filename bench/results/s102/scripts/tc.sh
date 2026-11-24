#!/bin/bash
# textcheck against a server: usage tc.sh <label> <cli-tree> <model-name> <model-path> [dump:1] [extra serve args...]
. <path>
lab=$1; tree=$2; m=$3; p=$4; dump=$5; shift 5
export CUDA_VISIBLE_DEVICES=0
if [ "$dump" = 1 ]; then export XDG_CACHE_HOME=$WORK/cache_dump_$m; rm -rf $XDG_CACHE_HOME; mkdir -p $OUT/dump/$m; export SHARPLIGHT_CUDA_DUMP_SOURCE=$OUT/dump/$m; else export XDG_CACHE_HOME=$WORK/cache_$lab; fi
setsid $tree/samples/Sharplight.Cli/bin/Release/net11.0/sharplight serve $p --slots 16 --ctx 1024 --listen http://127.0.0.1:18090 --temp 0 --provider cuda "$@" > $OUT/tc/srv_${m}_$lab.log 2>&1 &
sp=$!
for i in $(seq 900); do curl -sf localhost:18090/health >/dev/null && break; sleep 1; done
$PY -I $TREE/bench/textcheck.py --url http://127.0.0.1:18090 --waves 16,32 --json $OUT/tc/${m}_$lab.json > $OUT/tc/${m}_$lab.txt 2>&1
echo "$m $lab rc=$? $(tail -1 $OUT/tc/${m}_$lab.txt)"
kill -- -$sp; sleep 5; kill -9 -- -$sp 2>/dev/null; wait $sp 2>/dev/null
