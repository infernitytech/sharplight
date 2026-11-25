#!/bin/bash
# In-step recurrent-state snapshots (docs/benchmarks.md section 55): TTFT of the multi-turn conversation load of bench/prefix_bench.py
# (16 conversations of 6 turns resending their history) and the shared system prompt load on a hybrid model, per GPU provider, with
# snapshots kept inside a step (default), with snapshots copied after a step (SHARPLIGHT_STEP_SNAPSHOTS=0: a snapshot inside a prompt
# chunk cuts the chunk) and without prefix caching.
# usage: s55_step_snapshots.sh <out dir> <model dir> [provider...]   providers: vulkan cuda (default: both)
# env: PY (a Python with httpx), SHARPLIGHT_SERVER (default: this checkout's Release build), PORT (18090), ROUNDS (2)
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${1:?out dir}; MP=${2:?model dir}; shift 2
PROVIDERS=${*:-"vulkan cuda"}
PORT=${PORT:-18090}; ROUNDS=${ROUNDS:-2}; PY=${PY:?set PY}
SRV=${SHARPLIGHT_SERVER:-$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight}
mkdir -p "$OUT"
set -m
for P in $PROVIDERS; do
  for C in inside copy off; do
    LOG=$OUT/$P-$C.server.log
    EXTRA=(); [ "$C" = off ] && EXTRA=(--no-prefix-cache)
    STEP=1; [ "$C" = copy ] && STEP=0
    SHARPLIGHT_STEP_SNAPSHOTS=$STEP "$SRV" serve "$MP" --provider "$P" --slots 64 --ctx 8192 --kv-tokens 73728 \
      --listen "http://127.0.0.1:$PORT" --temp 0 "${EXTRA[@]}" > "$LOG" 2>&1 &
    SP=$!; T0=$SECONDS; OK=0
    while kill -0 $SP 2>/dev/null; do
      curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { OK=1; break; }
      [ $((SECONDS - T0)) -gt 900 ] && break; sleep 1
    done
    echo "$P $C ready=$OK after $((SECONDS - T0)) s"
    if [ $OK = 1 ]; then
      for R in $(seq "$ROUNDS"); do
        for spec in "multiturn 16 0" "shared 16 4"; do
          set -- $spec
          "$PY" -I "$HERE/prefix_bench.py" --url "http://127.0.0.1:$PORT" --workload "$1" -c "$2" --per-worker "$3" --seed "$R" \
            --json "$OUT/${P}-${C}_$1_c$2_r$R.json" | cut -c1-300
        done
      done
    fi
    kill -- -$SP 2>/dev/null; sleep 3; kill -9 -- -$SP 2>/dev/null; wait $SP 2>/dev/null; sleep 5
  done
done
