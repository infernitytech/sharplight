#!/bin/bash
. "$(dirname "$0")/models.sh"
for e in "${MODELS[@]}"; do n=${e%%|*}; p=${e##*|}
  [ $n = qwen27b-q8 ] && p=$HOME/models/sharplight/Qwen3.8-27B-Q8_0.slm
  echo "== $n"; "$(dirname "$0")/sharplight_single.sh" $n "$p" 3
done
