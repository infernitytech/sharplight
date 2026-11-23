#!/bin/bash
cd $WORK
run() { echo "== $1"; timeout -s KILL 90 ./b_$1 | grep -E "sustained|check" | head -3; }
for r in 1 2; do for s in gu dn q; do run ${s}_132; run ${s}_130; done; done
