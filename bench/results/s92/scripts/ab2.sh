#!/bin/bash
cd $WORK
run() { echo "== $1"; timeout -s KILL 90 ./b_$1 | grep -E "sustained|check" | head -3; }
for r in 1 2; do for s in gu dn; do for v in 132 130 129 132O1 132ru0 132ru10 130ru10; do run ${s}_$v; done; done; done
