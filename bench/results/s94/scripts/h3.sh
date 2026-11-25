#!/bin/bash
# harness A/B with output dumps: h3.sh rounds name...
cd <harness-dir>
R=$1; shift
for r in $(seq $R); do for b in "$@"; do echo "== $b r$r"; DUMP=<harness-dir>/out_$b.bin timeout -s KILL 90 ./b_$b | grep -E "sustained|check" | head -3; done; done
md5sum out_*.bin
