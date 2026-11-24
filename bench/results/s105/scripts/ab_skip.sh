#!/bin/bash
# ab_skip.sh <model>: decode-only steps of 64-256 rows with the mixed flash launch skipped (1) or launched (0), ABBA, branch build.
R=<path>
m=$1; i=0
for v in 1 0 0 1; do i=$((i+1))
  SHARPLIGHT_ATTN_MIXED_SKIP=$v V=new HOLD=500 CTX=2048 $R $m skip${v}_$i "64+0,96+0,128+0,192+0,256+0,256+256x7"
done
