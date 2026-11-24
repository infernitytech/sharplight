#!/bin/bash
# usage: build.sh name K N BM BN WM WN STAGES GS DUAL [MINB]
n=$1; shift; K=$1 N=$2 BM=$3 BN=$4 WM=$5 WN=$6 ST=$7 GS=$8 DUAL=$9 MINB=${10:-1}
D="-DK=$K -DN=$N -DY_STRIDE=$N -DX_STRIDE=$K -DF4_BM=$BM -DF4_BN=$BN -DF4_WM=$WM -DF4_WN=$WN -DF4_STAGES=$ST -DF4_GS=$GS -DDUAL=$DUAL -DF4_MINB=$MINB"
[ "$DUAL" = 1 ] && D="$D -include epi.h"
nice nvcc -O3 -gencode arch=compute_120f,code=sm_120f $D -o $n main.cu wrap_q.cu wrap_g.cu -Xptxas -v 2>&1 | grep -E "error|registers|spill" | head -6
