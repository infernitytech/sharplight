#!/bin/bash
cd $WORK/dump27; s=gemm_fp8_tma_gemm_fp8_tma_5_452868973910494
for t in 131192 65656 120; do JIT_TARGET=$t python3 -I $WORK/tools/drvlink.py $s.ptx $WORK/jit_$t.cubin && echo "target $t: maxreg $(cuobjdump -sass $WORK/jit_$t.cubin | grep -c USETMAXREG) arch $(cuobjdump -sass $WORK/jit_$t.cubin | grep -m1 -o 'sm_[0-9]*[af]*')"; done
