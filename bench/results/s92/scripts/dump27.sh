#!/bin/bash
. $ENV_SH
D=$WORK/dump27; rm -rf $D; mkdir -p $D
export SHARPLIGHT_CUDA_DUMP_SOURCE=$D XDG_CACHE_HOME=$WORK/cache_dump27 SHARPLIGHT_CUDA_NVRTC_SASS=0
$TREE/bench/BatchBench/bin/Release/net11.0/BatchBench ~/models/hf/Qwen3.8-27B-FP8 --provider cuda --slots 4 --ctx 1024 --depth 16 --batch 512 --modes greedy --reps 1 --mixed "0+500x1" 2>&1 | grep -E "^\| 0" 
cd $D; for f in gemm_fp8_tma*.ptx; do python3 -I $WORK/tools/drvlink.py $f ${f%.ptx}.drv.cubin || echo "fail $f"; done
cd $WORK; for b in b_pf_dn_ptx b_pf_dn90_ptx b_pf_dn_132_cubin b_pf_dn_130_cubin b_pf_dn_ptx b_pf_dn90_ptx b_pf_dn_132_cubin b_pf_dn_130_cubin; do echo "== $b"; timeout -s KILL 90 ./$b | grep -E "sustained|check|fail" | head -3; done
