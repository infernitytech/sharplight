#!/bin/bash
# hold 1: vLLM operands + CUTLASS timings, harness baselines, ncu of the down baseline
cd <harness-dir>
O=<out>/harness
[ -f <vd>/dn_xc.bin ] || <vllm-venv>/bin/python -I dump_vllm.py <vd> 2>&1 | grep -E "cutlass|torch.Size" > $O/cutlass.txt
run() { echo "== $1"; timeout -s KILL 90 ./b_$1 | grep -E "sustained|check|T=|qout" | head -5; }
for b in "$@"; do run $b; done
