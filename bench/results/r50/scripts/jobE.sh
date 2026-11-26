#!/bin/bash
# GPU job E: serving reruns at 1 stream (job B's branch run overlapped a foreign GPU load 11:08-11:14) and 16 streams, final branch build, and base again.
G=<path>/gemma-4-31B-it-Q4_K_M.gguf; S=<path>/servprof.sh
nvidia-smi --query-compute-apps=pid,name --format=csv > <path>/e_gpu_apps.txt
for c in 1 16; do PER=$([ $c = 1 ] && echo 8 || echo 4) NSYS_DELAY=14 TRACE=0 SRV=<path>/sharplight $S <path>/br sl gemma_q4 $G $c; done
PER=8 NSYS=0 TRACE=0 SRV=<path>/sharplight $S <path>/base sl gemma_q4 $G 1
echo jobE-done
