#!/bin/bash
source /workspace/tools/env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME=/workspace/work/fp4/cache
P=<tree>/bench/PplBench/bin/Release/net11.0/PplBench
cd /workspace/work/fp4
[ -s ppl/q3.ids ] || timeout 1200 $P /workspace/work/fp4/slm/nvidia_Qwen3-8B-FP4.slm --provider cuda --text wikitext2_test.txt --seq 3072 --chunk 1024 --tokens 12288 --ids ppl/q3.ids > ppl/ids_run.txt 2>&1
for m in nvidia_Qwen3-8B-FP4 RedHatAI_Qwen3-8B-NVFP4; do
  timeout 1800 <venvs>/vllm-r12/bin/python -I /workspace/work/fp4/vllm_ppl_fp4.py /workspace/work/fp4/models/$m ppl/q3.ids --seq 3072 --gpu-util 0.6 > ppl/vllm_$m.txt 2>&1
  echo "vllm $m: $(tail -1 ppl/vllm_$m.txt)"
done
