#!/bin/bash
# vLLM single-stream: HF bf16 for models that have an HF checkpoint, GGUF for the rest (may fail; failure logs are kept).
export VLLM_USE_FLASHINFER_SAMPLER=0
H=$(dirname "$0"); PY=$HOME/.venvs/vllm/bin/python; M=$HOME/models
run() { n=$1; shift; echo "== $n"; timeout 1500 $PY -I $H/vllm_single.py "$@" --out $H/results/single_vllm_$n.json > $H/results/single_vllm_$n.log 2>&1; grep -E "^RESULT" $H/results/single_vllm_$n.log | cut -c1-80 || tail -3 $H/results/single_vllm_$n.log; }
run llama3.2-1b-bf16 $M/hf/Llama-3.2-1B-Instruct llama3.2-1b-bf16
run gemma3-1b-bf16 $M/hf/gemma-3-1b-it gemma3-1b-bf16
run olmoe-bf16 $M/hf/OLMoE-1B-7B-0924-Instruct olmoe-bf16 --gpu-util 0.35
run qwen27b-q8-gguf $M/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf qwen27b-q8-gguf --gpu-util 0.5
run mistral7b-q4km-gguf $M/gguf/Mistral-7B-Instruct-v0.3/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf mistral7b-q4km-gguf
run nemotron30b-q4km-gguf $M/gguf/NVIDIA-Nemotron-3-Nano-30B-A3B/NVIDIA-Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf nemotron30b-q4km-gguf --gpu-util 0.45
