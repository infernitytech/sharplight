#!/bin/bash
cd "$(dirname "$0")"; . ./models.sh
export VLLM_USE_FLASHINFER_SAMPLER=0
~/.venvs/vllm/bin/python -I vllm_single.py $HOME/models/hf-vllm/Mistral-7B-Instruct-v0.3 mistral7b-bf16 --gpu-util 0.3 --out results/single_vllm_mistral7b-bf16.json > results/single_vllm_mistral7b-bf16.log 2>&1
VLLM_MODEL=$HOME/models/hf-vllm/Mistral-7B-Instruct-v0.3 VLLM_UTIL=0.3 ./run_serve_all.sh mistral7b-q4km vllm
NP=$HOME/models/gguf/NVIDIA-Nemotron-3-Nano-30B-A3B/NVIDIA-Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
./serve_bench.sh sharplight nemotron30b-q4km $NP "1 4 16 64"
./serve_bench.sh llama-cuda13 nemotron30b-q4km $NP "1 4 16 64" --cache-ram 0
./serve_bench.sh llama-vulkan nemotron30b-q4km $NP "1 4 16 64" --cache-ram 0
