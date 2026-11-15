#!/bin/bash
# vLLM single-stream numbers for the Qwen3-8B quantized checkpoints (bf16 KV cache, flash-attn, no FlashInfer JIT).
# Usage: VLLM_PYTHON=<python of the vLLM venv> vllm_bench.sh  (run from the repository root; one RESULT line per model)
VLLM_PYTHON=${VLLM_PYTHON:-$HOME/.venvs/vllm/bin/python}
for m in nvidia_Qwen3-8B-NVFP4 Qwen_Qwen3-8B-AWQ JunHowie_Qwen3-8B-GPTQ-Int4 unsloth_Qwen3-8B-bnb-4bit; do
  # The AWQ checkpoint is f16: vLLM rejects a bf16 KV cache for it, so that run leaves --kv at its default.
  VLLM_USE_FLASHINFER_SAMPLER=0 timeout 1500 "$VLLM_PYTHON" -I bench/vllm_single.py "$HOME/models/hf/quant/$m" "$m" --reps 5 --kv bfloat16 --attn FLASH_ATTN 2>&1 \
    | grep "^RESULT\|Error:"
done
