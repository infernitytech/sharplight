#!/bin/bash
# Sharplight single-stream numbers for the Qwen3-8B checkpoints, models interleaved per repetition.
# Usage: CLI=<path to samples/Sharplight.Cli binary> MODELS="<checkpoint dirs and .gguf files>" OUT=<results file> sharplight_bench.sh
CLI=${CLI:?set CLI to the sharplight CLI binary}
OUT=${OUT:-sharplight_after.txt}
MODELS="${MODELS:-$HOME/models/hf/quant/nvidia_Qwen3-8B-NVFP4 $HOME/models/hf/quant/Qwen_Qwen3-8B-AWQ $HOME/models/hf/quant/JunHowie_Qwen3-8B-GPTQ-Int4 $HOME/models/hf/quant/unsloth_Qwen3-8B-bnb-4bit $HOME/models/gguf/Qwen3-8B/Qwen3-8B-Q4_K_M.gguf}"
for r in 1 2 3; do for N in 16 512 2048 4096; do for m in $MODELS; do
  line=$(timeout 900 "$CLI" "$m" --bench $N --ctx $((N+256)) --batch 512 2>&1 | grep '^prefill')
  echo "$(basename "$m") N=$N run=$r $line" >> "$OUT"
done; done; done
