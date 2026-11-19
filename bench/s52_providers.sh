#!/bin/bash
# Section 52 of docs/benchmarks.md (procedure of section 51): the CUDA and the Vulkan provider on the same GPU and the same weights (procedure of section 42:
# one process per model and prompt length, `--bench N --ctx N+256`, prefill median of 5, decode median of 64 steps after the prompt).
# usage: s52_providers.sh <out file> <cli> <provider> [extra CLI arguments]
# env: MODELS (directory with gguf/ and hf/, default ~/models), ONLY (space-separated subset of the model names below),
#      LENS8 (prompt lengths of the 8B models, default "512 2048 8192"). Pin the GPU with CUDA_VISIBLE_DEVICES and SHARPLIGHT_VK_DEVICES.
OUT=${1:?out file}; CLI=${2:?cli}; PROV=${3:?provider}; shift 3
M=${MODELS:-$HOME/models}
for spec in "qwen3-0.6b-q8|$M/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf|512 2048" "llama3.2-1b-q8|$M/gguf/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-Q8_0.gguf|512 2048" \
            "gemma3-1b-q8|$M/gguf/gemma-3-1b-it/gemma-3-1b-it-Q8_0.gguf|512 2048" "qwen3-8b|$M/hf/Qwen3-8B|${LENS8:-512 2048 8192}" \
            "qwen3-8b-fp8|$M/hf/Qwen3-8B-FP8|${LENS8:-512 2048 8192}" "llama3.1-8b|$M/hf/Llama-3.1-8B-Instruct|${LENS8:-512 2048 8192}" \
            "gpt-oss-20b|$M/hf/gpt-oss-20b|${LENS8:-512 2048 8192}"; do
  IFS='|' read -r name path lens <<< "$spec"
  [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]] && continue
  for N in $lens; do
    # shellcheck disable=SC2086
    "$CLI" "$path" --provider "$PROV" --bench "$N" --ctx $((N + 256)) --no-pull "$@" 2>&1 | grep -E '^prefill|Unhandled' | sed "s/^/[$name N=$N] /" >> "$OUT"
  done
done
