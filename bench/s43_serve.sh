#!/bin/bash
# Serving gap table of docs/benchmarks.md section 43 (identical procedure on every engine): per model one server per engine
# with 64 slots and 512-token contexts, then bench/loadgen.py (closed loop, ~250-token random-word prompts, 128 output tokens,
# ignore_eos, temperature 0, one concurrent warm-up request per worker) at 1 / 4 / 16 / 64 concurrent streams, ROUNDS rounds
# per concurrency (median reported by s43_serve_summarize.py). Engines of one model run back to back.
# usage: s43_serve.sh <out dir> <model key>...   (keys: qwen3-30b-a3b qwen3.6-35b-a3b gpt-oss-20b gpt-oss-20b-hf qwen3-8b-fp8-hf
#        qwen3-8b-hf; the dense Hugging Face pairs since section 44)
# env: GGUF_DIR (folder with the GGUF files), HF_DIR (Hugging Face checkpoints), VLLM_VENV, LLAMA_DIR, ENGINES (default: all
#      engines of a model), CONCS ("1 4 16 64"), ROUNDS (3), SHARPLIGHT_SERVER.
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT=${1:?out dir}; shift
export OUT_DIR=$OUT ROUNDS=${ROUNDS:-3}
CONCS=${CONCS:-"1 4 16 64"}
for key in "$@"; do
  case $key in
    qwen3-30b-a3b)   MP=$GGUF_DIR/Qwen3-30B-A3B-Q4_K_M.gguf; ENG=${ENGINES:-"sharplight llama-cuda13"} ;;
    qwen3.6-35b-a3b) MP=$GGUF_DIR/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf; ENG=${ENGINES:-"sharplight llama-cuda13"} ;;
    gpt-oss-20b)     MP=$GGUF_DIR/gpt-oss-20b-MXFP4.gguf; ENG=${ENGINES:-"sharplight llama-cuda13"} ;;
    gpt-oss-20b-hf)  MP=$HF_DIR/gpt-oss-20b; ENG=${ENGINES:-"sharplight vllm"} ;;
    qwen3-8b-fp8-hf) MP=$HF_DIR/Qwen3-8B-FP8; ENG=${ENGINES:-"sharplight vllm"} ;;
    qwen3-8b-hf)     MP=$HF_DIR/Qwen3-8B; ENG=${ENGINES:-"sharplight vllm"} ;;
    *) echo "unknown model $key"; exit 1 ;;
  esac
  for e in $ENG; do
    "$HERE/s16_serve.sh" "$e" "$key" "$MP" "$CONCS"
  done
done
