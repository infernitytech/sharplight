#!/bin/bash
# Prefill with an identical procedure on both sides (docs/benchmarks.md section 40.9): for every engine and model one process
# run, per prompt length (512 / 2048 / 8192 random tokens) one warm-up prefill and 5 timed prefills, reported as the median.
#   Sharplight: `--bench N` (warm-up prefill, then 5 timed prefills; "prefill median of 5"), one process per length.
#   vLLM:       bench/vllm_single.py --points pp --reps 5 (one warm-up generate per length, 5 timed, median), same HF checkpoints.
#   llama.cpp:  llama-bench -p 512,2048,8192 -n 0 -r 5 (one warm-up run per test, 5 timed; median of samples_ns), same GGUFs.
# Engines run back to back per model. usage: s40_identical.sh <out dir>
# env: HF_DIR, GGUF_DIR, VLLM_VENV, LLAMA_BENCH (llama-bench of the CUDA build; its directory goes on LD_LIBRARY_PATH).
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT=${1:?out dir}; mkdir -p "$OUT"
CLI=$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight
HF="qwen3-0.6b:Qwen3-0.6B llama3.2-1b:Llama-3.2-1B-Instruct qwen3-8b:Qwen3-8B qwen3-8b-fp8:Qwen3-8B-FP8"
GGUF="qwen3-0.6b-q8_0:Qwen3-0.6B-Q8_0.gguf llama3.2-1b-q8_0:Llama-3.2-1B-Instruct-Q8_0.gguf qwen3-8b-q4_k_m:Qwen3-8B-Q4_K_M.gguf"
sl() {  # sl <name> <model path>
  : > "$OUT/sl_$1.txt"
  for N in 512 2048 8192; do
    "$CLI" "$2" --bench $N --ctx $((N + 256)) 2>&1 | grep -E '^prefill|rror' | sed "s/^/[N=$N] /" >> "$OUT/sl_$1.txt"
  done
}
for m in $HF; do
  n=${m%%:*}; d=$HF_DIR/${m#*:}
  sl "$n" "$d"
  "$VLLM_VENV/bin/python" -I "$HERE/vllm_single.py" "$d" "$n" --points pp --reps 5 --gpu-util 0.85 --depth 8192 --max-len 8704 \
    --out "$OUT/vllm_$n.json" > "$OUT/vllm_$n.log" 2>&1
done
for m in $GGUF; do
  n=${m%%:*}; g=$GGUF_DIR/${m#*:}
  sl "$n" "$g"
  LD_LIBRARY_PATH=$(dirname "$LLAMA_BENCH") "$LLAMA_BENCH" -m "$g" -ngl 99 -fa 1 -p 512,2048,8192 -n 0 -r 5 -o json \
    > "$OUT/llama_$n.json" 2> "$OUT/llama_$n.err"
done
echo done
