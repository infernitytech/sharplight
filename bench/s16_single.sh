#!/bin/bash
# Single-stream runs of docs/benchmarks.md section 16.
# usage: s16_single.sh <sharplight|llama-cuda13|llama-vulkan|vllm> <name> <model> [extra engine args...]
#   sharplight: `sharplight MODEL --bench N --ctx N+256 --batch 512` for N = 16, 512, 2048, 4096, REPS process runs each
#               (best prefill of 4, median / best of 64 decode steps at depth N, GPU ms per step, dispatches / barriers)
#   llama-*:    llama-bench -ngl 99 -fa 1 -p 512,2048 -n 128 -d 0,4096 -r 5 (jsonl)
#   vllm:       bench/vllm_single.py (offline LLM API, median of 5); extra args go to vllm_single.py (e.g. --gpu-util 0.5,
#               --spec-method mtp --spec-tokens 3)
# env: OUT_DIR (default bench/results/s16), REPS (3), NS ("16 512 2048 4096"), SHARPLIGHT_CLI (default: the Release build in
#      this checkout), LLAMA_DIR (folder holding runtime.llama-<LLAMA_BUILD>-<backend>/), LLAMA_BUILD (b11177),
#      VLLM_VENV (virtualenv with vLLM).
HERE="$(cd "$(dirname "$0")" && pwd)"
ENG=$1; NAME=$2; MP=$3; shift 3; EXTRA=("$@")
OUT_DIR=${OUT_DIR:-$HERE/results/s16}; mkdir -p "$OUT_DIR"
REPS=${REPS:-3}; NS=${NS:-16 512 2048 4096}
export VLLM_USE_FLASHINFER_SAMPLER=0
case $ENG in
  sharplight)
    CLI=${SHARPLIGHT_CLI:-$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight}
    OUT=$OUT_DIR/single_sharplight_$NAME.txt; : > "$OUT"
    for N in $NS; do for r in $(seq "$REPS"); do
      "$HERE/gpu_check.sh" | head -1 | sed "s/^/[N=$N run=$r] /" >> "$OUT"
      "$CLI" "$MP" --bench "$N" --ctx $((N + 256)) --batch 512 "${EXTRA[@]}" 2>/dev/null \
        | grep -E '^(prefill|segment)' | sed "s/^/[N=$N run=$r] /" >> "$OUT"
    done; done ;;
  llama-*)
    B=${ENG#llama-}; L=${LLAMA_DIR:?set LLAMA_DIR}/runtime.llama-${LLAMA_BUILD:-b11177}-$B
    OUT=$OUT_DIR/single_llama-${B}_$NAME.jsonl
    "$HERE/gpu_check.sh" > "${OUT%.jsonl}.gpu.txt"
    LD_LIBRARY_PATH=$L "$L/llama-bench" -m "$MP" -ngl 99 -fa 1 -p 512,2048 -n 128 -d 0,4096 -r 5 -o jsonl "${EXTRA[@]}" \
      > "$OUT" 2> "${OUT%.jsonl}.err" ;;
  vllm)
    OUT=$OUT_DIR/single_vllm_$NAME.json
    "$HERE/gpu_check.sh" > "${OUT%.json}.gpu.txt"
    "${VLLM_VENV:?set VLLM_VENV}/bin/python" -I "$HERE/vllm_single.py" "$MP" "$NAME" --out "$OUT" "${EXTRA[@]}" \
      > "${OUT%.json}.log" 2>&1 ;;
  *) echo "unknown engine $ENG"; exit 1 ;;
esac
echo "done $ENG $NAME -> $OUT"
