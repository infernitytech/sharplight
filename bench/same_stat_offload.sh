#!/bin/bash
# CPU expert offload, same statistic on both engines (llama-bench's): mean ± sd over 5 identical runs of
#   prefill 512 tokens from an empty cache (llama-bench pp512; Sharplight --bench 512, 1 + 4 timed prefills) and
#   decode 64 tokens after a 512-token prompt (llama-bench tg64 @ d512; Sharplight SHARPLIGHT_BENCH_DECODE_RUNS=5).
# Both after one warm-up prompt + one token (llama-bench's warm-up; Sharplight's WarmUp). Page faults sampled (sample.sh).
# env LLAMA_EXTRA: more llama-bench options (e.g. "-lm mmap,none" to give llama.cpp its best of mapped and read-in weights).
# Section "Same-statistic re-measurement vs llama.cpp" of docs/benchmarks.md. The caller takes the GPU lock; env: LLAMA_ROOT.
# usage: same_stat_offload.sh <out dir> <name> <gguf> <cuda|vulkan> <ncmoe list, e.g. 16 or 14,16,18> <tag> [sl-first|llama-first|sl|llama]
OUT=$1; NAME=$2; MODEL=$3; PROV=$4; NCMOE=$5; TAG=$6; ORDER=${7:-sl-first}
HERE="$(cd "$(dirname "$0")" && pwd)"
CLI=$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight
case $PROV in cuda) LB=cuda13 ;; vulkan) LB=vulkan ;; esac
LDIR=${LLAMA_ROOT:?LLAMA_ROOT}/runtime.llama-b11177-$LB
mkdir -p "$OUT"
sl() {
  echo "== $(date -u +%FT%TZ) sharplight $NAME $PROV $TAG" >> "$OUT/sl_${NAME}_$PROV.txt"
  SAMPLE_LOG="$OUT/sl_${NAME}_${PROV}_$TAG.log" $HERE/sample_faults.sh "$OUT/faults_sl_${NAME}_${PROV}_$TAG.txt" env SHARPLIGHT_BENCH_PREFILL_REPEATS=4 SHARPLIGHT_BENCH_DECODE_RUNS=5 \
    "$CLI" "$MODEL" --provider "$PROV" --cpu-offload --bench 512 > "$OUT/sl_${NAME}_${PROV}_$TAG.log" 2>&1
  echo "exit $?" >> "$OUT/sl_${NAME}_${PROV}_$TAG.log"
  grep -E '^prefill|rror|xception' "$OUT/sl_${NAME}_${PROV}_$TAG.log" | head -3 >> "$OUT/sl_${NAME}_$PROV.txt"
}
ll() {
  echo "== $(date -u +%FT%TZ) llama $NAME $PROV $TAG ncmoe $NCMOE" >> "$OUT/llama_${NAME}_$PROV.txt"
  # Two invocations: pp512 from an empty cache, then tg64 at depth 512 (a pp512 @ d512 test needs more device memory and would
  # abort llama-bench at the -ncmoe values where tg64 @ d512 still fits).
  SAMPLE_LOG="$OUT/llama_${NAME}_${PROV}_${TAG}pp.log" $HERE/sample_faults.sh "$OUT/faults_llama_${NAME}_${PROV}_${TAG}pp.txt" env LD_LIBRARY_PATH=$LDIR "$LDIR/llama-bench" -m "$MODEL" -ngl 99 \
    -ncmoe "$NCMOE" -fa 1 -p 512 -n 0 -r 5 -t 10 $LLAMA_EXTRA > "$OUT/llama_${NAME}_${PROV}_${TAG}pp.log" 2>&1
  SAMPLE_LOG="$OUT/llama_${NAME}_${PROV}_$TAG.log" $HERE/sample_faults.sh "$OUT/faults_llama_${NAME}_${PROV}_$TAG.txt" env LD_LIBRARY_PATH=$LDIR "$LDIR/llama-bench" -m "$MODEL" -ngl 99 \
    -ncmoe "$NCMOE" -fa 1 -p 0 -n 64 -d 512 -r 5 -t 10 $LLAMA_EXTRA > "$OUT/llama_${NAME}_${PROV}_$TAG.log" 2>&1
  grep -hE '^\| .*(pp|tg)|rror' "$OUT/llama_${NAME}_${PROV}_${TAG}pp.log" "$OUT/llama_${NAME}_${PROV}_$TAG.log" >> "$OUT/llama_${NAME}_$PROV.txt"
}
"$HERE/gpu_check.sh" >> "$OUT/gpu_${NAME}_$PROV.txt"
case $ORDER in sl-first) sl; ll ;; llama-first) ll; sl ;; sl) sl ;; llama) ll ;; esac
"$HERE/gpu_check.sh" >> "$OUT/gpu_${NAME}_$PROV.txt"
