#!/bin/bash
# Batched decode, same statistic on both engines: per batch size a warm-up pass and a timed pass of 64 decode steps at depth 256..319,
# each pass starting from a cleared cache prefilled to 256 (llama-batched-bench -npp 256 -ntg 64, the npl list given twice; BatchBench
# --refill --warmup 64 --reps 64). Statistic: mean step time over the 64 timed steps (S_TG = sequences x 64 / t_tg).
# Section "Same-statistic re-measurement vs llama.cpp" of docs/benchmarks.md. The caller takes the GPU lock; env: LLAMA_ROOT.
# usage: same_stat_batched.sh <out dir> <name> <gguf> <cuda|vulkan> <sizes, e.g. 1,16,32,48,64> <round tag> [sl-first|llama-first] [llama-per-size]
OUT=$1; NAME=$2; MODEL=$3; PROV=$4; SIZES=$5; TAG=$6; ORDER=${7:-sl-first}; PERSIZE=${8:-}
HERE="$(cd "$(dirname "$0")" && pwd)"
BB=$HERE/BatchBench/bin/Release/net11.0/BatchBench
case $PROV in cuda) LB=cuda13 ;; vulkan) LB=vulkan ;; esac
LDIR=${LLAMA_ROOT:?LLAMA_ROOT}/runtime.llama-b11177-$LB
mkdir -p "$OUT"
# Gemma 4 31B does not fit 32 sequences on either engine (32 GB card): Sharplight times 1 and 16 (llama.cpp tries 32 and fails).
SL_SIZES=$SIZES; [ "$NAME" = gemma4-31b ] && SL_SIZES=1,16
maxseq=$(echo "$SL_SIZES" | tr ',' '\n' | sort -n | tail -1)
sl() {
  echo "== $(date -u +%FT%TZ) sharplight $NAME $PROV $TAG" >> "$OUT/sl_${NAME}_$PROV.txt"
  timeout 1800 "$BB" "$MODEL" --provider "$PROV" --slots "$maxseq" --ctx 512 --depth 256 --sizes "$SL_SIZES" \
    --modes greedy,download --warmup 64 --reps 64 --refill --feedback --dump-steps "$OUT/steps_${NAME}_${PROV}_$TAG.csv" \
    >> "$OUT/sl_${NAME}_$PROV.txt" 2>"$OUT/sl_${NAME}_${PROV}_$TAG.err"
  echo "exit $?" >> "$OUT/sl_${NAME}_$PROV.txt"
}
ll() {
  echo "== $(date -u +%FT%TZ) llama $NAME $PROV $TAG" >> "$OUT/llama_${NAME}_$PROV.txt"
  if [ -n "$PERSIZE" ]; then
    for n in $(echo "$SIZES" | tr ',' ' '); do
      LD_LIBRARY_PATH=$LDIR timeout 1800 "$LDIR/llama-batched-bench" -m "$MODEL" -ngl 99 -fa 1 -c $((n * 320)) -b 2048 -ub 512 \
        -npp 256 -ntg 64 -npl "$n,$n" 2>&1 | grep -E '^\|[ ]+[0-9]|rror|failed' >> "$OUT/llama_${NAME}_$PROV.txt"
    done
  else
    lmax=$(echo "$SIZES" | tr ',' '\n' | sort -n | tail -1)
    LD_LIBRARY_PATH=$LDIR timeout 1800 "$LDIR/llama-batched-bench" -m "$MODEL" -ngl 99 -fa 1 -c $((lmax * 320)) -b 2048 -ub 512 \
      -npp 256 -ntg 64 -npl "$SIZES,$SIZES" 2>&1 | grep -E '^\|[ ]+[0-9]|rror|failed' >> "$OUT/llama_${NAME}_$PROV.txt"
  fi
}
"$HERE/gpu_check.sh" >> "$OUT/gpu_${NAME}_$PROV.txt"
case $ORDER in sl-first) sl; ll ;; llama-first) ll; sl ;; sl) sl ;; llama) ll ;; esac
"$HERE/gpu_check.sh" >> "$OUT/gpu_${NAME}_$PROV.txt"
