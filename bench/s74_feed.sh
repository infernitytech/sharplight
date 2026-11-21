#!/bin/bash
# Feed GGUF models fully on the GPU against llama.cpp (docs/benchmarks.md section 74), one provider at a time, both engines in the same
# call so their runs alternate inside one GPU-lock hold (the caller takes the lock):
#   prefill at LENS (Sharplight: `--bench N`, best and median of 5; llama.cpp: llama-bench pp at depth 0, 3 reps),
#   decode of one stream (Sharplight: the `--bench 512` decode; llama.cpp: tg128),
#   decode at DEPTHS (Sharplight: BatchBench, one row of a two-slot paged decoder sized for one sequence, tokens fed back; llama.cpp: llama-bench tg64 at -d D),
#   batched decode at SEQS sequences (Sharplight: BatchBench `--feedback`, GPU ms per step at depth BDEPTH; llama.cpp:
#   llama-batched-bench -npp BDEPTH -ntg 64, S_TG).
# Dense models compute the same work for any token id, so fixed and fed-back ids measure the same thing here; BatchBench feeds back anyway.
# usage: s74_feed.sh <out dir> <name> <gguf> <cuda|vulkan> [sharplight|llama|both]
# env: LENS ("512 2048 8192 32768"), DEPTHS ("32768"), SEQS ("16 64"), BDEPTH (256), BCTX (BDEPTH + 128), SL_ARGS, LLAMA_ARGS,
#      RUN_TIMEOUT (1800 s), LLAMA_ROOT.
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT=${1:?out dir}; NAME=${2:?name}; MODEL=${3:?gguf}; PROV=${4:?cuda or vulkan}; WHICH=${5:-both}
mkdir -p "$OUT"
LENS=${LENS:-"512 2048 8192 32768"}; DEPTHS=${DEPTHS:-"32768"}; SEQS=${SEQS:-"16 64"}; BDEPTH=${BDEPTH:-256}; BCTX=${BCTX:-$((BDEPTH + 128))}
T=${RUN_TIMEOUT:-1800}
CLI=$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight
BB=$HERE/BatchBench/bin/Release/net11.0/BatchBench
case $PROV in cuda) LB=cuda13 ;; vulkan) LB=vulkan ;; *) echo "provider cuda or vulkan"; exit 2 ;; esac
LDIR=${LLAMA_ROOT:?LLAMA_ROOT}/runtime.llama-b11177-$LB
llama() { local tool=$1; shift; LD_LIBRARY_PATH=$LDIR timeout "$T" "$LDIR/$tool" -m "$MODEL" -ngl 99 -fa 1 "$@"; }
S=$OUT/sl_${NAME}_$PROV.txt; L=$OUT/llama_${NAME}_$PROV.txt
maxseq=$(echo "$SEQS" | tr ' ' '\n' | sort -n | tail -1)
"$HERE/gpu_check.sh" >> "$OUT/gpu_${NAME}_$PROV.txt"
for N in $LENS; do
  if [ "$WHICH" != llama ]; then
    # shellcheck disable=SC2086
    r=$(timeout "$T" "$CLI" "$MODEL" --provider "$PROV" --bench "$N" --ctx $((N + 256)) $SL_ARGS 2>&1 | grep -E '^prefill|rror|xception' | head -3)
    echo "[N=$N] ${r:-failed or timed out}" >> "$S"
  fi
  if [ "$WHICH" != sharplight ]; then
    # shellcheck disable=SC2086
    llama llama-bench -p "$N" -n 0 -r 3 $LLAMA_ARGS 2>&1 | grep -E '^\| .*pp|rror' >> "$L"
  fi
done
if [ "$WHICH" != sharplight ]; then
  # shellcheck disable=SC2086
  llama llama-bench -p 0 -n 128 -r 3 $LLAMA_ARGS 2>&1 | grep -E '^\| .*tg|rror' >> "$L"
fi
for D in $DEPTHS; do
  if [ "$WHICH" != llama ]; then
    # shellcheck disable=SC2086
    r=$(timeout "$T" "$BB" "$MODEL" --provider "$PROV" --slots 2 --ctx $((D + 512)) --kv-tokens $((D + 1024)) --depth "$D" --batch 2048 --sizes 1 --modes greedy \
      --reps 20 --feedback 2>&1 | grep -E '^\| [0-9]|rror|xception' | head -3)
    echo "[decode depth $D] ${r:-failed or timed out}" >> "$S"
  fi
  if [ "$WHICH" != sharplight ]; then
    # shellcheck disable=SC2086
    llama llama-bench -p 0 -n 64 -d "$D" -r 2 $LLAMA_ARGS 2>&1 | grep -E '^\| .*tg|rror' >> "$L"
  fi
done
if [ -n "$SEQS" ]; then
  if [ "$WHICH" != llama ]; then
    # shellcheck disable=SC2086
    r=$(timeout "$T" "$BB" "$MODEL" --provider "$PROV" --slots "$maxseq" --ctx "$BCTX" --depth "$BDEPTH" --sizes "1,$(echo $SEQS | tr ' ' ,)" \
      --modes greedy --reps 20 --feedback 2>&1 | grep -E '^\| [0-9]|rror|xception' | head -6)
    echo "[batched depth $BDEPTH]" >> "$S"; echo "${r:-failed or timed out}" >> "$S"
  fi
  if [ "$WHICH" != sharplight ]; then
    # shellcheck disable=SC2086
    llama llama-batched-bench -c $((maxseq * (BDEPTH + 64))) -b 2048 -ub 512 -npp "$BDEPTH" -ntg 64 -npl "1,$(echo $SEQS | tr ' ' ,)" $LLAMA_ARGS 2>&1 \
      | grep -E '^\|[ ]+[0-9]|rror' >> "$L"
  fi
fi
