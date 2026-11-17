#!/bin/bash
# Apples-to-apples prefill runs of docs/benchmarks.md section 38: the same Hugging Face checkpoints (bf16 / FP8) in Sharplight
# (HF importer, Vulkan) and vLLM.
# usage: s38_all.sh <sharplight|vllm> [tag]
#   single stream: sharplight `--bench N --ctx N+256` for N = 16, 512, 2048, 8192, 2 process runs (best prefill of 4, median decode
#                  step); vLLM bench/vllm_single.py --depth 8192 (pp512 / pp2048 / pp8192, tg at depth 16 and 8192, median of 5)
#   serving:       one server (16 slots, 4096-token context), bench/loadgen.py with 2000-word random prompts (2000 tokens), 16 output
#                  tokens, concurrency 1 and 16, 3 requests per worker, 2 rounds; TTFT p50 is the reported figure
# env: MODELS ("name:dir ..." under HF_DIR), HF_DIR (~/models/hf), OUT_DIR (bench/results/s38), TAG (results tag, default "after"),
#      SL_ARGS (extra Sharplight CLI / server args, e.g. "--batch 512"), SRV_ARGS (server only, e.g. "--prefill-chunk 512"),
#      VLLM_VENV (virtualenv with vLLM and httpx), PORT (18093). GPU: SHARPLIGHT_VK_DEVICES / CUDA_VISIBLE_DEVICES.
HERE="$(cd "$(dirname "$0")" && pwd)"
ENG=$1; TAG=${2:-${TAG:-after}}
HF_DIR=${HF_DIR:-$HOME/models/hf}; OUT_DIR=${OUT_DIR:-$HERE/results/s38}; mkdir -p "$OUT_DIR"
MODELS=${MODELS:-"qwen3-0.6b:Qwen3-0.6B llama3.2-1b:Llama-3.2-1B-Instruct qwen3-8b:Qwen3-8B qwen3-8b-fp8:Qwen3-8B-FP8"}
PORT=${PORT:-18093}; PY=${VLLM_VENV:?set VLLM_VENV}/bin/python
export VLLM_USE_FLASHINFER_SAMPLER=0
CLI=$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight
SRV=$HERE/../samples/Sharplight.Cli/bin/Release/net11.0/sharplight

serve() {  # serve <engine label> <name> <server pid>
  local T0=$SECONDS
  until curl -sf "localhost:$PORT/health" >/dev/null 2>&1; do
    kill -0 "$3" 2>/dev/null || return; [ $((SECONDS - T0)) -gt 4000 ] && break; sleep 2
  done
  for c in 1 16; do for r in 1 2; do
    timeout 900 "$PY" -I "$HERE/loadgen.py" --url "http://127.0.0.1:$PORT" --model m -c $c --per-worker 3 --prompt-words 2000 \
      --max-tokens 16 --ignore-eos --seed $r --json "$OUT_DIR/serve_$1_$2_c${c}_r$r.json" | cut -c1-300
  done; done
  kill "$3"; sleep 5; kill -9 "$3" 2>/dev/null; wait "$3" 2>/dev/null
}

for m in $MODELS; do
  n=${m%%:*}; d=$HF_DIR/${m#*:}
  case $ENG in
    sharplight)
      f=$OUT_DIR/single_sl-${TAG}_$n.txt; : > "$f"
      for N in 16 512 2048 8192; do for r in 1 2; do
        "$CLI" "$d" --bench $N --ctx $((N + 256)) $SL_ARGS 2>&1 | grep -E '^prefill|rror|Rows per' | sed "s/^/[N=$N run=$r] /" >> "$f"
      done; done
      "$SRV" serve "$d" --slots 16 --ctx 4096 --listen "http://127.0.0.1:$PORT" --temp 0 $SL_ARGS $SRV_ARGS > "$OUT_DIR/serve_sl-${TAG}_$n.server.log" 2>&1 &
      serve "sl-$TAG" "$n" $! ;;
    vllm)
      "$PY" -I "$HERE/vllm_single.py" "$d" "$n" --gpu-util 0.85 --depth 8192 --max-len 8704 --out "$OUT_DIR/single_vllm_$n.json" \
        > "$OUT_DIR/single_vllm_$n.log" 2>&1
      "$VLLM_VENV/bin/vllm" serve "$d" --served-model-name m --max-model-len 4096 --max-num-seqs 16 --port "$PORT" \
        --gpu-memory-utilization 0.85 --no-enable-prefix-caching > "$OUT_DIR/serve_vllm_$n.server.log" 2>&1 &
      serve vllm "$n" $! ;;
    *) echo "unknown engine $ENG"; exit 1 ;;
  esac
done
echo "done $ENG"
