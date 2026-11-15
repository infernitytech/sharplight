#!/bin/bash
# Generators that execute a model repository's own modeling code (trust_remote_code): run on tiny random-weight configs only, in a venv of their
# own, with an empty environment (no Hub / GitHub tokens) and offline flags. Download the code first (phase 1, with a network):
#   python stage_remote.py $STAGE ; and the code directories of Kimi-K3, K2-Horizon-36B, Xing4.0-29B-A4B, clef-flash, Decision-2.0-Kai-0.6B,
#   THUDM/cogvlm-chat-hf (+ lmsys/vicuna-7b-v1.5 tokenizer files) as the headers of the mk_tiny_* scripts describe.
#   STAGE=<stage dir> SRC=<dir with the code directories> MODELS=~/models VENV=<venv> ./run_remote_sandboxed.sh
MODELS=${MODELS:-$HOME/models}; HF=$MODELS/hf; T="$(cd "$(dirname "$0")/../../tests/Sharplight.Families.Tests/tools" && pwd)"
SANDBOX_HOME=$(mktemp -d)
run() { n=$1; shift; echo "== $n"
  ( cd "$T" && env -i PATH=/usr/bin:/bin HOME=$SANDBOX_HOME TMPDIR=$SANDBOX_HOME HF_HOME=$SANDBOX_HOME/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES= TINY_SOURCE_DIR=$STAGE "$VENV/bin/python" "$@" ) > "$SANDBOX_HOME/$n.log" 2>&1 && echo "OK $n" || echo "FAIL $n (log: $SANDBOX_HOME/$n.log)"; }
run kimi_k3 mk_tiny_kimi_k3.py "$HF" "$SRC/Kimi-K3"
run k2_horizon mk_tiny_k2_horizon.py "$HF" "$SRC/K2-Horizon-36B"
run xing4_0 mk_tiny_xing4_0.py "$HF" "$SRC/Xing4.0-29B-A4B"
run xing4_0_unclamped mk_tiny_xing4_0.py "$HF" "$SRC/Xing4.0-29B-A4B" --unclamped
run clef mk_tiny_clef.py "$HF" "$SRC/clef-flash"
run vllm_decision2 mk_tiny_vllm_decision2.py "$HF" "$SRC/Decision-2.0-Kai-0.6B"
run cogvlm mk_tiny_cogvlm.py "$SRC/cogvlm-chat-hf" "$SRC/vicuna-7b-v1.5" "$HF/tiny-random-cogvlm"
for f in lfm3 spark2_5 clover_moe mimo_v2 llava_onevision2; do run remote_$f mk_tiny_remote_code.py $f "$HF/tiny-random-$f"; done
