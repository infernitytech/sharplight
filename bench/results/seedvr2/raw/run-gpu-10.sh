#!/bin/bash
# GPU round 10: PyTorch at 960p and 1088p with the current harness (round 1's failures were its video reader), all three settings.
set -x
cd <repo> && . <tools>/env.sh
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0 SHARPLIGHT_FFMPEG_DIR=<path>
export SEEDVR_REPO=<work>/src-seedvr SHIMS=<work>/shims COLOR_FIX=<work>/color_fix.py
export PYTHON=<work>/venv/bin/python INPUTS=<work>/inputs
export MODEL=<models>/video/seedvr2/SeedVR2-3B OUT=<work>/bench-5 PROVIDERS=" "
export CASES="fox-480p-33:832:480:1664:960 fox-544p-33:960:544:1920:1088"
timeout 5400 <repo>/bench/results/seedvr2/run-bench.sh
echo ROUND10-DONE
