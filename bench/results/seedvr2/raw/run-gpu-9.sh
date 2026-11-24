#!/bin/bash
# GPU round 9: 65 frames (Concat limit fix).
set -x
cd <repo> && . <tools>/env.sh
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0 SHARPLIGHT_FFMPEG_DIR=<path>
export SEEDVR_REPO=<work>/src-seedvr SHIMS=<work>/shims COLOR_FIX=<work>/color_fix.py
export PYTHON=<work>/venv/bin/python INPUTS=<work>/inputs
export MODEL=<models>/video/seedvr2/SeedVR2-3B OUT=<work>/bench-3
export CASES="fox-240p-65:416:240:832:480:sharplight-only"
timeout 3600 <repo>/bench/results/seedvr2/run-bench.sh
echo ROUND9-DONE
