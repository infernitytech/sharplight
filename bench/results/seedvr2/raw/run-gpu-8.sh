#!/bin/bash
# GPU round 8: the final Sharplight runs with the constant-memory VAE (PyTorch's results of rounds 3 and 4 kept; its dumps give the parity).
set -x
cd <repo> && . <tools>/env.sh
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0 SHARPLIGHT_FFMPEG_DIR=<path>
B=<repo>/bench/VideoEnhancementBench/bin/Release/net11.0/VideoEnhancementBench
M=<models>/video/seedvr2/SeedVR2-3B
L=<work>/logs/memory-8.log
: > $L
for size in 33x480x832 65x480x832 33x720x1280 33x960x1664 33x1088x1920; do
  echo "== $size" >> $L
  $B --model $M --provider cuda --reduced-precision --memory $size --graphs vae_encode,vae_decode 2>/dev/null | grep -E "GiB|MiB|memory" >> $L
done
$B --model $M --provider cuda --reduced-precision --profile 33x480x832 > <work>/logs/profile-cuda-480p-8.log 2>&1
export SEEDVR_REPO=<work>/src-seedvr SHIMS=<work>/shims COLOR_FIX=<work>/color_fix.py
export PYTHON=<work>/venv/bin/python INPUTS=<work>/inputs
export MODEL=$M OUT=<work>/bench-3
export CASES="fox-240p-33:416:240:832:480:sharplight-only fox-240p-65:416:240:832:480:sharplight-only fox-360p-33:640:360:1280:720:sharplight-only fox-480p-33:832:480:1664:960:sharplight-only fox-544p-33:960:544:1920:1088:sharplight-only"
timeout 7200 <repo>/bench/results/seedvr2/run-bench.sh
M=<models>/video/seedvr2/SeedVR2-7B-fp16
export MODEL=$M ARCH=7b DIT=$M/seedvr2_ema_7b_fp16.safetensors DIT_WEIGHTS=bf16 VAE=<models>/video/seedvr2/SeedVR2-3B/ema_vae.pth TAG=7b OUT=<work>/bench-4
export CASES="fox-240p-17:416:240:832:480:sharplight-only"
timeout 3600 <repo>/bench/results/seedvr2/run-bench.sh
echo ROUND8-DONE
