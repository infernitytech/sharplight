#!/bin/bash
# SeedVR2 7B and 7B sharp, official ByteDance-Seed/SeedVR2-7B files (DiT .pth f32 + ema_vae.pth; text embeddings of the official
# release, identical in the 3B repo and the GitHub repo): official PyTorch inference vs Sharplight CUDA / Vulkan, end-to-end parity
# (Sharplight on PyTorch's decoded frames and noise) and timings. Run under the GPU lock.
set -x
cd <repo> && . <env.sh>
export CUDA_VISIBLE_DEVICES=0 SHARPLIGHT_VK_DEVICES=0 GGML_VK_VISIBLE_DEVICES=0 SHARPLIGHT_FFMPEG_DIR=<ffmpeg>
export SEEDVR_REPO=<work>/src-seedvr SHIMS=<work>/shims COLOR_FIX=<work>/color_fix.py
export PYTHON=<work>/venv/bin/python INPUTS=<work>/inputs RUNS=${RUNS:-3}
export ARCH=7b DIT_WEIGHTS=bf16 CASES=${CASES:-"fox-240p-17:416:240:832:480"}
for variant in 7b 7b_sharp; do
  folder=<models>/video/seedvr2/SeedVR2-7B; [ $variant = 7b_sharp ] && folder=<models>/video/seedvr2/SeedVR2-7B-sharp
  export MODEL=$folder DIT=$folder/seedvr2_ema_$variant.pth VAE=$folder/ema_vae.pth TAG=${variant/_/-}-official OUT=<work>/bench
  timeout 5400 <repo>/bench/results/seedvr2/run-bench.sh
done
echo GPU-DONE
