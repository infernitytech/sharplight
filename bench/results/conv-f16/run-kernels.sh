#!/bin/bash
# Kernel-level convolution benchmark (bench/ConvBench, default shapes: SeedVR2's causal VAE with 3 stacked frame windows at 480 x 832, an
# SDXL VAE decoder at 1024 x 1024, ESRGAN, ResNet and a 1 x 1 layer), reduced precision, 10 profiled runs, the sum of each launch's
# minimum GPU time. Run it under the machine's GPU lock (flock). Modes per provider (MODES_CUDA, MODES_VULKAN):
#   base    SHARPLIGHT_CONV_CHANNELS_LAST=0: conv_cm with f32 activations (the state before this round)
#   new     the defaults: conv_cl (channels-last f16 activations; CUDA: asynchronous-copy ring)
#   sync    CUDA: conv_cl with the register prefetch instead of the asynchronous copies (SHARPLIGHT_CONV_ASYNC=0)
#   tileW,X,Y[,S[,K]]  SHARPLIGHT_CONV_CL_TILE (subgroup rows, columns, 16-pixel fragments, stages, K step)
# Environment: BENCH (the ConvBench executable), OUT (results folder), TAG.
set -u
for prov in cuda vulkan; do
  modes_var=MODES_${prov^^}; modes=${!modes_var:-base new}
  for mode in $modes; do
    unset SHARPLIGHT_CONV_CHANNELS_LAST SHARPLIGHT_CONV_ASYNC SHARPLIGHT_CONV_CL_TILE
    case $mode in
      base) export SHARPLIGHT_CONV_CHANNELS_LAST=0 ;;
      sync) export SHARPLIGHT_CONV_ASYNC=0 ;;
      tile*) export SHARPLIGHT_CONV_CL_TILE=${mode#tile} ;;
    esac
    timeout 1800 "$BENCH" --provider $prov --iters 10 > "$OUT/kernels-$TAG-$prov-$mode.txt" 2>&1
  done
done
