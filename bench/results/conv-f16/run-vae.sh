#!/bin/bash
# Image-generation VAE decoders at 1024 x 1024 (ConvBench --vae: one warm-up, median of 5 runs, the per-kernel profile of one more run, and
# --check: the reduced-precision image against the same decoder with f32 products), base (SHARPLIGHT_CONV_CHANNELS_LAST=0) and new.
# Environment: BENCH (the ConvBench executable), MODELS (folders of diffusers pipelines), OUT, TAG. Under the GPU lock.
set -u
for prov in cuda vulkan; do
  for mode in base new; do
    unset SHARPLIGHT_CONV_CHANNELS_LAST
    [ $mode = base ] && export SHARPLIGHT_CONV_CHANNELS_LAST=0
    for m in $MODELS; do
      timeout 1200 "$BENCH" --provider $prov --vae "$m" --size 1024 --runs 5 --top 40 --check > "$OUT/vae-$TAG-$(basename "$m")-$prov-$mode.txt" 2>&1
    done
  done
done
