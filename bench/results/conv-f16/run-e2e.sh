#!/bin/bash
# SeedVR2 3B end to end with bench/VideoEnhancementBench, the procedure of benchmarks.md section 108 (median of 5 runs after one warm-up,
# --reduced-precision), base (SHARPLIGHT_CONV_CHANNELS_LAST=0) and new, CUDA and Vulkan; before each, the parity run against the official
# bf16 run `fox-480p-cuda-bf16` (17 frames 416 x 240 -> 832 x 480, its input frames and noise; worst-frame PSNR with ../seedvr2/psnr.py).
# Environment: BENCH (VideoEnhancementBench), MODEL (SeedVR2-3B folder), INPUTS (fox-240p-33.mp4, fox-360p-33.mp4), REFERENCE (the
# reference dump), PYTHON (with numpy), OUT, TAG. Under the GPU lock.
set -u
here=$(cd "$(dirname "$0")" && pwd)
for prov in cuda vulkan; do
  for mode in base new; do
    unset SHARPLIGHT_CONV_CHANNELS_LAST
    [ $mode = base ] && export SHARPLIGHT_CONV_CHANNELS_LAST=0
    log="$OUT/$TAG-parity-$prov-$mode.log"
    timeout 1800 "$BENCH" --model "$MODEL" --provider $prov --input "$REFERENCE/input_frames.u8" --input-shape 17x240x416 --width 832 --height 480 \
      --runs 1 --warmup 0 --reduced-precision --noise "$REFERENCE" --output-u8 "$OUT/parity.u8" > "$log" 2>&1
    "$PYTHON" "$here/../seedvr2/psnr.py" "$OUT/parity.u8" "$REFERENCE/output.u8" 832 480 >> "$log" 2>&1
    rm -f "$OUT/parity.u8"
    for case in fox-240p-33:832:480 fox-360p-33:1280:720; do
      IFS=: read -r clip w h <<< "$case"
      timeout 3600 "$BENCH" --model "$MODEL" --provider $prov --input "$INPUTS/$clip.mp4" --width $w --height $h --runs 5 --warmup 1 \
        --reduced-precision --json "$OUT/$TAG-$clip-$prov-$mode.json" > "$OUT/$TAG-$clip-$prov-$mode.log" 2>&1
    done
  done
done
