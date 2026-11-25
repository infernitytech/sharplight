#!/bin/bash
# SeedVR2 benchmark: the official PyTorch inference (tools/seedvr2-reference.py) against Sharplight on CUDA and Vulkan, same checkpoint,
# input clip, frame count, sizes and colour correction; median of 5 runs after one warm-up, end to end (read the input file, enhance,
# write the output file) with the per-stage split, and the peak device memory of the process (nvidia-smi, 100 ms samples).
# Run it under the machine's GPU lock (flock). Environment:
#   SEEDVR_REPO   checkout of github.com/ByteDance-Seed/SeedVR      SHIMS      apex / flash_attn stand-ins (see tools/seedvr2-reference.py)
#   COLOR_FIX     color_fix.py of the official Space                 PYTHON     python with torch (CUDA), einops, omegaconf, diffusers 0.29
#   MODEL         folder with the DiT, the VAE, pos_emb.pt and neg_emb.pt (Sharplight loads the folder)
#   ARCH          3b (default) or 7b; DIT / VAE the files PyTorch loads (default: the official 3B names in MODEL);
#   DIT_WEIGHTS   fp32 (default, official) or bf16 (the 7B on a 32 GB GPU); TAG prefixes the result names (default: ARCH)
#   INPUTS        folder with the clips (<name>.mp4 and <name>.u8)    OUT        results folder
#   CASES         "name:inW:inH:outW:outH[:sharplight-only] ..." (default below; sharplight-only skips PyTorch, e.g. sizes it cannot fit)
#   PROVIDERS     "cuda vulkan" (default)
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
bench="$repo/bench/VideoEnhancementBench/bin/Release/net11.0/VideoEnhancementBench"
CASES=${CASES:-"fox-480p-33:832:480:1664:960 fox-544p-33:960:544:1920:1088"}
PROVIDERS=${PROVIDERS:-"cuda vulkan"}
RUNS=${RUNS:-5}
ARCH=${ARCH:-3b}
DIT=${DIT:-$MODEL/seedvr2_ema_3b.pth}
VAE=${VAE:-$MODEL/ema_vae.pth}
DIT_WEIGHTS=${DIT_WEIGHTS:-fp32}
TAG=${TAG:-$ARCH}
mkdir -p "$OUT"
for case in $CASES; do
    IFS=: read -r clip inW inH outW outH only <<< "$case"
    name="$TAG-$clip"
    if [ "$only" != "sharplight-only" ]; then
    log="$OUT/$name-pytorch.log"
    : > "$log"
    # The fastest setting that fits: DiT and VAE resident together, else the official script's offloading between the stages, else (the
    # official f32 weights plus activations exceed 32 GB at large sizes) the DiT's weights held in bf16, which autocast multiplies in anyway.
    for setting in "--no-offload|$DIT_WEIGHTS" "|$DIT_WEIGHTS" "|bf16"; do
        IFS='|' read -r offload weights <<< "$setting"
        "$here/peak-memory.sh" "$log" "$PYTHON" "$repo/tools/seedvr2-reference.py" --repo "$SEEDVR_REPO" --shims "$SHIMS" --color-fix-file "$COLOR_FIX" \
            --model "$ARCH" --dit "$DIT" --dit-weights "$weights" --vae "$VAE" --pos-emb "$MODEL/pos_emb.pt" --neg-emb "$MODEL/neg_emb.pt" \
            --input "$INPUTS/$clip.mp4" --res-h "$outH" --res-w "$outW" --device cuda --dit-dtype bf16 --vae-dtype bf16 --attention flash $offload \
            --repeat "$RUNS" --seed 42 --timing-json "$OUT/$name-pytorch.json" --save-video "$OUT/$name-pytorch.mp4" --out "$OUT/$name-pytorch-dump" >> "$log" 2>&1 \
            && { echo "pytorch setting: offload '${offload:-official}', DiT weights $weights" >> "$log"; break; }
        echo "pytorch run with offload '${offload:-official}', DiT weights $weights failed" >> "$log"
    done
    fi
    for provider in $PROVIDERS; do
        # Correctness first (when PyTorch ran): the frames PyTorch decoded and its noise, compared with its output.
        if [ -f "$OUT/$name-pytorch-dump/index.json" ]; then
        frames=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['tensors']['input_frames'][0])" "$OUT/$name-pytorch-dump/index.json")
        "$bench" --model "$MODEL" --provider "$provider" --input "$OUT/$name-pytorch-dump/input_frames.u8" --input-shape "${frames}x${inH}x${inW}" \
            --width "$outW" --height "$outH" --runs 1 --warmup 0 --reduced-precision --noise "$OUT/$name-pytorch-dump" \
            --output-u8 "$OUT/$name-sharplight-$provider-parity.u8" > "$OUT/$name-sharplight-$provider-parity.log" 2>&1
        "$PYTHON" "$here/psnr.py" "$OUT/$name-sharplight-$provider-parity.u8" "$OUT/$name-pytorch-dump/output.u8" "$outW" "$outH" \
            >> "$OUT/$name-sharplight-$provider-parity.log"
        fi
        log="$OUT/$name-sharplight-$provider.log"
        : > "$log"
        "$here/peak-memory.sh" "$log" "$bench" --model "$MODEL" --provider "$provider" --input "$INPUTS/$clip.mp4" --width "$outW" --height "$outH" \
            --runs "$RUNS" --warmup 1 --reduced-precision --json "$OUT/$name-sharplight-$provider.json" --output-video "$OUT/$name-sharplight-$provider.mp4" \
            --output-u8 "$OUT/$name-sharplight-$provider.u8" >> "$log" 2>&1
    done
done
