#!/usr/bin/env python3
"""Diffusers references of a real LTX-Video 0.9.0 pipeline for the video-generation tests.

Runs the diffusers LTXPipeline in f32 on CUDA with explicit initial latents (components moved to the GPU only while they run) and
writes, to OUT/reference: text_hidden / negative_hidden (the unmasked T5 states) with their masks, noise [C, T, h, w],
transformer_out (one conditional evaluation on the noise at the first timestep), video [F, 3, H, W] in [0, 1], index.json.

Usage: make-ltx-references.py MODEL_FOLDER OUT [--frames 25] [--height 480] [--width 704] [--steps 20]
Requires torch (CUDA), diffusers, transformers.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

PROMPT = "A red fox trotting through fresh snow in a pine forest, morning light, cinematic"
NEGATIVE = "blurry, low quality, distorted"


def save(folder, index, name, tensor):
    array = tensor.detach().to(torch.float32).cpu().numpy()
    array.astype("<f4").tofile(os.path.join(folder, name + ".f32"))
    index[name] = list(array.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from diffusers import LTXPipeline

    torch.set_grad_enabled(False)
    folder = os.path.join(args.out, "reference")
    os.makedirs(folder, exist_ok=True)
    index = {}
    pipe = LTXPipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    pipe.enable_model_cpu_offload()
    embeds, mask, negative, negative_mask = pipe.encode_prompt(PROMPT, NEGATIVE, do_classifier_free_guidance=True, max_sequence_length=128)
    save(folder, index, "text_hidden", embeds[0])
    save(folder, index, "text_mask", mask[0].float())
    save(folder, index, "negative_hidden", negative[0])
    save(folder, index, "negative_mask", negative_mask[0].float())

    z = pipe.transformer.config.in_channels
    latent_frames = (args.frames - 1) // 8 + 1
    lh, lw = args.height // 32, args.width // 32
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(1, z, latent_frames, lh, lw, generator=generator)
    save(folder, index, "noise", noise[0])

    transformer = pipe.transformer.to("cuda")
    out = transformer(pipe._pack_latents(noise).cuda(), embeds.cuda(), torch.tensor([1000.0]).cuda(), mask.cuda().float(),
                      num_frames=latent_frames, height=lh, width=lw, rope_interpolation_scale=(8 / 25, 32, 32), return_dict=False)[0]
    save(folder, index, "transformer_out", pipe._unpack_latents(out, latent_frames, lh, lw)[0])
    transformer.to("cpu")
    torch.cuda.empty_cache()

    start = time.time()
    video = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, height=args.height, width=args.width, num_frames=args.frames, frame_rate=25,
                 num_inference_steps=args.steps, guidance_scale=args.guidance, latents=pipe._pack_latents(noise.clone()), output_type="np",
                 max_sequence_length=128).frames[0]
    elapsed = time.time() - start
    save(folder, index, "video", torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2))
    with open(os.path.join(folder, "index.json"), "w") as f:
        json.dump({"tensors": index, "prompt": PROMPT, "negative_prompt": NEGATIVE, "frames": args.frames, "height": args.height,
                   "width": args.width, "steps": args.steps, "guidance_scale": args.guidance, "seed": args.seed, "seconds": elapsed,
                   "transformer_timestep": 1000.0}, f, indent=1)
    print(f"wrote {args.out} ({elapsed:.1f} s for the pipeline)")


if __name__ == "__main__":
    main()
