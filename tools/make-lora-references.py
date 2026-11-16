#!/usr/bin/env python3
"""Writes diffusers reference images of a real LoRA on an SDXL or SD 1.x single-file checkpoint for RealLoraTests.

The base is loaded with from_single_file (configuration from the base model's diffusers folder), the LoRA with load_lora_weights at
--scale, sampled with the configuration's Euler from seeded latents in f32. Writes OUT/reference/{noise,image,image_base}.f32 and
index.json (the base path relative to ~/models, the LoRA's file name in OUT).

Usage: make-lora-references.py OUT BASE(relative to ~/models) LORA [--family sdxl] [--scale 1] [--steps 8] [--guidance 1] [--size 1024] [--device cuda]
"""

import argparse
import json
import os

import torch

PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"


def save(folder, name, tensor, index):
    os.makedirs(folder, exist_ok=True)
    tensor.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(os.path.join(folder, name + ".f32"))
    index[name] = list(tensor.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("base")
    parser.add_argument("lora")
    parser.add_argument("--family", default="sdxl")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline

    xl = args.family == "sdxl"
    config = args.config or os.path.expanduser("~/models/sd/sdxl-diffusers" if xl else "~/models/sd/sd15-diffusers")
    base = os.path.join(os.path.expanduser("~/models"), args.base)
    pipe = (StableDiffusionXLPipeline if xl else StableDiffusionPipeline).from_single_file(base, config=config, torch_dtype=torch.float32)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(args.device)
    noise = torch.randn(1, 4, args.size // 8, args.size // 8, generator=torch.Generator().manual_seed(42))
    folder = os.path.join(args.out, "reference")
    index = {}
    save(folder, "noise", noise, index)
    kwargs = dict(height=args.size, width=args.size, num_inference_steps=args.steps, guidance_scale=args.guidance, output_type="pt")
    save(folder, "image_base", pipe(PROMPT, latents=noise.clone().to(args.device), **kwargs).images[0], index)
    pipe.load_lora_weights(os.path.dirname(os.path.abspath(args.lora)), weight_name=os.path.basename(args.lora), adapter_name="lora")
    pipe.set_adapters(["lora"], adapter_weights=[args.scale])
    save(folder, "image", pipe(PROMPT, latents=noise.clone().to(args.device), **kwargs).images[0], index)
    with open(os.path.join(folder, "index.json"), "w") as f:
        json.dump({"tensors": index, "base": args.base, "lora": os.path.basename(args.lora), "family": args.family, "scale": args.scale,
                   "steps": args.steps, "guidance": args.guidance, "size": args.size, "prompt": PROMPT, "sampler": "euler"}, f, indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()
