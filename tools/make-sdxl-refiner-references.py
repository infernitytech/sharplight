#!/usr/bin/env python3
"""Writes diffusers reference outputs of the SDXL base + refiner ensemble for the Sharplight image-generation tests.

  base     stabilityai/stable-diffusion-xl-base-1.0 (StableDiffusionXLPipeline, denoising_end)
  refiner  stabilityai/stable-diffusion-xl-refiner-1.0 (StableDiffusionXLImg2ImgPipeline: OpenCLIP bigG only, aesthetic-score
           conditioning, denoising_start on the base's latents)

Usage: make-sdxl-refiner-references.py BASE REFINER OUT [--size 1024] [--steps 20] [--switch 0.8] [--device cuda] [--variant fp16]

Everything runs in float32 (fp16 files are upcast on load) with the pipelines' Euler schedulers from shared initial noise. Written to
OUT as raw little-endian f32 plus index.json:

  noise                    unit Gaussian noise [1, 4, H/8, W/8] (the base's initial latents before init_noise_sigma)
  text_hidden / text_pooled  the refiner's prompt embeddings (penultimate layer of OpenCLIP bigG [77, 1280]) and pooled embedding [1280]
  unet_latent/_timestep/_context/_pooled/_time_ids/_out  one refiner UNet evaluation on seeded random inputs (batch 2; time ids
                           [orig_h, orig_w, crop_top, crop_left, aesthetic score] as the pipeline makes them)
  base_latents             the base's latents at the switch (its output_type="latent")
  image                    the ensemble's image (the refiner's decode) [3, H, W] in [0, 1]
"""

import argparse
import json
import os

import numpy as np
import torch


def write(out, name, tensor):
    array = tensor.detach().to("cpu", torch.float32).contiguous().numpy()
    array.tofile(os.path.join(out, name + ".f32"))
    return list(array.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("refiner")
    parser.add_argument("out")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--switch", type=float, default=0.8)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", default="fp16")
    args = parser.parse_args()

    from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    size, steps = args.size, args.steps
    prompt = "a majestic lion jumping from a big stone at night"
    negative = "blurry, low quality"
    variant = args.variant or None
    base = StableDiffusionXLPipeline.from_pretrained(args.base, torch_dtype=torch.float32, variant=variant).to(args.device)
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(args.refiner, torch_dtype=torch.float32, variant=variant).to(args.device)
    index = {"size": size, "steps": steps, "switch": args.switch, "guidance_scale": args.guidance, "prompt": prompt, "negative_prompt": negative,
             "sampler": base.scheduler.__class__.__name__, "tensors": {}}
    shapes = index["tensors"]

    generator = torch.Generator().manual_seed(11)
    noise = torch.randn(1, 4, size // 8, size // 8, generator=generator)
    shapes["noise"] = write(args.out, "noise", noise)

    with torch.no_grad():
        # The refiner's text encoder alone (its only encoder: OpenCLIP bigG with the projection).
        embeds, _, pooled, _ = refiner.encode_prompt(prompt, device=args.device, num_images_per_prompt=1, do_classifier_free_guidance=False)
        shapes["text_hidden"] = write(args.out, "text_hidden", embeds[0])
        shapes["text_pooled"] = write(args.out, "text_pooled", pooled[0])

        # One refiner UNet evaluation on random inputs.
        g = torch.Generator().manual_seed(5)
        unet_latent = torch.randn(2, 4, size // 8, size // 8, generator=g)
        context = torch.randn(2, 77, 1280, generator=g) * 0.5
        unet_pooled = torch.randn(2, 1280, generator=g) * 0.5
        time_ids = torch.tensor([[size, size, 0, 0, 2.5], [size, size, 0, 0, 6.0]], dtype=torch.float32)
        timestep = torch.tensor([700.0, 250.0])
        out = refiner.unet(unet_latent.to(args.device), timestep.to(args.device), encoder_hidden_states=context.to(args.device),
                           added_cond_kwargs={"text_embeds": unet_pooled.to(args.device), "time_ids": time_ids.to(args.device)}).sample
        for name, tensor in [("unet_latent", unet_latent), ("unet_timestep", timestep), ("unet_context", context), ("unet_pooled", unet_pooled),
                             ("unet_time_ids", time_ids), ("unet_out", out)]:
            shapes[name] = write(args.out, name, tensor)

        latents = base(prompt=prompt, negative_prompt=negative, height=size, width=size, num_inference_steps=steps, guidance_scale=args.guidance,
                       denoising_end=args.switch, latents=noise.to(args.device), output_type="latent").images
        shapes["base_latents"] = write(args.out, "base_latents", latents[0])
        image = refiner(prompt=prompt, negative_prompt=negative, image=latents, num_inference_steps=steps, guidance_scale=args.guidance,
                        denoising_start=args.switch, output_type="pt").images
        shapes["image"] = write(args.out, "image", image[0].clamp(0, 1))

    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
