#!/usr/bin/env python3
"""Writes diffusers reference outputs of the tiny autoencoders (TAESD) for the Sharplight image-generation tests.

  taesd, taesdxl, taesd3, taef1   madebyollin's AutoencoderTiny folders (config.json + diffusion_pytorch_model.safetensors)

Usage: make-taesd-references.py TAE_ROOT OUT [--size 512] [--device cuda] [--sd15 SD15_FOLDER]

For every tiny autoencoder folder under TAE_ROOT (written to OUT/<name>, raw little-endian f32 plus index.json):

  latent        seeded unit Gaussian model latents [1, C, size/8, size/8]
  decoded       AutoencoderTiny.decode(latent) as the pipelines post-process it: [3, size, size] in [0, 1]
  image         a smooth test picture [3, size, size] in [0, 1]
  encoded       AutoencoderTiny.encode(2 * image - 1) (the model latents) [1, C, size/8, size/8]

With --sd15: StableDiffusionPipeline (float32, Euler, 20 steps, guidance 7.5) with the TAESD as its VAE from shared initial noise, in
OUT/sd15-taesd: noise [1, 4, size/8, size/8] and the image [3, size, size].
"""

import argparse
import json
import math
import os

import numpy as np
import torch


def write(out, name, tensor):
    array = tensor.detach().to("cpu", torch.float32).contiguous().numpy()
    array.tofile(os.path.join(out, name + ".f32"))
    return list(array.shape)


def picture(size):
    y, x = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    r = 0.5 + 0.5 * torch.sin(6 * math.pi * x) * torch.cos(4 * math.pi * y)
    g = (x + y) / 2
    b = ((x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.1).float() * 0.8 + 0.1
    return torch.stack([r, g, b])[None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("out")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sd15")
    args = parser.parse_args()

    from diffusers import AutoencoderTiny

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    size = args.size
    for name in sorted(os.listdir(args.root)):
        folder = os.path.join(args.root, name)
        if not os.path.isfile(os.path.join(folder, "config.json")):
            continue
        vae = AutoencoderTiny.from_pretrained(folder, torch_dtype=torch.float32).to(args.device).eval()
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        channels = vae.config.latent_channels
        generator = torch.Generator("cpu").manual_seed(7)
        latent = torch.randn(1, channels, size // 8, size // 8, generator=generator)
        image = picture(size)
        with torch.no_grad():
            # The pipelines' decode: vae.decode(latents / scaling_factor + shift_factor), then (x / 2 + 0.5).clamp(0, 1).
            z = latent.to(args.device) / vae.config.scaling_factor + vae.config.shift_factor
            decoded = (vae.decode(z).sample / 2 + 0.5).clamp(0, 1)
            # The pipelines' encode: (vae.encode(2 x - 1) - shift_factor) * scaling_factor.
            encoded = (vae.encode(image.to(args.device) * 2 - 1).latents - vae.config.shift_factor) * vae.config.scaling_factor
        index = {"size": size, "latent_channels": channels, "tensors": {}}
        index["tensors"]["latent"] = write(out, "latent", latent)
        index["tensors"]["decoded"] = write(out, "decoded", decoded[0])
        index["tensors"]["image"] = write(out, "image", image[0])
        index["tensors"]["encoded"] = write(out, "encoded", encoded)
        with open(os.path.join(out, "index.json"), "w") as f:
            json.dump(index, f, indent=2)
        print(name, "written")

    if args.sd15:
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

        tiny = AutoencoderTiny.from_pretrained(os.path.join(args.root, "taesd"), torch_dtype=torch.float32)
        from diffusers import UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        # The folder may lack model_index.json (partial downloads): the pipeline is assembled from its components.
        pipe = StableDiffusionPipeline(
            vae=tiny,
            text_encoder=CLIPTextModel.from_pretrained(args.sd15, subfolder="text_encoder", torch_dtype=torch.float32),
            tokenizer=CLIPTokenizer.from_pretrained(args.sd15, subfolder="tokenizer"),
            unet=UNet2DConditionModel.from_pretrained(args.sd15, subfolder="unet", torch_dtype=torch.float32),
            scheduler=EulerDiscreteScheduler.from_pretrained(args.sd15, subfolder="scheduler"),
            safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(args.device)
        out = os.path.join(args.out, "sd15-taesd")
        os.makedirs(out, exist_ok=True)
        prompt = "a photograph of an astronaut riding a horse"
        steps = 20
        guidance = 7.5
        noise = torch.randn(1, 4, size // 8, size // 8, generator=torch.Generator("cpu").manual_seed(42))
        with torch.no_grad():
            image = pipe(prompt, height=size, width=size, num_inference_steps=steps, guidance_scale=guidance,
                         latents=noise.to(args.device), output_type="pt").images[0]
        index = {"size": size, "steps": steps, "guidance_scale": guidance, "prompt": prompt, "tensors": {}}
        index["tensors"]["noise"] = write(out, "noise", noise)
        index["tensors"]["image"] = write(out, "image", image)
        with open(os.path.join(out, "index.json"), "w") as f:
            json.dump(index, f, indent=2)
        print("sd15-taesd written")


if __name__ == "__main__":
    main()
