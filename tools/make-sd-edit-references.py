#!/usr/bin/env python3
"""Writes diffusers reference outputs of the Stable Diffusion editing pipelines for the Sharplight image-generation tests.

  instruct-pix2pix  timbrooks/instruct-pix2pix (StableDiffusionInstructPix2PixPipeline: 8-channel UNet, text and image guidance)
  inpaint           stable-diffusion-v1-5/stable-diffusion-inpainting (StableDiffusionInpaintPipeline: 9-channel UNet)

Usage: make-sd-edit-references.py KIND MODEL OUT [--size 512] [--steps 20] [--device cuda]

Everything runs in float32 with deterministic samplers (Euler for InstructPix2Pix, whose own ancestral sampler draws fresh noise each
step; the inpainting model's PNDM) from shared initial noise. The VAE posterior is taken at its mode (diffusers samples it for the
masked image of the inpainting pipeline; Sharplight encodes the mean). Written to OUT as raw little-endian f32 plus index.json:

  input.png / mask.png   the input image and the mask (white = regenerate)
  noise                  unit Gaussian noise [1, 4, H/8, W/8]
  image_latents          the input image's VAE latent mode, unscaled [1, 4, h, w] (InstructPix2Pix)
  unet_latent/_condition/_context/_timestep/_out   one UNet evaluation on seeded random inputs (batch 2; the extra channels are the
                         condition: image latents, or mask and masked-image latents)
  image / latents        the pipeline's final image [3, H, W] in [0, 1] and latents
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw


def synthetic_image(size):
    """A deterministic picture with smooth gradients, a few shapes and some texture (no downloads)."""
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:size, 0:size] / size
    base = np.stack([0.3 + 0.5 * x, 0.4 + 0.3 * y, 0.8 - 0.4 * x * y], -1)
    image = Image.fromarray((np.clip(base + rng.normal(0, 0.02, base.shape), 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.ellipse([size * 0.2, size * 0.25, size * 0.6, size * 0.7], fill=(220, 60, 50))
    draw.rectangle([size * 0.55, size * 0.5, size * 0.9, size * 0.85], fill=(40, 120, 200))
    draw.polygon([(size * 0.1, size * 0.9), (size * 0.35, size * 0.55), (size * 0.5, size * 0.9)], fill=(240, 220, 80))
    return image


def write(out, name, tensor):
    np.asarray(tensor.detach().float().cpu().numpy(), dtype="<f4").tofile(os.path.join(out, name + ".f32"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["instruct-pix2pix", "inpaint"])
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = args.device
    size = args.size

    from diffusers import EulerDiscreteScheduler

    image = synthetic_image(size)
    image.save(os.path.join(args.out, "input.png"))
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(1, 4, size // 8, size // 8, generator=generator)
    write(args.out, "noise", noise)
    index = {"kind": args.kind, "size": size, "steps": args.steps}

    if args.kind == "instruct-pix2pix":
        from diffusers import StableDiffusionInstructPix2PixPipeline

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model, torch_dtype=torch.float32, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        prompt, negative = "make the circle green", None
        guidance, image_guidance = 7.5, 1.5
        index.update(prompt=prompt, guidance_scale=guidance, image_guidance_scale=image_guidance, sampler="euler")
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(device)
            latent_dist = pipe.vae.encode(tensor).latent_dist
            image_latents = latent_dist.mode()
            write(args.out, "image_latents", image_latents)
            g = torch.Generator().manual_seed(11)
            unet_latent = torch.randn(2, 4, size // 8, size // 8, generator=g)
            condition = torch.randn(2, 4, size // 8, size // 8, generator=g)
            context = torch.randn(2, 77, 768, generator=g) * 0.5
            timestep = torch.tensor([801.0, 801.0])
            unet_out = pipe.unet(torch.cat([unet_latent, condition], 1).to(device), timestep.to(device), context.to(device)).sample
            for name, value in [("unet_latent", unet_latent), ("unet_condition", condition), ("unet_context", context), ("unet_timestep", timestep), ("unet_out", unet_out)]:
                write(args.out, name, value)
            result = pipe(prompt, image=image, num_inference_steps=args.steps, guidance_scale=guidance, image_guidance_scale=image_guidance,
                          latents=noise.to(device), output_type="pt")
        write(args.out, "image", result.images[0])
    else:
        import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint as module
        from diffusers import StableDiffusionInpaintPipeline

        original = module.retrieve_latents
        module.retrieve_latents = lambda output, generator=None, sample_mode="sample": original(output, generator, "argmax")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(args.model, torch_dtype=torch.float32, variant="fp16", safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(device)
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse([size * 0.2, size * 0.25, size * 0.6, size * 0.7], fill=255)
        mask.save(os.path.join(args.out, "mask.png"))
        prompt, negative = "a green apple on a table", None
        guidance = 7.5
        index.update(prompt=prompt, guidance_scale=guidance, sampler=pipe.scheduler.__class__.__name__)
        with torch.no_grad():
            g = torch.Generator().manual_seed(11)
            unet_latent = torch.randn(2, 4, size // 8, size // 8, generator=g)
            condition = torch.randn(2, 5, size // 8, size // 8, generator=g)
            context = torch.randn(2, 77, 768, generator=g) * 0.5
            timestep = torch.tensor([801.0, 801.0])
            unet_out = pipe.unet(torch.cat([unet_latent, condition], 1).to(device), timestep.to(device), context.to(device)).sample
            for name, value in [("unet_latent", unet_latent), ("unet_condition", condition), ("unet_context", context), ("unet_timestep", timestep), ("unet_out", unet_out)]:
                write(args.out, name, value)
            result = pipe(prompt, image=image, mask_image=mask, num_inference_steps=args.steps, guidance_scale=guidance, strength=1.0,
                          latents=noise.to(device), output_type="pt")
        write(args.out, "image", result.images[0])

    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
