#!/usr/bin/env python3
"""Writes diffusers reference outputs of a Stable Diffusion 1.x / 2.x pipeline for the Sharplight image-generation tests.

For a diffusers pipeline folder (MODEL) this writes raw little-endian f32 tensors plus OUT/index.json:

  tokens / text_hidden_{i}      CLIP token ids and text-encoder outputs of a few prompts
  unet_latent/context/timesteps/out    one UNet evaluation on seeded random inputs (batch 2)
  vae_latent / vae_decoded       a VAE decode of seeded random latents (raw decoder output, [-1, 1])
  vae_image / vae_encoded_mean   a VAE encode of a seeded random image in [-1, 1] (raw latent mean)
  noise                          unit Gaussian initial noise [1, 4, H/8, W/8] shared by the pipeline runs
  {sampler}_latents / _image     the final latents and image ([3, H, W] in [0, 1]) of a txt2img run per deterministic sampler,
                                 started from `noise` (the pipeline multiplies it by the scheduler's init_noise_sigma)

Usage: make-diffusion-references.py MODEL OUT [--size 512] [--steps 20] [--device cuda]. Needs torch, diffusers, transformers.
"""

import argparse
import json
import os

import numpy as np
import torch

PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123", ""]
PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"
NEGATIVE = "blurry, low quality"


class Reference:
    def __init__(self, folder):
        self.folder = folder
        self.index = {}
        os.makedirs(folder, exist_ok=True)

    def add(self, name, tensor):
        array = tensor.detach().to(torch.float32).cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor, np.float32)
        array.astype("<f4").tofile(os.path.join(self.folder, name + ".f32"))
        self.index[name] = list(array.shape)

    def save(self, extra):
        with open(os.path.join(self.folder, "index.json"), "w") as f:
            json.dump({"tensors": self.index, **extra}, f, indent=1)


def samplers(config):
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, PNDMScheduler
    return {
        "euler": EulerDiscreteScheduler.from_config(config),
        "euler-karras": EulerDiscreteScheduler.from_config(config, use_karras_sigmas=True),
        "ddim": DDIMScheduler.from_config(config),
        "pndm": PNDMScheduler.from_config(config),
        "dpm++2m": DPMSolverMultistepScheduler.from_config(config, algorithm_type="dpmsolver++", solver_order=2),
        "dpm++2m-karras": DPMSolverMultistepScheduler.from_config(config, algorithm_type="dpmsolver++", solver_order=2,
                                                                  use_karras_sigmas=True),
    }


def load_pipeline(folder):
    """Loads the pipeline; folders without model_index.json are assembled from their component folders."""
    from diffusers import AutoencoderKL, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    if os.path.exists(os.path.join(folder, "model_index.json")):
        return StableDiffusionPipeline.from_pretrained(folder, torch_dtype=torch.float32, safety_checker=None,
                                                       requires_safety_checker=False)
    return StableDiffusionPipeline(
        vae=AutoencoderKL.from_pretrained(folder, subfolder="vae"),
        text_encoder=CLIPTextModel.from_pretrained(folder, subfolder="text_encoder"),
        tokenizer=CLIPTokenizer.from_pretrained(folder, subfolder="tokenizer"),
        unet=UNet2DConditionModel.from_pretrained(folder, subfolder="unet"),
        scheduler=PNDMScheduler.from_pretrained(folder, subfolder="scheduler"),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", default=None, help="comma-separated sampler names to run")
    args = parser.parse_args()

    from diffusers import StableDiffusionPipeline
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipe = load_pipeline(args.model).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    device = args.device
    ref = Reference(args.out)
    tokens = {}
    latent_size = args.size // 8
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
            tokens[prompt] = ids[0].tolist()
            ref.add(f"text_hidden_{i}", pipe.text_encoder(ids.to(device)).last_hidden_state[0])

        g = torch.Generator().manual_seed(1)
        channels = pipe.unet.config.in_channels
        width = pipe.text_encoder.config.hidden_size
        latent = torch.randn(2, channels, latent_size, latent_size, generator=g)
        context = torch.randn(2, 77, width, generator=g)
        timesteps = torch.tensor([999.0, 321.5])
        ref.add("unet_latent", latent)
        ref.add("unet_context", context)
        ref.add("unet_timesteps", timesteps)
        ref.add("unet_out", pipe.unet(latent.to(device), timesteps.to(device), encoder_hidden_states=context.to(device)).sample)

        vae_latent = torch.randn(1, 4, latent_size, latent_size, generator=g)
        ref.add("vae_latent", vae_latent)
        ref.add("vae_decoded", pipe.vae.decode(vae_latent.to(device)).sample)
        image = torch.rand(1, 3, args.size, args.size, generator=g) * 2 - 1
        ref.add("vae_image", image)
        ref.add("vae_encoded_mean", pipe.vae.encode(image.to(device)).latent_dist.mean)

        noise = torch.randn(1, 4, latent_size, latent_size, generator=torch.Generator().manual_seed(42))
        ref.add("noise", noise)

    runs = {}
    chosen = samplers(pipe.scheduler.config)
    if args.only:
        chosen = {k: v for k, v in chosen.items() if k in args.only.split(",")}
    for name, sampler in chosen.items():
        pipe.scheduler = sampler
        final = {}

        def record(p, step, timestep, kwargs):
            final["latents"] = kwargs["latents"].clone()
            return kwargs

        with torch.no_grad():
            image = pipe(PROMPT, negative_prompt=NEGATIVE, height=args.size, width=args.size, num_inference_steps=args.steps,
                         guidance_scale=args.guidance, latents=noise.clone().to(device), output_type="pt",
                         callback_on_step_end=record).images
        ref.add(f"{name}_latents", final["latents"])
        ref.add(f"{name}_image", image[0])
        runs[name] = {"timesteps": [float(t) for t in sampler.timesteps],
                      "init_noise_sigma": float(sampler.init_noise_sigma)}
        if hasattr(sampler, "sigmas"):
            runs[name]["sigmas"] = [float(s) for s in sampler.sigmas]
        print(name, "done", flush=True)

    ref.save({"tokens": tokens, "prompt": PROMPT, "negative_prompt": NEGATIVE, "steps": args.steps, "guidance_scale": args.guidance,
              "height": args.size, "width": args.size, "samplers": runs})
    print("written to", args.out)


if __name__ == "__main__":
    main()
