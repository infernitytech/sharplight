#!/usr/bin/env python3
"""Writes diffusers reference images of popular single-file image-generation checkpoints for the sweep tests (SweepTests).

Every MODEL is a single-file checkpoint (an SD 1.x or SDXL fine-tune / merge as distributed on the Hub), loaded with diffusers'
from_single_file (configuration from --sd15-config / --sdxl-config, diffusers folders of the base models) and run in f32 with Euler
from seeded latents. Writes OUT/<id>/reference/{noise,image}.f32 and index.json (file, family, size, steps, guidance, prompt, negative,
timesteps); SweepTests loads the same file in Sharplight and compares the images (PSNR).

Usage: make-sweep-references.py OUT ID=FILE[:family[:size[:steps[:guidance]]]] ... [--device cuda]
"""

import argparse
import json
import os

import numpy as np
import torch

PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"
NEGATIVE = "blurry, low quality"


def save(folder, name, tensor, index):
    os.makedirs(folder, exist_ok=True)
    array = tensor.detach().to(torch.float32).cpu().numpy()
    array.astype("<f4").tofile(os.path.join(folder, name + ".f32"))
    index[name] = list(array.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("models", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sd15-config", default=os.path.expanduser("~/models/sd/sd15-diffusers"))
    parser.add_argument("--sdxl-config", default=os.path.expanduser("~/models/sd/sdxl-diffusers"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline

    for spec in args.models:
        model_id, rest = spec.split("=", 1)
        parts = rest.split(":")
        path = parts[0]
        family = parts[1] if len(parts) > 1 and parts[1] else ("sdxl" if os.path.getsize(path) > 5e9 else "sd1")
        xl = family == "sdxl"
        size = int(parts[2]) if len(parts) > 2 and parts[2] else (1024 if xl else 512)
        steps = int(parts[3]) if len(parts) > 3 and parts[3] else 20
        guidance = float(parts[4]) if len(parts) > 4 and parts[4] else 7.0
        cls = StableDiffusionXLPipeline if xl else StableDiffusionPipeline
        pipe = cls.from_single_file(path, config=args.sdxl_config if xl else args.sd15_config, torch_dtype=torch.float32)
        if not xl:
            pipe.safety_checker = None
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        pipe.to(args.device)
        noise = torch.randn(1, 4, size // 8, size // 8, generator=torch.Generator().manual_seed(42))
        folder = os.path.join(args.out, model_id, "reference")
        index = {}
        save(folder, "noise", noise, index)
        image = pipe(PROMPT, negative_prompt=NEGATIVE if guidance > 1 else None, height=size, width=size, num_inference_steps=steps,
                     guidance_scale=guidance, latents=noise.to(args.device), output_type="pt").images[0]
        save(folder, "image", image, index)
        with open(os.path.join(folder, "index.json"), "w") as f:
            json.dump({"tensors": index, "file": os.path.basename(path), "family": family, "size": size, "steps": steps, "guidance": guidance,
                       "prompt": PROMPT, "negative": NEGATIVE if guidance > 1 else None, "sampler": "euler",
                       "timesteps": [float(t) for t in pipe.scheduler.timesteps]}, f, indent=1)
        print("written", model_id, flush=True)
        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
