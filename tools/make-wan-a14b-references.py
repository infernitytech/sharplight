#!/usr/bin/env python3
"""Diffusers references of Wan 2.2 T2V-A14B (two experts) running transformer GGUF files, for the video-generation tests.

The high-noise and low-noise transformers come from GGUF files (QuantStack / city96, dequantized on the fly by diffusers' GGUF
support, computed in f32), configured from the Wan2.2-T2V-A14B-Diffusers configuration folder; the umT5 encoder, tokenizer and VAE
from a Wan 2.1 diffusers folder (the same components). Writes to OUT/reference: text_hidden, negative_hidden, noise,
transformer_out (high-noise expert, first timestep), transformer_2_out (low-noise expert, same inputs), video [F, 3, H, W], index.json.

Usage: make-wan-a14b-references.py HIGH.gguf LOW.gguf CONFIG_FOLDER WAN21_FOLDER OUT [--frames 17] [--height 480] [--width 832] [--steps 20]
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
    parser.add_argument("high")
    parser.add_argument("low")
    parser.add_argument("config")
    parser.add_argument("components")
    parser.add_argument("out")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--guidance-2", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from diffusers import GGUFQuantizationConfig, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
    from transformers import AutoTokenizer, UMT5EncoderModel

    torch.set_grad_enabled(False)
    folder = os.path.join(args.out, "reference")
    os.makedirs(folder, exist_ok=True)
    index = {}
    quantization = GGUFQuantizationConfig(compute_dtype=torch.float32)
    high = WanTransformer3DModel.from_single_file(args.high, quantization_config=quantization, config=os.path.join(args.config, "transformer"),
                                                  torch_dtype=torch.float32)
    low = WanTransformer3DModel.from_single_file(args.low, quantization_config=quantization, config=os.path.join(args.config, "transformer_2"),
                                                 torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.components, "tokenizer"))
    text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(args.components, "text_encoder"), torch_dtype=torch.float32)
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(os.path.join(args.components, "vae"), torch_dtype=torch.float32)
    with open(os.path.join(args.config, "scheduler", "scheduler_config.json")) as f:
        scheduler = UniPCMultistepScheduler.from_config(json.load(f))
    with open(os.path.join(args.config, "model_index.json")) as f:
        boundary = json.load(f)["boundary_ratio"]
    pipe = WanPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler, transformer=high, transformer_2=low,
                       boundary_ratio=boundary)
    pipe.enable_model_cpu_offload()
    embeds, negative = pipe.encode_prompt(PROMPT, NEGATIVE, do_classifier_free_guidance=True, max_sequence_length=512)
    save(folder, index, "text_hidden", embeds[0])
    save(folder, index, "negative_hidden", negative[0])

    latent_frames = (args.frames - 1) // 4 + 1
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(1, 16, latent_frames, args.height // 8, args.width // 8, generator=generator)
    save(folder, index, "noise", noise[0])
    pipe.scheduler.set_timesteps(args.steps)
    t0 = float(pipe.scheduler.timesteps[0])
    for name, model in (("transformer_out", high), ("transformer_2_out", low)):
        model.to("cuda")
        out = model(noise.cuda(), torch.tensor([t0]).cuda(), embeds.cuda(), return_dict=False)[0]
        save(folder, index, name, out[0])
        model.to("cpu")
        torch.cuda.empty_cache()

    start = time.time()
    video = pipe(prompt_embeds=embeds, negative_prompt_embeds=negative, height=args.height, width=args.width, num_frames=args.frames,
                 num_inference_steps=args.steps, guidance_scale=args.guidance, guidance_scale_2=args.guidance_2, latents=noise.clone(),
                 output_type="np").frames[0]
    elapsed = time.time() - start
    save(folder, index, "video", torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2))
    with open(os.path.join(folder, "index.json"), "w") as f:
        json.dump({"tensors": index, "prompt": PROMPT, "negative_prompt": NEGATIVE, "frames": args.frames, "height": args.height,
                   "width": args.width, "steps": args.steps, "guidance_scale": args.guidance, "guidance_scale_2": args.guidance_2,
                   "boundary_ratio": boundary, "seed": args.seed, "seconds": elapsed, "transformer_timestep": t0}, f, indent=1)
    print(f"wrote {args.out} ({elapsed:.1f} s for the pipeline)")


if __name__ == "__main__":
    main()
