#!/usr/bin/env python3
"""Diffusers references of a real Wan pipeline (Wan 2.1 T2V, Wan 2.2 TI2V) for the video-generation tests.

Runs the diffusers pipeline in f32 on CUDA (the text encoder first, then freed) with explicit initial latents and writes, to OUT:

  reference/*.f32        raw little-endian f32 tensors described by reference/index.json:
                         text_hidden (the masked, zero-padded prompt conditioning), negative_hidden, noise, transformer_out (one
                         conditional evaluation on the noise at the first timestep), latents (the final latents), video [F, 3, H, W]
                         in [0, 1] (the decoded result)

Usage: make-wan-references.py MODEL_FOLDER OUT [--frames 17] [--height 480] [--width 832] [--steps 20] [--image start.png]
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
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--image", default=None, help="start image (TI2V image-to-video)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline, WanTransformer3DModel
    from transformers import AutoTokenizer, UMT5EncoderModel

    import diffusers.pipelines.wan.pipeline_wan as t2v_module
    import diffusers.pipelines.wan.pipeline_wan_i2v as i2v_module
    i2v_module.prompt_clean = t2v_module.prompt_clean  # the I2V module calls ftfy without importing it
    torch.set_grad_enabled(False)
    folder = os.path.join(args.out, "reference")
    os.makedirs(folder, exist_ok=True)
    index = {}
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model, "tokenizer"))
    text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(args.model, "text_encoder"), torch_dtype=torch.float32).to(device)
    vae = AutoencoderKLWan.from_pretrained(os.path.join(args.model, "vae"), torch_dtype=torch.float32)
    pipe_class = WanImageToVideoPipeline if args.image else WanPipeline
    pipe = pipe_class.from_pretrained(args.model, transformer=None, text_encoder=text_encoder, tokenizer=tokenizer, vae=vae,
                                      torch_dtype=torch.float32)
    prompt_embeds, negative_embeds = pipe.encode_prompt(PROMPT, NEGATIVE, do_classifier_free_guidance=True, device=device,
                                                        max_sequence_length=512)
    save(folder, index, "text_hidden", prompt_embeds[0])
    save(folder, index, "negative_hidden", negative_embeds[0])
    del text_encoder, pipe
    torch.cuda.empty_cache()

    transformer = WanTransformer3DModel.from_pretrained(os.path.join(args.model, "transformer"), torch_dtype=torch.float32)
    pipe = pipe_class.from_pretrained(args.model, transformer=transformer, text_encoder=None, tokenizer=tokenizer, vae=vae,
                                      torch_dtype=torch.float32)
    # Components move to the GPU only while they run (an f32 5B transformer and the VAE do not fit 32 GB together).
    pipe.enable_model_cpu_offload()
    z = transformer.config.in_channels
    scale = vae.config.scale_factor_spatial
    latent_frames = (args.frames - 1) // 4 + 1
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(1, z, latent_frames, args.height // scale, args.width // scale, generator=generator)
    save(folder, index, "noise", noise[0])

    pipe.scheduler.set_timesteps(args.steps, device=device)
    t0 = pipe.scheduler.timesteps[0]
    timestep = t0.expand(1).to(device)
    if pipe.config.expand_timesteps:
        tokens = latent_frames * (args.height // scale // 2) * (args.width // scale // 2)
        timestep = torch.full((1, tokens), float(t0), device=device)
    transformer.to(device)
    out = transformer(noise.to(device), timestep, prompt_embeds, return_dict=False)[0]
    transformer.to("cpu")
    torch.cuda.empty_cache()
    save(folder, index, "transformer_out", out[0])
    index_extra = {"transformer_timestep": float(t0)}

    captured = {}

    def keep_latents(pipeline, step, t, kwargs):
        captured["latents"] = kwargs["latents"]
        return {}

    extra = {}
    if args.image:
        from PIL import Image
        extra["image"] = Image.open(args.image).convert("RGB")
    start = time.time()
    video = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, height=args.height, width=args.width,
                 num_frames=args.frames, num_inference_steps=args.steps, guidance_scale=args.guidance, latents=noise.clone().to(device),
                 output_type="latent", callback_on_step_end=keep_latents, callback_on_step_end_tensor_inputs=["latents"], **extra).frames
    elapsed = time.time() - start
    # Decode after freeing the transformer (an f32 5B transformer and the VAE do not fit 32 GB together).
    del pipe, transformer
    torch.cuda.empty_cache()
    vae.to(device)
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(device)
    std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(device)
    decoded = vae.decode(video.to(device) * std + mean, return_dict=False)[0]
    video = ((decoded[0].clamp(-1, 1) + 1) / 2).permute(1, 2, 3, 0).cpu().numpy()
    save(folder, index, "latents", captured["latents"][0])
    save(folder, index, "video", torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2))
    with open(os.path.join(folder, "index.json"), "w") as f:
        json.dump({"tensors": index, "prompt": PROMPT, "negative_prompt": NEGATIVE, "frames": args.frames, "height": args.height,
                   "width": args.width, "steps": args.steps, "guidance_scale": args.guidance, "seed": args.seed, "seconds": elapsed,
                   "image": os.path.basename(args.image) if args.image else None, **index_extra}, f, indent=1)
    print(f"wrote {args.out} ({elapsed:.1f} s for the pipeline)")


if __name__ == "__main__":
    main()
