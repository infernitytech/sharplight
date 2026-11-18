#!/usr/bin/env python3
"""Writes diffusers reference images for the image-generation hot-model sweep (HotSweepTests, docs/model-support.md section 19.6).

Writes OUT/reference/{noise,image}.f32 (little-endian f32) and OUT/manifest.json (what the C# test needs: the model file, the auxiliary
files by role, size, steps, guidance, prompts, minimum PSNR). The model files are named relative to OUT.

  flux2      MODEL = FLUX.2 Klein diffusers folder (text_encoder, tokenizer, vae, scheduler, transformer config), --gguf a transformer-only
             GGUF (dequantized to f32 on the fly); --base for an undistilled model (guidance, empty negative prompt)
  zimage     MODEL = Z-Image diffusers folder, --gguf the transformer-only GGUF; guidance and a negative prompt for the base model
  sd         MODEL = SD 1.x / 2.x / XL diffusers folder (e.g. sd-turbo); --scheduler euler-ancestral-trailing etc.

Usage: make-hot-imagegen-references.py FAMILY MODEL OUT --size 512 --steps N [--guidance G] [--gguf FILE] [--device cpu|cuda]
"""

import argparse
import json
import os

import numpy as np
import torch

PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"


def save(folder, name, tensor):
    os.makedirs(folder, exist_ok=True)
    tensor.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(os.path.join(folder, name + ".f32"))


def dequantize(transformer, dtype):
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

    for module in transformer.modules():
        if isinstance(module, torch.nn.Linear):
            continue
        for name, parameter in list(module.named_parameters(recurse=False)):
            if hasattr(parameter, "quant_type"):
                setattr(module, name, torch.nn.Parameter(dequantize_gguf_tensor(parameter).to(dtype), requires_grad=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", choices=["flux2", "zimage", "sd"])
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--gguf", default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--negative", default=None)
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--min-psnr", type=float, default=25.0)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    dtype = torch.float32
    device = args.device
    size = args.size
    reference = os.path.join(args.out, "reference")
    manifest = {"size": size, "steps": args.steps, "prompt": PROMPT, "reference": "reference", "min_psnr": args.min_psnr, "auxiliary": {}}
    if args.guidance:
        manifest["guidance"] = args.guidance
    if args.negative is not None:
        manifest["negative"] = args.negative

    if args.family == "flux2":
        from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel, GGUFQuantizationConfig
        from transformers import AutoTokenizer, Qwen3ForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(args.model, subfolder="tokenizer")
        text = Qwen3ForCausalLM.from_pretrained(os.path.join(args.model, "text_encoder"), torch_dtype=dtype)
        vae = AutoencoderKLFlux2.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
        pipe = Flux2KleinPipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=None, is_distilled=not args.base)
        embeds = pipe.encode_prompt(PROMPT, device="cpu", max_sequence_length=512)[0]
        negative = pipe.encode_prompt("", device="cpu", max_sequence_length=512)[0] if args.base else None
        pipe.text_encoder = None
        del text
        transformer = Flux2Transformer2DModel.from_single_file(
            args.gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype, config=args.model, subfolder="transformer")
        dequantize(transformer, dtype)
        transformer.to(device)
        vae.to(device)
        pipe.transformer = transformer
        noise = torch.randn(1, transformer.config.in_channels, size // 16, size // 16, generator=torch.Generator().manual_seed(42))
        image = pipe(prompt_embeds=embeds.to(device), negative_prompt_embeds=negative.to(device) if negative is not None else None,
                     height=size, width=size, num_inference_steps=args.steps, guidance_scale=args.guidance if args.base else 1.0,
                     latents=noise.to(device), output_type="pt").images
    elif args.family == "zimage":
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel
        from transformers import AutoTokenizer, Qwen3Model

        transformer = ZImageTransformer2DModel.from_single_file(
            args.gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype, config=args.model, subfolder="transformer")
        dequantize(transformer, dtype)
        text = Qwen3Model.from_pretrained(os.path.join(args.model, "text_encoder"), torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(args.model, subfolder="tokenizer")
        vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
        pipe = ZImagePipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=transformer)
        cfg = args.guidance > 1
        embeds, negative = pipe.encode_prompt(PROMPT, negative_prompt=args.negative, device="cpu", do_classifier_free_guidance=cfg, max_sequence_length=512)
        pipe.text_encoder = None
        transformer.to(device)
        vae.to(device)
        noise = torch.randn(1, 16, size // 8, size // 8, generator=torch.Generator().manual_seed(42))
        image = pipe(prompt_embeds=[e.to(device) for e in embeds], negative_prompt_embeds=[e.to(device) for e in negative] if cfg else None,
                     height=size, width=size, num_inference_steps=args.steps, guidance_scale=args.guidance, latents=noise.to(device),
                     output_type="pt").images
    else:
        from diffusers import AutoPipelineForText2Image, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler

        pipe = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=dtype, variant="fp16", safety_checker=None)
        if args.scheduler == "euler-ancestral-trailing":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        elif args.scheduler == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        noise = torch.randn(1, 4, size // 8, size // 8, generator=torch.Generator().manual_seed(42))
        image = pipe(PROMPT, negative_prompt=args.negative, height=size, width=size, num_inference_steps=args.steps, guidance_scale=args.guidance,
                     latents=noise.to(device), output_type="pt").images
        manifest["sampler"] = args.scheduler
    save(reference, "noise", noise)
    save(reference, "image", image[0])
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("written to", args.out)


if __name__ == "__main__":
    main()
