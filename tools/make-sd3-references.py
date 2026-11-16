#!/usr/bin/env python3
"""Writes diffusers reference outputs of an SD3 / SD3.5 pipeline for the Sharplight image-generation tests.

For a diffusers SD3 folder (MODEL) this writes raw little-endian f32 tensors plus OUT/index.json:

  tokens / tokens_2 / tokens_3          CLIP-L, CLIP-G and T5 token ids of a few prompts
  text{,2}_pooled_{i} / text{,2}_hidden_{i} / text3_hidden_{i}
                                        projected pooled embeddings and penultimate hidden states of both CLIP encoders, T5 hidden states
  transformer_*                         one transformer evaluation at SIZE on seeded random latents / context / pooled embedding
  vae_latent / vae_decoded              a VAE decode of seeded random latents (raw decoder output in [-1, 1], latents unscaled)
  noise_{size}                          unit Gaussian initial latents [1, 16, size/8, size/8] (seed 42)
  image_{size} / latents_{size}         the image ([3, H, W] in [0, 1]) and final latents of a guided txt2img run from noise_{size}

Weights are upcast to f32 (--dtype). The text encoders always run on the CPU; the transformer and the VAE on --device.
--transformer-gguf replaces the folder's transformer with a transformer-only GGUF (dequantized by diffusers).
Usage: make-sd3-references.py MODEL OUT [--sizes 512 1024] [--steps 28] [--guidance 7] [--device cuda] [--threads 16]
"""

import argparse
import json
import os

import numpy as np
import torch

PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123", ""]
PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"
T5_LENGTH = 256


class Reference:
    def __init__(self, folder):
        self.folder = folder
        self.index = {}
        self.extra = {}
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "index.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self.index = data.pop("tensors", {})
            self.extra = data

    def add(self, name, tensor):
        array = tensor.detach().to(torch.float32).cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor, np.float32)
        array.astype("<f4").tofile(os.path.join(self.folder, name + ".f32"))
        self.index[name] = list(array.shape)
        self.save({})

    def save(self, extra):
        self.extra.update(extra)
        with open(os.path.join(self.folder, "index.json"), "w") as f:
            json.dump({"tensors": self.index, **self.extra}, f, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--skip", nargs="*", default=[], help="parts to skip: text transformer vae images")
    parser.add_argument("--transformer-gguf", default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    from diffusers import StableDiffusion3Pipeline

    dtype = getattr(torch, args.dtype)
    extra = {}
    if args.transformer_gguf:
        from diffusers import GGUFQuantizationConfig, SD3Transformer2DModel
        extra["transformer"] = SD3Transformer2DModel.from_single_file(
            args.transformer_gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype,
            config=args.model, subfolder="transformer")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=dtype, **extra)
    device = args.device
    pipe.transformer.to(device)
    pipe.vae.to(device)
    ref = Reference(args.out)
    ref.save({"prompt": PROMPT, "steps": args.steps, "guidance_scale": args.guidance, "t5_length": T5_LENGTH})

    if "text" not in args.skip:
        tokens, tokens_2, tokens_3 = {}, {}, {}
        for i, prompt in enumerate(PROMPTS):
            ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
            ids_2 = pipe.tokenizer_2(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
            ids_3 = pipe.tokenizer_3(prompt, padding="max_length", max_length=T5_LENGTH, truncation=True, add_special_tokens=True,
                                     return_tensors="pt").input_ids
            tokens[prompt] = ids[0].tolist()
            tokens_2[prompt] = ids_2[0].tolist()
            tokens_3[prompt] = ids_3[0].tolist()
            o = pipe.text_encoder(ids, output_hidden_states=True)
            ref.add(f"text_pooled_{i}", o[0][0])
            ref.add(f"text_hidden_{i}", o.hidden_states[-2][0])
            o = pipe.text_encoder_2(ids_2, output_hidden_states=True)
            ref.add(f"text2_pooled_{i}", o[0][0])
            ref.add(f"text2_hidden_{i}", o.hidden_states[-2][0])
            ref.add(f"text3_hidden_{i}", pipe.text_encoder_3(ids_3)[0][0])
        ref.save({"tokens": tokens, "tokens_2": tokens_2, "tokens_3": tokens_3})

    size = args.sizes[0]
    lh = lw = size // 8
    g = torch.Generator().manual_seed(1)
    latent = torch.randn(1, 16, lh, lw, generator=g)
    context = torch.randn(1, 77 + T5_LENGTH, 4096, generator=g)
    pooled = torch.randn(1, 2048, generator=g)
    if "transformer" not in args.skip:
        timestep = torch.tensor([750.0])
        out = pipe.transformer(latent.to(device, dtype), encoder_hidden_states=context.to(device, dtype), pooled_projections=pooled.to(device, dtype),
                               timestep=timestep.to(device, dtype), return_dict=False)[0]
        ref.add("transformer_latent", latent)
        ref.add("transformer_context", context)
        ref.add("transformer_pooled", pooled)
        ref.add("transformer_timestep", timestep)
        ref.add("transformer_out", out)
        ref.save({"transformer_size": size})

    if "vae" not in args.skip:
        vae_latent = torch.randn(1, 16, lh, lw, generator=g)
        ref.add("vae_latent", vae_latent)
        ref.add("vae_decoded", pipe.vae.decode(vae_latent.to(device, dtype)).sample)
        ref.save({"vae_size": size})

    if "images" not in args.skip:
        embeds = pipe.encode_prompt(PROMPT, None, None, device="cpu", do_classifier_free_guidance=True, max_sequence_length=T5_LENGTH)
        embeds = [e.to(device, dtype) for e in embeds]
        # The text encoders stay on the CPU; without them the pipeline runs on the transformer's device.
        pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
        for size in args.sizes:
            noise = torch.randn(1, 16, size // 8, size // 8, generator=torch.Generator().manual_seed(42))
            ref.add(f"noise_{size}", noise)
            latents = pipe(prompt_embeds=embeds[0], negative_prompt_embeds=embeds[1], pooled_prompt_embeds=embeds[2],
                           negative_pooled_prompt_embeds=embeds[3], height=size, width=size, num_inference_steps=args.steps,
                           guidance_scale=args.guidance, latents=noise.to(device, dtype), output_type="latent").images
            ref.add(f"latents_{size}", latents)
            decoded = pipe.vae.decode((latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor).to(device, dtype)).sample
            ref.add(f"image_{size}", (decoded[0].float() / 2 + 0.5).clamp(0, 1))
            ref.save({"timesteps_" + str(size): [float(t) for t in pipe.scheduler.timesteps], "sizes": args.sizes})
    print("written to", args.out)


if __name__ == "__main__":
    main()
