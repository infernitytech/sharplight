#!/usr/bin/env python3
"""Writes diffusers reference outputs of a Chroma pipeline (Chroma1-HD) for the Sharplight image-generation tests.

Chroma's transformer is taken from a transformer-only GGUF (--transformer-gguf, dequantized on the fly by diffusers; the configurations
come from CONFIGS, a Chroma diffusers folder with at least transformer/, scheduler/ and tokenizer/), the T5 encoder and the VAE from a
FLUX.1 diffusers folder (--flux: text_encoder_2/ and vae/, the same T5-XXL and autoencoder). The transformer mask is boolean, as in the
original Chroma code (diffusers' float mask is added to the attention scores instead; see make-tiny-chroma.py).

Writes raw little-endian f32 tensors plus OUT/index.json:

  tokens / text_hidden_{i}              T5 token ids and (masked) hidden states of a few prompts
  transformer_*                         one transformer evaluation at SIZE on seeded random latents / context / mask
  noise_{size} / image_{size}           initial latents [1, 16, size/8, size/8] (seed 42) and the image of a guided txt2img run from them

The text encoder runs on the CPU, the transformer and the VAE on --device.
Usage: make-chroma-references.py CONFIGS OUT --transformer-gguf FILE --flux FOLDER [--sizes 512] [--steps 20] [--guidance 4] [--device cuda]
"""

import argparse
import json
import os

import numpy as np
import torch

PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123", ""]
PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"
NEGATIVE = "low quality, blurry"
T5_LENGTH = 512


class Reference:
    def __init__(self, folder):
        self.folder = folder
        self.index = {}
        self.extra = {}
        os.makedirs(folder, exist_ok=True)

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
    parser.add_argument("configs")
    parser.add_argument("out")
    parser.add_argument("--transformer-gguf", required=True)
    parser.add_argument("--flux", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    from diffusers import AutoencoderKL, ChromaPipeline, ChromaTransformer2DModel, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig
    from transformers import T5EncoderModel, T5TokenizerFast

    dtype = torch.float32
    transformer = ChromaTransformer2DModel.from_single_file(
        args.transformer_gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype, config=args.configs,
        subfolder="transformer")
    text = T5EncoderModel.from_pretrained(args.flux, subfolder="text_encoder_2", torch_dtype=dtype)
    tokenizer = T5TokenizerFast.from_pretrained(args.flux, subfolder="tokenizer_2")
    vae = AutoencoderKL.from_pretrained(args.flux, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.configs, subfolder="scheduler")
    pipe = ChromaPipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=transformer)
    original = pipe._prepare_attention_mask
    pipe._prepare_attention_mask = lambda *a, **k: (lambda m: None if m is None else m.bool())(original(*a, **k))
    device = args.device
    ref = Reference(args.out)
    ref.save({"prompt": PROMPT, "negative_prompt": NEGATIVE, "steps": args.steps, "guidance_scale": args.guidance, "t5_length": T5_LENGTH})

    tokens = {}
    for i, prompt in enumerate(PROMPTS):
        t = tokenizer(prompt, padding="max_length", max_length=T5_LENGTH, truncation=True, return_tensors="pt")
        tokens[prompt] = t.input_ids[0].tolist()
        ref.add(f"text_hidden_{i}", text(t.input_ids, attention_mask=t.attention_mask)[0][0])
    ref.save({"tokens": tokens})
    embeds = pipe.encode_prompt(PROMPT, negative_prompt=NEGATIVE, device="cpu", do_classifier_free_guidance=True, max_sequence_length=T5_LENGTH)
    pipe.text_encoder = None
    transformer.to(device)
    vae.to(device)

    size = args.sizes[0]
    lh = lw = size // 8
    g = torch.Generator().manual_seed(1)
    latent = torch.randn(1, 16, lh, lw, generator=g)
    context = torch.randn(1, T5_LENGTH, 4096, generator=g)
    mask = (torch.arange(T5_LENGTH) <= 11).to(torch.float32)[None]
    img_ids = ChromaPipeline._prepare_latent_image_ids(1, lh // 2, lw // 2, device, dtype)
    txt_ids = torch.zeros(T5_LENGTH, 3, device=device)
    full_mask = pipe._prepare_attention_mask(1, img_ids.shape[0], dtype, mask.to(device))
    out = transformer(ChromaPipeline._pack_latents(latent, 1, 16, lh, lw).to(device), encoder_hidden_states=context.to(device),
                      timestep=torch.tensor([0.75], device=device), img_ids=img_ids, txt_ids=txt_ids, attention_mask=full_mask, return_dict=False)[0]
    ref.add("transformer_latent", latent)
    ref.add("transformer_context", context)
    ref.add("transformer_context_mask", mask)
    ref.add("transformer_timestep", torch.tensor([750.0]))
    ref.add("transformer_out", ChromaPipeline._unpack_latents(out.float().cpu(), size, size, 8))
    ref.save({"transformer_size": size})

    prompt_embeds, prompt_ids, prompt_mask, negative_embeds, negative_ids, negative_mask = embeds
    for size in args.sizes:
        noise = torch.randn(1, 16, size // 8, size // 8, generator=torch.Generator().manual_seed(42))
        ref.add(f"noise_{size}", noise)
        packed = ChromaPipeline._pack_latents(noise, 1, 16, size // 8, size // 8).to(device)
        image = pipe(prompt_embeds=prompt_embeds.to(device), prompt_attention_mask=prompt_mask.to(device),
                     negative_prompt_embeds=negative_embeds.to(device), negative_prompt_attention_mask=negative_mask.to(device), height=size,
                     width=size, num_inference_steps=args.steps, guidance_scale=args.guidance, latents=packed, output_type="pt").images
        ref.add(f"image_{size}", image[0])
        ref.save({"timesteps_" + str(size): [float(t) for t in pipe.scheduler.timesteps], "sizes": args.sizes})
    print("written to", args.out)


if __name__ == "__main__":
    main()
