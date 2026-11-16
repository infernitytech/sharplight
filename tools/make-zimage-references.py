#!/usr/bin/env python3
"""Writes diffusers reference outputs of a Z-Image pipeline (Z-Image-Turbo) for the Sharplight image-generation tests.

MODEL is a Z-Image diffusers folder (configurations, text_encoder/, tokenizer/, vae/; its transformer/ weights are not needed when
--transformer-gguf gives a transformer-only GGUF, dequantized on the fly by diffusers). Writes raw little-endian f32 tensors plus
OUT/index.json:

  tokens / lengths / text_hidden_{i}    chat-templated token ids, prompt lengths and the text encoder's hidden_states[-2]
  transformer_*                         one transformer evaluation at SIZE on seeded random latents / caption features / mask
  noise_{size} / image_{size}           initial latents [1, 16, size/8, size/8] (seed 42) and the image of a txt2img run from them

The text encoder runs on the CPU, the transformer and the VAE on --device.
Usage: make-zimage-references.py MODEL OUT [--transformer-gguf FILE] [--sizes 512 1024] [--steps 9] [--device cuda]
"""

import argparse
import json
import os

import numpy as np
import torch

PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123"]
PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"
LENGTH = 512


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
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--transformer-gguf", default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel
    from transformers import AutoTokenizer, Qwen3Model

    dtype = torch.float32
    if args.transformer_gguf:
        transformer = ZImageTransformer2DModel.from_single_file(
            args.transformer_gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype, config=args.model,
            subfolder="transformer")
    else:
        transformer = ZImageTransformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=dtype)
    text = Qwen3Model.from_pretrained(os.path.join(args.model, "text_encoder"), torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
    pipe = ZImagePipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=transformer)
    device = args.device
    ref = Reference(args.out)
    ref.save({"prompt": PROMPT, "steps": args.steps, "length": LENGTH})

    tokens, lengths = {}, {}
    for i, prompt in enumerate(PROMPTS):
        templated = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        t = tokenizer([templated], padding="max_length", max_length=LENGTH, truncation=True, return_tensors="pt")
        tokens[prompt] = t.input_ids[0].tolist()
        lengths[prompt] = int(t.attention_mask[0].sum())
        ref.add(f"text_hidden_{i}", text(input_ids=t.input_ids, attention_mask=t.attention_mask.bool(), output_hidden_states=True).hidden_states[-2][0])
    ref.save({"tokens": tokens, "lengths": lengths})
    embeds = pipe.encode_prompt(PROMPT, device="cpu", do_classifier_free_guidance=False, max_sequence_length=LENGTH)[0]
    pipe.text_encoder = None
    transformer.to(device)
    vae.to(device)

    size = args.sizes[0]
    g = torch.Generator().manual_seed(1)
    latent = torch.randn(1, 16, size // 8, size // 8, generator=g)
    context = torch.randn(LENGTH, 2560, generator=g)
    valid = 23
    mask = (torch.arange(LENGTH) < valid).to(torch.float32)
    sigma = 0.75
    out = transformer([latent[0].unsqueeze(1).to(device)], torch.tensor([1 - sigma], device=device), [context[:valid].to(device)], return_dict=False)[0]
    ref.add("transformer_latent", latent)
    ref.add("transformer_context", context[None])
    ref.add("transformer_context_mask", mask[None])
    ref.add("transformer_timestep", torch.tensor([sigma * 1000]))
    ref.add("transformer_out", -out[0].squeeze(1)[None])
    ref.save({"transformer_size": size})

    for size in args.sizes:
        noise = torch.randn(1, 16, size // 8, size // 8, generator=torch.Generator().manual_seed(42))
        ref.add(f"noise_{size}", noise)
        image = pipe(prompt_embeds=[e.to(device) for e in embeds], height=size, width=size, num_inference_steps=args.steps, guidance_scale=0.0,
                     latents=noise.to(device), output_type="pt").images
        ref.add(f"image_{size}", image[0])
        ref.save({"timesteps_" + str(size): [float(t) for t in pipe.scheduler.timesteps], "sizes": args.sizes})
    print("written to", args.out)


if __name__ == "__main__":
    main()
