#!/usr/bin/env python3
"""Writes diffusers reference outputs of a Qwen-Image pipeline for the Sharplight image-generation tests.

MODEL is a Qwen-Image diffusers folder (configurations, text_encoder/, tokenizer/, vae/; its transformer/ weights are not needed when
--transformer-gguf gives a transformer-only GGUF, dequantized on the fly by diffusers). Writes raw little-endian f32 tensors plus
OUT/index.json:

  tokens / lengths / text_hidden_{i}    system-templated token ids (34 template tokens first), prompt lengths and the prompt embeddings
                                        (the text encoder's final-normed last hidden states after the template's tokens)
  transformer_*                         one transformer evaluation at SIZE on seeded random latents / text features / mask
  noise_{size} / image_{size}           initial latents [1, 16, size/8, size/8] (seed 42) and the image of a guided txt2img run from them
                                        (negative prompt " ", true_cfg_scale 4)

The text encoder runs first (on --text-device, f32), then is freed; the transformer and the VAE run on --device.
Usage: make-qwen-image-references.py MODEL OUT [--transformer-gguf FILE] [--sizes 512 1024] [--steps 20] [--device cuda]
"""

import argparse
import gc
import json
import os

import numpy as np
import torch

PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123"]
PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"
NEGATIVE = " "
LENGTH = 512
DROP = 34


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
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    from diffusers import (AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, QwenImagePipeline,
                           QwenImageTransformer2DModel)
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text = Qwen2_5_VLForConditionalGeneration.from_pretrained(os.path.join(args.model, "text_encoder"), torch_dtype=dtype).to(args.text_device)
    vae = AutoencoderKLQwenImage.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
    pipe = QwenImagePipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=None)
    ref = Reference(args.out)
    ref.save({"prompt": PROMPT, "negative": NEGATIVE, "guidance": args.guidance, "steps": args.steps, "length": LENGTH, "drop": DROP})

    tokens, lengths = {}, {}
    for i, prompt in enumerate(PROMPTS):
        t = tokenizer([pipe.prompt_template_encode.format(prompt)], padding="max_length", max_length=LENGTH + DROP, truncation=True, return_tensors="pt")
        tokens[prompt] = t.input_ids[0].tolist()
        lengths[prompt] = int(t.attention_mask[0].sum())
        ref.add(f"text_hidden_{i}", pipe.encode_prompt(prompt, device=args.text_device, max_sequence_length=LENGTH)[0][0])
    ref.save({"tokens": tokens, "lengths": lengths})
    embeds, embeds_mask = pipe.encode_prompt(PROMPT, device=args.text_device, max_sequence_length=LENGTH)
    negative, negative_mask = pipe.encode_prompt(NEGATIVE, device=args.text_device, max_sequence_length=LENGTH)
    pipe.text_encoder = None
    del text
    gc.collect()

    if args.transformer_gguf:
        transformer = QwenImageTransformer2DModel.from_single_file(
            args.transformer_gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype, config=args.model,
            subfolder="transformer")
    else:
        transformer = QwenImageTransformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=dtype)
    device = args.device
    transformer.to(device)
    vae.to(device)
    pipe.transformer = transformer

    size = args.sizes[0]
    lh = size // 8
    g = torch.Generator().manual_seed(1)
    latent = torch.randn(1, 16, lh, lh, generator=g)
    context = torch.randn(LENGTH, transformer.config.joint_attention_dim, generator=g)
    valid = 23
    mask = (torch.arange(LENGTH) < valid).to(torch.float32)
    sigma = 0.75
    packed = QwenImagePipeline._pack_latents(latent, 1, 16, lh, lh).to(device)
    out = transformer(hidden_states=packed, encoder_hidden_states=context[None, :valid].to(device), encoder_hidden_states_mask=None,
                      timestep=torch.tensor([sigma], device=device), img_shapes=[[(1, lh // 2, lh // 2)]], return_dict=False)[0]
    ref.add("transformer_latent", latent)
    ref.add("transformer_context", context[None])
    ref.add("transformer_context_mask", mask[None])
    ref.add("transformer_timestep", torch.tensor([sigma * 1000]))
    ref.add("transformer_out", QwenImagePipeline._unpack_latents(out, size, size, 8)[:, :, 0])
    ref.save({"transformer_size": size})

    for size in args.sizes:
        lh = size // 8
        noise = torch.randn(1, 16, lh, lh, generator=torch.Generator().manual_seed(42))
        ref.add(f"noise_{size}", noise)
        image = pipe(prompt_embeds=embeds.to(device), prompt_embeds_mask=None if embeds_mask is None else embeds_mask.to(device),
                     negative_prompt_embeds=negative.to(device), negative_prompt_embeds_mask=None if negative_mask is None else negative_mask.to(device),
                     true_cfg_scale=args.guidance, height=size, width=size, num_inference_steps=args.steps,
                     latents=QwenImagePipeline._pack_latents(noise, 1, 16, lh, lh).to(device), output_type="pt").images
        ref.add(f"image_{size}", image[0])
        ref.save({"timesteps_" + str(size): [float(t) for t in pipe.scheduler.timesteps], "sizes": args.sizes})
    print("written to", args.out)


if __name__ == "__main__":
    main()
