#!/usr/bin/env python3
"""Writes diffusers reference outputs of a FLUX.2 Klein pipeline for the Sharplight image-generation tests.

MODEL is a FLUX.2 Klein diffusers folder (configurations, text_encoder/, tokenizer/, vae/; its transformer/ weights are not needed when
--transformer-gguf gives a transformer-only GGUF, dequantized on the fly by diffusers). Writes raw little-endian f32 tensors plus
OUT/index.json:

  tokens / lengths / text_hidden_{i}    chat-templated token ids (thinking disabled), prompt lengths and the prompt embeddings (Qwen3's
                                        hidden states entering layers 9, 18 and 27, concatenated per token, over all 512 rows)
  transformer_*                         one transformer evaluation at SIZE on seeded random latents / text features
  noise_{size} / image_{size}           initial latents [1, 128, size/16, size/16] (seed 42) and the image of a txt2img run from them

The text encoder runs on --text-device (f32), the transformer and the VAE on --device.
Usage: make-flux2-references.py MODEL OUT [--transformer-gguf FILE] [--sizes 512 1024] [--steps 4] [--device cuda]
"""

import argparse
import gc
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
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    from diffusers import (AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel,
                           GGUFQuantizationConfig)
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text = Qwen3ForCausalLM.from_pretrained(os.path.join(args.model, "text_encoder"), torch_dtype=dtype).to(args.text_device)
    vae = AutoencoderKLFlux2.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
    pipe = Flux2KleinPipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=None, is_distilled=True)
    ref = Reference(args.out)
    ref.save({"prompt": PROMPT, "steps": args.steps, "length": LENGTH})

    tokens, lengths = {}, {}
    for i, prompt in enumerate(PROMPTS):
        templated = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        t = tokenizer(templated, padding="max_length", max_length=LENGTH, truncation=True, return_tensors="pt")
        tokens[prompt] = t.input_ids[0].tolist()
        lengths[prompt] = int(t.attention_mask[0].sum())
        ref.add(f"text_hidden_{i}", pipe.encode_prompt(prompt, device=args.text_device, max_sequence_length=LENGTH)[0][0])
    ref.save({"tokens": tokens, "lengths": lengths})
    embeds = pipe.encode_prompt(PROMPT, device=args.text_device, max_sequence_length=LENGTH)[0]
    pipe.text_encoder = None
    del text
    gc.collect()

    if args.transformer_gguf:
        transformer = Flux2Transformer2DModel.from_single_file(
            args.transformer_gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype, config=args.model,
            subfolder="transformer")
        from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

        for module in transformer.modules():
            if isinstance(module, torch.nn.Linear):
                continue
            for name, parameter in list(module.named_parameters(recurse=False)):
                if hasattr(parameter, "quant_type"):
                    setattr(module, name, torch.nn.Parameter(dequantize_gguf_tensor(parameter).to(dtype), requires_grad=False))
    else:
        transformer = Flux2Transformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=dtype)
    device = args.device
    transformer.to(device)
    vae.to(device)
    pipe.transformer = transformer
    channels = transformer.config.in_channels

    size = args.sizes[0]
    lh = size // 16
    g = torch.Generator().manual_seed(1)
    latent = torch.randn(1, channels, lh, lh, generator=g)
    context = torch.randn(1, LENGTH, transformer.config.joint_attention_dim, generator=g)
    sigma = 0.75
    out = transformer(hidden_states=Flux2KleinPipeline._pack_latents(latent).to(device), encoder_hidden_states=context.to(device),
                      timestep=torch.tensor([sigma], device=device), img_ids=Flux2KleinPipeline._prepare_latent_ids(latent).to(device),
                      txt_ids=Flux2KleinPipeline._prepare_text_ids(context).to(device), return_dict=False)[0]
    ref.add("transformer_latent", latent)
    ref.add("transformer_context", context)
    ref.add("transformer_timestep", torch.tensor([sigma * 1000]))
    ref.add("transformer_out", out.permute(0, 2, 1).reshape(1, channels, lh, lh))
    ref.save({"transformer_size": size})

    for size in args.sizes:
        lh = size // 16
        noise = torch.randn(1, channels, lh, lh, generator=torch.Generator().manual_seed(42))
        ref.add(f"noise_{size}", noise)
        image = pipe(prompt_embeds=embeds.to(device), height=size, width=size, num_inference_steps=args.steps, guidance_scale=1.0,
                     latents=noise.to(device), output_type="pt").images
        ref.add(f"image_{size}", image[0])
        ref.save({"timesteps_" + str(size): [float(t) for t in pipe.scheduler.timesteps], "sizes": args.sizes})
    print("written to", args.out)


if __name__ == "__main__":
    main()
