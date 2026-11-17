#!/usr/bin/env python3
"""Writes diffusers reference outputs of a Krea 2 pipeline for the Sharplight image-generation tests.

MODEL is a Krea 2 diffusers folder (configurations, text_encoder/, tokenizer/, vae/, model_index.json; its transformer/ weights are not
needed when --transformer-gguf gives a transformer-only GGUF, dequantized on the fly by diffusers). Writes raw little-endian f32 tensors plus
OUT/index.json:

  tokens / lengths / text_hidden_{i}    the valid token ids in Sharplight's order (system turn, prompt, assistant turn), their count, and
                                        the tapped hidden states of those rows after the system turn [n, layers * 2560]
  transformer_*                         one transformer evaluation at SIZE on seeded random latents / text features / mask
  noise_{size} / image_{size}           initial latents [1, 16, size/8, size/8] (seed 42) and the image of a txt2img run from them
                                        (Turbo: 8 steps without guidance; --guidance G uses diffusers' Krea convention, Sharplight's 1 + G)

The text encoder runs first (on --text-device, f32), then is freed; the transformer and the VAE run on --device.
Usage: make-krea2-references.py MODEL OUT [--transformer-gguf FILE] [--sizes 512 1024] [--steps 8] [--device cuda]
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
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=0.0, help="diffusers' Krea guidance (0: none)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, Krea2Pipeline, Krea2Transformer2DModel
    from transformers import AutoTokenizer, Qwen3VLModel

    dtype = torch.float32
    with open(os.path.join(args.model, "model_index.json")) as f:
        index = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model, "tokenizer"))
    text = Qwen3VLModel.from_pretrained(os.path.join(args.model, "text_encoder"), torch_dtype=dtype).to(args.text_device)
    vae = AutoencoderKLQwenImage.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
    pipe = Krea2Pipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=None,
                         text_encoder_select_layers=index.get("text_encoder_select_layers"), is_distilled=index.get("is_distilled", False),
                         patch_size=index.get("patch_size", 2))
    skip = pipe.prompt_template_encode_start_idx
    ref = Reference(args.out)
    ref.save({"prompt": PROMPT, "guidance": 1 + args.guidance, "steps": args.steps, "length": LENGTH, "drop": skip})

    tokens, lengths = {}, {}
    for i, prompt in enumerate(PROMPTS):
        hidden, mask = pipe.get_text_hidden_states(prompt, LENGTH, device=args.text_device)
        text_ids = tokenizer([pipe.prompt_template_encode_prefix + prompt], truncation=True, padding="max_length",
                             max_length=LENGTH + skip - pipe.prompt_template_encode_num_suffix_tokens, return_tensors="pt")
        suffix = tokenizer([pipe.prompt_template_encode_suffix], return_tensors="pt")
        ids = torch.cat([text_ids.input_ids, suffix.input_ids], 1)[0]
        valid = torch.cat([text_ids.attention_mask, suffix.attention_mask], 1)[0].bool()
        tokens[prompt] = ids[valid].tolist()
        lengths[prompt] = int(valid.sum())
        ref.add(f"text_hidden_{i}", hidden[0][mask[0]].reshape(int(mask[0].sum()), -1))
    ref.save({"tokens": tokens, "lengths": lengths})
    embeds, embeds_mask = pipe.encode_prompt(PROMPT, device=args.text_device)
    negative, negative_mask = pipe.encode_prompt("", device=args.text_device)
    pipe.text_encoder = None
    del text
    gc.collect()

    if args.transformer_gguf:
        from diffusers.models.model_loading_utils import load_gguf_checkpoint

        # diffusers' GGUF reader drops unit dimensions: the [1, 12] projector and the [2, D] final table get their shapes back.
        state = load_gguf_checkpoint(args.transformer_gguf)
        state["txtfusion.projector.weight"] = state["txtfusion.projector.weight"].reshape(1, -1)
        transformer = Krea2Transformer2DModel.from_single_file(
            state, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype, config=args.model,
            subfolder="transformer")
        # diffusers dequantizes GGUF weights on the fly only in linear layers: other quantized or f16 parameters are dequantized up front.
        from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

        for module in transformer.modules():
            if isinstance(module, torch.nn.Linear):
                continue
            for name, parameter in list(module.named_parameters(recurse=False)):
                if hasattr(parameter, "quant_type"):
                    setattr(module, name, torch.nn.Parameter(dequantize_gguf_tensor(parameter).to(dtype), requires_grad=False))
    else:
        transformer = Krea2Transformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=dtype)
    device = args.device
    transformer.to(device)
    vae.to(device)
    pipe.transformer = transformer

    size = args.sizes[0]
    lh = size // 8
    g = torch.Generator().manual_seed(1)
    layers = transformer.config.num_text_layers
    width = transformer.config.text_hidden_dim
    latent = torch.randn(1, 16, lh, lh, generator=g)
    context = torch.randn(LENGTH, layers, width, generator=g)
    valid = 23
    mask = torch.arange(LENGTH) < valid
    sigma = 0.75
    packed = pipe._pack_latents(latent, 1, 16, lh, lh).to(device)
    positions = Krea2Pipeline.prepare_position_ids(LENGTH, lh // 2, lh // 2, device)
    out = transformer(hidden_states=packed, encoder_hidden_states=context[None].to(device), timestep=torch.tensor([sigma], device=device),
                      position_ids=positions, encoder_attention_mask=mask[None].to(device), return_dict=False)[0]
    out = out.view(1, lh // 2, lh // 2, 16, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(1, 16, lh, lh)
    ref.add("transformer_latent", latent)
    ref.add("transformer_context", context.reshape(1, LENGTH, -1))
    ref.add("transformer_context_mask", mask.to(torch.float32)[None])
    ref.add("transformer_timestep", torch.tensor([sigma * 1000]))
    ref.add("transformer_out", out)
    ref.save({"transformer_size": size})
    del out
    torch.cuda.empty_cache() if device.startswith("cuda") else None

    for size in args.sizes:
        lh = size // 8
        noise = torch.randn(1, 16, lh, lh, generator=torch.Generator().manual_seed(42))
        ref.add(f"noise_{size}", noise)
        extra = dict(negative_prompt_embeds=negative.to(device), negative_prompt_embeds_mask=negative_mask.to(device)) if args.guidance > 0 else {}
        image = pipe(prompt_embeds=embeds.to(device), prompt_embeds_mask=embeds_mask.to(device), guidance_scale=args.guidance, height=size, width=size,
                     num_inference_steps=args.steps, latents=pipe._pack_latents(noise, 1, 16, lh, lh).to(device), output_type="pt", **extra).images
        ref.add(f"image_{size}", image[0])
        ref.save({"timesteps_" + str(size): [float(t) for t in pipe.scheduler.timesteps], "sizes": args.sizes})
    print("written to", args.out)


if __name__ == "__main__":
    main()
