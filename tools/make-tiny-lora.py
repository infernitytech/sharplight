#!/usr/bin/env python3
"""Creates random LoRA files for the tiny random SD 1.x, SDXL and FLUX pipelines and diffusers reference images with them loaded.

For every family a LoRA on the denoiser's attention / MLP projections (and, for SD / SDXL, the text encoders) is written in diffusers' PEFT
format and in kohya-ss' format (SD 1.x: diffusers module names with underscores; SDXL: the SGM UNet's names (input_blocks / middle_block /
output_blocks), the format kohya trains SDXL in; FLUX: Black Forest Labs names with the fused q/k/v). The reference images come from
diffusers' load_lora_weights (scale via cross_attention_kwargs / joint_attention_kwargs) from the same initial noise.

Written to OUT (default ~/models/lora/tiny-lora):

  {family}/peft.safetensors, {family}/kohya.safetensors     the LoRA files
  {family}/reference/{noise,image_peft,image_kohya,image_base}.f32 + index.json

Requires torch, diffusers, peft and safetensors and the tiny pipelines (tools/make-tiny-diffusion-models.py, make-tiny-sdxl.py, make-tiny-flux.py).
"""

import argparse
import json
import os
import re

import numpy as np
import torch

PROMPT = "a photograph of an astronaut riding a horse on the moon"
SCALE = 0.8
RANK = 4
ALPHA = 8.0


def save_ref(folder, name, tensor, index):
    os.makedirs(folder, exist_ok=True)
    array = tensor.detach().to(torch.float32).cpu().numpy()
    array.astype("<f4").tofile(os.path.join(folder, name + ".f32"))
    index[name] = list(array.shape)


def random_lora(module_shapes, generator):
    """module name -> (down [r, in], up [out, r]) random LoRA matrices (up not zero, unlike a fresh adapter)."""
    out = {}
    for name, (rows, columns) in module_shapes.items():
        down = torch.randn(RANK, columns, generator=generator) * 0.3
        up = torch.randn(rows, RANK, generator=generator) * 0.3
        out[name] = (down, up)
    return out


def linear_shapes(model, pattern):
    return {name: tuple(m.weight.shape) for name, m in model.named_modules() if isinstance(m, torch.nn.Linear) and re.search(pattern, name)}


def peft_state(prefix, lora):
    state = {}
    for name, (down, up) in lora.items():
        state[f"{prefix}.{name}.lora_A.weight"] = down.contiguous()
        state[f"{prefix}.{name}.lora_B.weight"] = up.contiguous()
    return state


def kohya_state(prefix, lora, rename=lambda n: n):
    state = {}
    for name, (down, up) in lora.items():
        key = prefix + rename(name).replace(".", "_")
        state[key + ".lora_down.weight"] = down.contiguous()
        state[key + ".lora_up.weight"] = up.contiguous()
        state[key + ".alpha"] = torch.tensor(ALPHA)
    return state


def sgm_name(name, layers_per_block=2):
    """A diffusers SDXL UNet module name -> the SGM UNet's (inverse of diffusers' _maybe_map_sgm_blocks_to_diffusers)."""
    m = re.match(r"down_blocks\.(\d+)\.attentions\.(\d+)\.(.*)", name)
    if m:
        return f"input_blocks.{int(m[1]) * (layers_per_block + 1) + int(m[2]) + 1}.1.{m[3]}"
    m = re.match(r"up_blocks\.(\d+)\.attentions\.(\d+)\.(.*)", name)
    if m:
        return f"output_blocks.{int(m[1]) * (layers_per_block + 1) + int(m[2])}.1.{m[3]}"
    m = re.match(r"mid_block\.attentions\.0\.(.*)", name)
    if m:
        return f"middle_block.1.{m[1]}"
    raise ValueError(name)


def scale_lora(lora):
    """The LoRA as diffusers applies kohya's alpha: PEFT files carry no alpha (scale 1), so the PEFT matrices are pre-scaled."""
    return {k: (d, u * (ALPHA / RANK)) for k, (d, u) in lora.items()}


def run_sd(root, family, out):
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    from safetensors.torch import save_file

    xl = family == "sdxl"
    pipe = (StableDiffusionXLPipeline if xl else StableDiffusionPipeline).from_pretrained(os.path.join(root, "diffusers"), torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    g = torch.Generator().manual_seed(11)
    attention = r"attentions\.\d+\.transformer_blocks\.\d+\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)$|attentions\.\d+\.transformer_blocks\.\d+\.ff\.net\.2$"
    unet = random_lora(linear_shapes(pipe.unet, attention), g)
    text = random_lora(linear_shapes(pipe.text_encoder, r"self_attn\.(q_proj|k_proj|v_proj|out_proj)$|mlp\.fc1$"), g)
    # diffusers recognizes a kohya text-encoder LoRA by its out_proj modules: the second encoder gets all four projections.
    text2 = random_lora(linear_shapes(pipe.text_encoder_2, r"self_attn\.(q_proj|k_proj|v_proj|out_proj)$"), g) if xl else {}
    folder = os.path.join(out, family)
    os.makedirs(folder, exist_ok=True)
    peft = {**peft_state("unet", scale_lora(unet)), **peft_state("text_encoder", scale_lora(text))}
    kohya = {**kohya_state("lora_unet_", unet, sgm_name if xl else (lambda n: n)), **kohya_state("lora_te1_" if xl else "lora_te_", text)}
    if xl:
        peft.update(peft_state("text_encoder_2", scale_lora(text2)))
        kohya.update(kohya_state("lora_te2_", text2))
    save_file(peft, os.path.join(folder, "peft.safetensors"))
    save_file(kohya, os.path.join(folder, "kohya.safetensors"))

    size = 64 if not xl else 64
    noise = torch.randn(1, 4, size // 8, size // 8, generator=torch.Generator().manual_seed(42))
    index = {}
    ref = os.path.join(folder, "reference")
    save_ref(ref, "noise", noise, index)
    kwargs = dict(height=size, width=size, num_inference_steps=4, guidance_scale=7.5, output_type="pt")
    save_ref(ref, "image_base", pipe(PROMPT, latents=noise.clone(), **kwargs).images[0], index)
    for name in ("peft", "kohya"):
        pipe.load_lora_weights(folder, weight_name=f"{name}.safetensors", adapter_name=name)
        pipe.set_adapters([name], adapter_weights=[SCALE])
        save_ref(ref, f"image_{name}", pipe(PROMPT, latents=noise.clone(), **kwargs).images[0], index)
        pipe.delete_adapters(name)
    with open(os.path.join(ref, "index.json"), "w") as f:
        json.dump({"tensors": index, "prompt": PROMPT, "scale": SCALE, "steps": 4, "guidance": 7.5, "size": size,
                   "scheduler": pipe.scheduler.__class__.__name__}, f, indent=1)


def run_flux(root, out):
    from diffusers import FluxPipeline
    from safetensors.torch import save_file

    pipe = FluxPipeline.from_pretrained(os.path.join(root, "diffusers"), torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    t = pipe.transformer
    g = torch.Generator().manual_seed(12)
    double = random_lora(linear_shapes(t, r"^transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj|to_out\.0|to_add_out)|ff\.net\.2)$"), g)
    single = random_lora(linear_shapes(t, r"^single_transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v)|proj_mlp|proj_out)$"), g)
    folder = os.path.join(out, "flux")
    os.makedirs(folder, exist_ok=True)
    save_file(peft_state("transformer", scale_lora({**double, **single})), os.path.join(folder, "peft.safetensors"))

    # kohya / BFL: one LoRA per fused projection; q, k, v as one module whose up matrix stacks the three (block-diagonal rank 3r).
    kohya = {}

    def put(key, down, up):
        kohya[f"lora_unet_{key}.lora_down.weight"] = down.contiguous()
        kohya[f"lora_unet_{key}.lora_up.weight"] = up.contiguous()
        kohya[f"lora_unet_{key}.alpha"] = torch.tensor(ALPHA * down.shape[0] / RANK)

    def fused(parts):
        downs = [d for d, _ in parts]
        ups = [u for _, u in parts]
        rank = sum(d.shape[0] for d in downs)
        up = torch.zeros(sum(u.shape[0] for u in ups), rank)
        r = o = 0
        for d, u in parts:
            up[o:o + u.shape[0], r:r + d.shape[0]] = u
            r += d.shape[0]
            o += u.shape[0]
        return torch.cat(downs, 0), up

    blocks = len(t.transformer_blocks)
    for i in range(blocks):
        p = f"transformer_blocks.{i}."
        put(f"double_blocks_{i}_img_attn_qkv", *fused([double[p + f"attn.to_{x}"] for x in "qkv"]))
        put(f"double_blocks_{i}_txt_attn_qkv", *fused([double[p + f"attn.add_{x}_proj"] for x in "qkv"]))
        put(f"double_blocks_{i}_img_attn_proj", *double[p + "attn.to_out.0"])
        put(f"double_blocks_{i}_txt_attn_proj", *double[p + "attn.to_add_out"])
        put(f"double_blocks_{i}_img_mlp_2", *double[p + "ff.net.2"])
    for i in range(len(t.single_transformer_blocks)):
        p = f"single_transformer_blocks.{i}."
        # diffusers' kohya converter splits linear1 with FLUX.1's real widths (3 x 3072 + 12288), so the tiny file leaves it out
        # (the PEFT file adapts q / k / v / proj_mlp, the row blocks of linear1).
        put(f"single_blocks_{i}_linear2", *single[p + "proj_out"])
    save_file(kohya, os.path.join(folder, "kohya.safetensors"))

    index = {}
    ref = os.path.join(folder, "reference")
    h, w = 64, 96
    noise = torch.randn(1, 16, h // 8, w // 8, generator=torch.Generator().manual_seed(42))
    save_ref(ref, "noise", noise, index)
    packed = FluxPipeline._pack_latents(noise, 1, 16, h // 8, w // 8)
    kwargs = dict(height=h, width=w, num_inference_steps=4, guidance_scale=3.5, output_type="pt", max_sequence_length=512)
    save_ref(ref, "image_base", pipe(PROMPT, latents=packed.clone(), **kwargs).images[0], index)
    for name in ("peft", "kohya"):
        pipe.load_lora_weights(folder, weight_name=f"{name}.safetensors", adapter_name=name)
        pipe.set_adapters([name], adapter_weights=[SCALE])
        save_ref(ref, f"image_{name}", pipe(PROMPT, latents=packed.clone(), **kwargs).images[0], index)
        pipe.delete_adapters(name)
    with open(os.path.join(ref, "index.json"), "w") as f:
        json.dump({"tensors": index, "prompt": PROMPT, "scale": SCALE, "steps": 4, "guidance": 3.5, "height": h, "width": w,
                   "length": 512}, f, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/lora/tiny-lora"))
    parser.add_argument("--families", nargs="+", default=["sd1", "sdxl", "flux"])
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    home = os.path.expanduser("~/models")
    for family in args.families:
        if family == "flux":
            run_flux(os.path.join(home, "flux/tiny-random-flux/dev"), args.out)
        else:
            run_sd(os.path.join(home, "sd/tiny-random-sd" if family == "sd1" else "sd/tiny-random-sdxl"), family, args.out)
        print("written", family)


if __name__ == "__main__":
    main()
