#!/usr/bin/env python3
"""Creates random LyCORIS (LoCon, LoHa, LoKr, with Tucker cores) and DoRA files for the tiny random SD 1.x pipeline and PEFT reference
images and merged weights for them.

Every variant is a PEFT adapter on the tiny UNet's attention projections and residual convolutions (3x3) with random (non-zero) factors,
merged with PEFT's merge_and_unload at strength 0.8 (PEFT's set_scale: the strength multiplies alpha / r, inside DoRA's norm); the files are
written in kohya / LyCORIS naming (lora_unet_<module with underscores>.<part> plus .alpha), the layout of the files LyCORIS and kohya-ss
produce, and DoRA also in PEFT's own names (unet.<module>.lora_A / lora_B / lora_magnitude_vector).

  locon      plain LoRA on linears and convolutions (lora_down / lora_up)
  loha       LoHa, the convolutions with Tucker cores (hada_w1_a/b, hada_w2_a/b, hada_t1/t2)
  lokr       LoKr with the second factor decomposed, the convolutions' with a Tucker core (lokr_w1, lokr_w2_a/b, lokr_t2)
  dora       DoRA (LoRA with lora_magnitude_vector / dora_scale)

Written to OUT (default ~/models/lora/tiny-lycoris):

  {variant}.safetensors (and dora-peft.safetensors)
  reference/{noise,image_base,image_<variant>}.f32, reference/<variant>_<weight>.f32 (merged weights) + index.json

Requires torch, diffusers, peft and safetensors and the tiny pipeline (tools/make-tiny-diffusion-models.py).
"""

import argparse
import json
import os

import torch

PROMPT = "a photograph of an astronaut riding a horse on the moon"
SCALE = 0.8
RANK = 4
ALPHA = 8.0
# LoKr at rank 2 decomposes the second factor of the tiny model's 32-wide layers (as LoKr files of real models do), so the file's alpha
# applies; LyCORIS writes alpha = rank (factor 1) for layers whose factors are both whole, which PEFT's alpha / r does not model.
LOKR_RANK = 2
LOKR_ALPHA = 4.0
TARGETS = r".*(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)|.*resnets\.\d+\.conv1"
# Merged weights compared directly: diffusers name -> Sharplight's canonical name.
CHECKED = {
    "down_blocks.0.resnets.0.conv1.weight": "unet.down.0.res.0.conv1.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight": "unet.down.0.attn.0.blocks.0.self.q.weight",
}


def save_ref(folder, name, tensor, index):
    os.makedirs(folder, exist_ok=True)
    array = tensor.detach().to(torch.float32).cpu().numpy()
    array.astype("<f4").tofile(os.path.join(folder, name + ".f32"))
    index[name] = list(array.shape)


def config(variant):
    from peft import LoHaConfig, LoKrConfig, LoraConfig

    if variant == "loha":
        return LoHaConfig(r=RANK, alpha=ALPHA, target_modules=TARGETS, use_effective_conv2d=True, init_weights=False)
    if variant == "lokr":
        return LoKrConfig(r=LOKR_RANK, alpha=LOKR_ALPHA, target_modules=TARGETS, use_effective_conv2d=True, decompose_both=True, decompose_factor=4,
                          init_weights=False)
    return LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=TARGETS, use_dora=variant == "dora", init_lora_weights=False)


def randomize(model, variant, generator):
    """Random adapter factors (a fresh adapter is a no-op); DoRA's magnitudes perturbed around the weights' row norms."""
    for name, parameter in model.named_parameters():
        if any(part in name for part in ("hada_", "lokr_", "lora_A", "lora_B")):
            parameter.data = torch.randn(parameter.shape, generator=generator) * (0.3 if variant != "lokr" else 0.5)
        elif "lora_magnitude_vector" in name:
            parameter.data = parameter.data * (1 + 0.2 * torch.randn(parameter.shape, generator=generator))


def kohya_state(state, variant):
    """PEFT's state dict (base_model.model.<module>.<part>) in kohya / LyCORIS names."""
    out = {}
    modules = set()
    for key, value in state.items():
        module, part = key[len("base_model.model."):].rsplit(".", 1)
        if part == "weight":
            module, part = module.rsplit(".", 1)
        if part == "lora_magnitude_vector":
            part = "dora_scale"
            value = value.reshape(-1, 1)
        part = {"lora_A": "lora_down.weight", "lora_B": "lora_up.weight"}.get(part, part)
        kohya = "lora_unet_" + module.replace(".", "_")
        out[f"{kohya}.{part}"] = value.contiguous()
        modules.add(kohya)
    for kohya in modules:
        out[f"{kohya}.alpha"] = torch.tensor(LOKR_ALPHA if variant == "lokr" else ALPHA)
    return out


def peft_file_state(state):
    """PEFT's state dict in diffusers' LoRA file names (unet.<module>.lora_A.weight ...); B pre-scaled by alpha / r (no alpha in the file)."""
    out = {}
    for key, value in state.items():
        name = "unet." + key[len("base_model.model."):]
        if ".lora_B." in name:
            value = value * (ALPHA / RANK)
        out[name] = value.contiguous()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", default=os.path.expanduser("~/models/sd/tiny-random-sd/diffusers"))
    parser.add_argument("--out", default=os.path.expanduser("~/models/lora/tiny-lycoris"))
    args = parser.parse_args()

    from diffusers import StableDiffusionPipeline
    from peft import get_peft_model, get_peft_model_state_dict
    from safetensors.torch import save_file

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    ref = os.path.join(args.out, "reference")
    index = {}
    size = 64
    noise = torch.randn(1, 4, size // 8, size // 8, generator=torch.Generator().manual_seed(42))
    save_ref(ref, "noise", noise, index)
    kwargs = dict(height=size, width=size, num_inference_steps=4, guidance_scale=7.5, output_type="pt")
    base = StableDiffusionPipeline.from_pretrained(args.pipeline, torch_dtype=torch.float32, safety_checker=None)
    base.set_progress_bar_config(disable=True)
    save_ref(ref, "image_base", base(PROMPT, latents=noise.clone(), **kwargs).images[0], index)

    for seed, variant in enumerate(["locon", "loha", "lokr", "dora"]):
        pipe = StableDiffusionPipeline.from_pretrained(args.pipeline, torch_dtype=torch.float32, safety_checker=None)
        pipe.set_progress_bar_config(disable=True)
        model = get_peft_model(pipe.unet, config(variant))
        randomize(model, variant, torch.Generator().manual_seed(100 + seed))
        state = get_peft_model_state_dict(model)
        save_file(kohya_state(state, variant), os.path.join(args.out, f"{variant}.safetensors"))
        if variant == "dora":
            save_file(peft_file_state(state), os.path.join(args.out, "dora-peft.safetensors"))
        for module in model.modules():
            if hasattr(module, "set_scale") and hasattr(module, "scaling"):
                module.set_scale("default", SCALE)
        merged = model.merge_and_unload()
        weights = dict(merged.named_parameters())
        for diffusers_name, canonical in CHECKED.items():
            save_ref(ref, f"{variant}_{canonical}", weights[diffusers_name], index)
        pipe.unet = merged
        save_ref(ref, f"image_{variant}", pipe(PROMPT, latents=noise.clone(), **kwargs).images[0], index)
        print(variant, "written")
    with open(os.path.join(ref, "index.json"), "w") as f:
        json.dump({"tensors": index, "prompt": PROMPT, "scale": SCALE, "steps": 4, "guidance": 7.5, "size": size, "checked": CHECKED,
                   "scheduler": base.scheduler.__class__.__name__}, f, indent=1)


if __name__ == "__main__":
    main()
