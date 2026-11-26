#!/usr/bin/env python3
"""Writes diffusers reference outputs of ControlNet-conditioned SD 1.x / SDXL generation for the Sharplight image-generation tests.

  sd15   StableDiffusionControlNetPipeline (e.g. stable-diffusion-v1-5 + lllyasviel/control_v11p_sd15_canny)
  sdxl   StableDiffusionXLControlNetPipeline (e.g. stable-diffusion-xl-base-1.0 + diffusers/controlnet-canny-sdxl-1.0)

Usage: make-controlnet-references.py {sd15|sdxl} PIPELINE CONTROLNET CONTROL_IMAGE OUT [--width W] [--height H] [--steps 20]
       [--guidance G] [--scale S] [--device cuda]

The pipeline folder may lack model_index.json (it is assembled from its components). Everything runs in float32 (fp16 files upcast) with
Euler from shared initial noise. Written to OUT as raw little-endian f32 plus index.json:

  control.png              the control image at the output size (the CONTROL_IMAGE resized with Lanczos), the test's input
  noise                    unit Gaussian noise [1, 4, H/8, W/8]
  unet_latent/_timestep/_context/_control/_scale (and SDXL's _pooled/_time_ids), unet_out
                           one ControlNet + UNet evaluation on seeded random inputs (batch 2, the control image in both rows, the
                           scales [S, S / 2]): the UNet's prediction with the ControlNet's residuals
  unet_plain_out           the same UNet evaluation without the ControlNet
  image                    the pipeline's image [3, H, W] in [0, 1] (controlnet_conditioning_scale S)
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image


def write(out, name, tensor):
    array = tensor.detach().to("cpu", torch.float32).contiguous().numpy()
    array.tofile(os.path.join(out, name + ".f32"))
    return list(array.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=["sd15", "sdxl"])
    parser.add_argument("pipeline")
    parser.add_argument("controlnet")
    parser.add_argument("control")
    parser.add_argument("out")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from diffusers import AutoencoderKL, ControlNetModel, EulerDiscreteScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    width, height, device = args.width, args.height, args.device
    xl = args.family == "sdxl"
    f32 = torch.float32

    control = Image.open(args.control).convert("RGB").resize((width, height), Image.LANCZOS)
    control.save(os.path.join(args.out, "control.png"))

    def component(cls, name, **kwargs):
        folder = os.path.join(args.pipeline, name)
        variant = "fp16" if not any(f.endswith(".safetensors") and ".fp16." not in f for f in os.listdir(folder)) else None
        return cls.from_pretrained(folder, torch_dtype=f32, variant=variant, **kwargs)

    controlnet_folder = args.controlnet
    controlnet_variant = "fp16" if not os.path.exists(os.path.join(controlnet_folder, "diffusion_pytorch_model.safetensors")) else None
    controlnet = ControlNetModel.from_pretrained(controlnet_folder, torch_dtype=f32, variant=controlnet_variant)
    unet = component(UNet2DConditionModel, "unet")
    parts = dict(vae=component(AutoencoderKL, "vae"), unet=unet, controlnet=controlnet,
                 scheduler=EulerDiscreteScheduler.from_pretrained(os.path.join(args.pipeline, "scheduler")),
                 tokenizer=CLIPTokenizer.from_pretrained(os.path.join(args.pipeline, "tokenizer")))
    if xl:
        from diffusers import StableDiffusionXLControlNetPipeline

        parts.update(text_encoder=component(CLIPTextModel, "text_encoder"), text_encoder_2=component(CLIPTextModelWithProjection, "text_encoder_2"),
                     tokenizer_2=CLIPTokenizer.from_pretrained(os.path.join(args.pipeline, "tokenizer_2")))
        pipe = StableDiffusionXLControlNetPipeline(**parts)
        pipe.enable_model_cpu_offload()
        prompt = "a colorful bird sitting on a branch, detailed feathers, photograph"
    else:
        from diffusers import StableDiffusionControlNetPipeline

        parts.update(text_encoder=component(CLIPTextModel, "text_encoder"), safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        pipe = StableDiffusionControlNetPipeline(**parts).to(device)
        prompt = "a colorful bird sitting on a branch, detailed feathers, photograph"
    negative = "blurry, low quality"
    index = {"family": args.family, "width": width, "height": height, "steps": args.steps, "guidance_scale": args.guidance, "scale": args.scale,
             "prompt": prompt, "negative_prompt": negative, "sampler": pipe.scheduler.__class__.__name__, "tensors": {}}
    tensors = index["tensors"]

    # One ControlNet + UNet evaluation on seeded random inputs.
    generator = torch.Generator("cpu").manual_seed(11)
    rows = 2
    latent = torch.randn(rows, 4, height // 8, width // 8, generator=generator)
    timestep = torch.tensor([801.0, 801.0])
    context_width = unet.config.cross_attention_dim
    context = torch.randn(rows, 77, context_width, generator=generator)
    pixels = torch.from_numpy(np.asarray(control, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].repeat(rows, 1, 1, 1)
    scales = [args.scale, args.scale / 2]
    added = {}
    if xl:
        pooled = torch.randn(rows, 1280, generator=generator)
        time_ids = torch.tensor([[height, width, 0, 0, height, width]] * rows, dtype=f32)
        added = {"text_embeds": pooled.to(device), "time_ids": time_ids.to(device)}
        tensors["unet_pooled"] = write(args.out, "unet_pooled", pooled)
        tensors["unet_time_ids"] = write(args.out, "unet_time_ids", time_ids)
    with torch.no_grad():
        unet.to(device)
        controlnet.to(device)
        downs, mids = [], []
        for row in range(rows):
            d, m = controlnet(latent[row:row + 1].to(device), timestep[row:row + 1].to(device), encoder_hidden_states=context[row:row + 1].to(device),
                              controlnet_cond=pixels[row:row + 1].to(device), conditioning_scale=scales[row],
                              added_cond_kwargs={k: v[row:row + 1] for k, v in added.items()} if xl else None, return_dict=False)
            downs.append(d)
            mids.append(m)
        down = [torch.cat([downs[r][i] for r in range(rows)]) for i in range(len(downs[0]))]
        mid = torch.cat(mids)
        out = unet(latent.to(device), timestep.to(device), encoder_hidden_states=context.to(device), down_block_additional_residuals=down,
                   mid_block_additional_residual=mid, added_cond_kwargs=added if xl else None).sample
        plain = unet(latent.to(device), timestep.to(device), encoder_hidden_states=context.to(device), added_cond_kwargs=added if xl else None).sample
        if xl:
            unet.to("cpu")
            controlnet.to("cpu")
    tensors["unet_latent"] = write(args.out, "unet_latent", latent)
    tensors["unet_timestep"] = write(args.out, "unet_timestep", timestep)
    tensors["unet_context"] = write(args.out, "unet_context", context)
    tensors["unet_control"] = write(args.out, "unet_control", pixels)
    tensors["unet_scale"] = write(args.out, "unet_scale", torch.tensor(scales))
    tensors["unet_out"] = write(args.out, "unet_out", out)
    tensors["unet_plain_out"] = write(args.out, "unet_plain_out", plain)

    noise = torch.randn(1, 4, height // 8, width // 8, generator=torch.Generator("cpu").manual_seed(42))
    with torch.no_grad():
        image = pipe(prompt, negative_prompt=negative, image=control, height=height, width=width, num_inference_steps=args.steps,
                     guidance_scale=args.guidance, controlnet_conditioning_scale=args.scale, latents=noise, output_type="pt").images[0]
    tensors["noise"] = write(args.out, "noise", noise)
    tensors["image"] = write(args.out, "image", image)
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print("written", args.out)


if __name__ == "__main__":
    main()
