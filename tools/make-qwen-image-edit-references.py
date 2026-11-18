#!/usr/bin/env python3
"""Writes diffusers reference outputs of a Qwen-Image-Edit-Plus pipeline (Qwen-Image-Edit-2509 / -2511, FireRed-Image-Edit) for the Sharplight
image-generation tests.

MODEL is a Qwen-Image-Edit diffusers folder (configurations, text_encoder/, processor/, tokenizer/, vae/, scheduler/; its transformer/
weights are not needed when --transformer-gguf gives a transformer-only GGUF, dequantized on the fly by diffusers). Writes OUT:

  ref_{i}.png                         the input images (synthetic scenes of 640 x 480 pixels)
  noise                               initial latents [1, 16, H/8, W/8] (seed 42)
  image_{n}                           the edit with the first n images (n = 1, 2): [3, H, W] in [0, 1]; true_cfg_scale 4, negative prompt " "
  condition_0.png, vision_pixels_0 / vision_embeds_0   the vision tower alone: the first picture at the language model's size, transformers'
                                      normalized patches and merged-token embeddings of it
  text_hidden_{n} / negative_hidden_{n} the text encoder's rows (after the template) of the prompt and the negative prompt with n pictures
  index.json                          prompt, size, steps, the expanded prompt token ids with the images

The reference images are encoded at the output size (the module's VAE_IMAGE_SIZE is set to the output's area: Sharplight resizes the
references to the output's size); the language model reads them at 384 x 384 pixels as the pipeline does. The text encoder runs first (on
--text-device, float32), then is freed; the transformer and the VAE run on --device.
Usage: make-qwen-image-edit-references.py MODEL OUT [--transformer-gguf FILE] [--size 512 384] [--steps 20] [--device cuda]
"""

import argparse
import gc
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

PROMPT = "turn the red circle blue and put a small yellow star above the square"
NEGATIVE = " "


def scene(seed):
    rng = np.random.default_rng(seed)
    w, h = 640, 480
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([0.35 + 0.4 * x / w, 0.45 + 0.3 * y / h, 0.75 - 0.3 * x / w * y / h], -1) if seed == 1 else np.stack([0.8 - 0.4 * y / h, 0.5 * x / w, 0.4 + 0.2 * y / h], -1)
    image = Image.fromarray((np.clip(base + rng.normal(0, 0.015, base.shape), 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.ellipse([w * 0.12, h * 0.25, w * 0.45, h * 0.75], fill=(220, 50, 40))
    draw.rectangle([w * 0.55, h * 0.45, w * 0.88, h * 0.85], fill=(60, 140, 90) if seed == 1 else (200, 160, 40))
    draw.polygon([(w * 0.5, h * 0.1), (w * 0.62, h * 0.4), (w * 0.38, h * 0.4)], fill=(240, 230, 120))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--transformer-gguf", default=None)
    parser.add_argument("--size", type=int, nargs=2, default=[512, 384], help="width height")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--bicubic-condition", action="store_true",
                        help="resize the pictures the language model reads with PIL's bicubic instead of the pipeline's Lanczos: how far the images move "
                             "under the resampling differences (a few gray levels) Sharplight's own resize has; written as image_{n}_bicubic")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)
    width, height = args.size

    from diffusers import (AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, QwenImageEditPlusPipeline,
                           QwenImageTransformer2DModel)
    from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as module
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

    dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    processor = Qwen2VLProcessor.from_pretrained(args.model, subfolder="processor")
    text = Qwen2_5_VLForConditionalGeneration.from_pretrained(os.path.join(args.model, "text_encoder"), torch_dtype=dtype).to(args.text_device)
    vae = AutoencoderKLQwenImage.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
    pipe = QwenImageEditPlusPipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, processor=processor, transformer=None)
    module.VAE_IMAGE_SIZE = width * height
    images = [scene(1), scene(2)]
    for i, image in enumerate(images):
        image.save(os.path.join(args.out, f"ref_{i}.png"))

    cw, ch = module.calculate_dimensions(module.CONDITION_IMAGE_SIZE, 640 / 480)
    index = {"prompt": PROMPT, "negative": NEGATIVE, "guidance": args.guidance, "steps": args.steps, "width": width, "height": height,
             "condition_size": [cw, ch]}
    if not args.bicubic_condition:
        # The vision tower alone on the first condition picture: the picture, transformers' patches and merged-token embeddings.
        first = pipe.image_processor.resize(images[0], ch, cw)
        first.save(os.path.join(args.out, "condition_0.png"))
        inputs = processor(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[first], return_tensors="pt")
        tower = text.model.visual(inputs["pixel_values"].to(args.text_device), grid_thw=inputs["image_grid_thw"].to(args.text_device))
        np.asarray(inputs["pixel_values"].numpy(), dtype="<f4").tofile(os.path.join(args.out, "vision_pixels_0.f32"))
        np.asarray((tower.pooler_output if hasattr(tower, "pooler_output") else tower).float().cpu().numpy(), dtype="<f4").tofile(
            os.path.join(args.out, "vision_embeds_0.f32"))
    embeds, negatives = {}, {}
    for n in args.counts:
        condition = [pipe.image_processor.resize(image, ch, cw) for image in images[:n]]
        if args.bicubic_condition:
            condition = [image.resize((cw, ch), Image.BICUBIC) for image in images[:n]]
        label = "".join(f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>" for i in range(n))
        inputs = processor(text=[pipe.prompt_template_encode.format(label + PROMPT)], images=condition, padding=True, return_tensors="pt")
        index[f"ids_{n}"] = inputs["input_ids"][0].tolist()
        index[f"grid_{n}"] = inputs["image_grid_thw"].tolist()
        embeds[n] = pipe.encode_prompt(image=condition, prompt=PROMPT, device=args.text_device)
        negatives[n] = pipe.encode_prompt(image=condition, prompt=NEGATIVE, device=args.text_device)
        if not args.bicubic_condition:
            np.asarray(embeds[n][0][0].float().numpy(), dtype="<f4").tofile(os.path.join(args.out, f"text_hidden_{n}.f32"))
        if not args.bicubic_condition:
            np.asarray(negatives[n][0][0].float().numpy(), dtype="<f4").tofile(os.path.join(args.out, f"negative_hidden_{n}.f32"))
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
    index["zero_cond_t"] = bool(transformer.config.zero_cond_t)

    lh, lw = height // 8, width // 8
    noise = torch.randn(1, 16, lh, lw, generator=torch.Generator().manual_seed(42))
    np.asarray(noise.numpy(), dtype="<f4").tofile(os.path.join(args.out, "noise.f32"))
    packed = QwenImageEditPlusPipeline._pack_latents(noise, 1, 16, lh, lw).to(device)
    for n in args.counts:
        image = pipe(image=images[:n], prompt_embeds=embeds[n][0].to(device), prompt_embeds_mask=None if embeds[n][1] is None else embeds[n][1].to(device),
                     negative_prompt_embeds=negatives[n][0].to(device),
                     negative_prompt_embeds_mask=None if negatives[n][1] is None else negatives[n][1].to(device),
                     true_cfg_scale=args.guidance, height=height, width=width, num_inference_steps=args.steps, latents=packed.clone(),
                     output_type="pt").images
        np.asarray(image[0].float().cpu().numpy(), dtype="<f4").tofile(os.path.join(args.out, f"image_{n}{'_bicubic' if args.bicubic_condition else ''}.f32"))
        index["timesteps"] = [float(t) for t in pipe.scheduler.timesteps]
        with open(os.path.join(args.out, "index.json"), "w") as f:
            json.dump(index, f, indent=1)
    print("written to", args.out)


if __name__ == "__main__":
    main()
