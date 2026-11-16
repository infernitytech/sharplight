#!/usr/bin/env python3
"""Diffusers references of the image-conditioned Wan models running transformer GGUF files, for the video-generation tests.

Modes:
  i2v     Wan 2.1 I2V-14B (one transformer over the concatenated channels, CLIP ViT-H/14 image states in every cross-attention)
  flf2v   Wan 2.1 FLF2V-14B (as i2v, conditioned on a first and a last frame)
  a14b    Wan 2.2 I2V-A14B (two experts over the concatenated channels, no image encoder; LOW is the low-noise expert)

The transformers come from GGUF files (city96 / QuantStack, dequantized on the fly by diffusers' GGUF support, computed in f32),
configured from the model's diffusers configuration folder (CONFIG: transformer/config.json, scheduler/, model_index.json and, for
i2v / flf2v, image_encoder/ and image_processor/); the umT5 encoder, tokenizer and VAE from a Wan 2.1 diffusers folder. The start
image is START (and the last image LAST for flf2v). Writes to OUT/reference: text_hidden, negative_hidden, image_hidden (i2v, flf2v),
noise, the first conditional evaluation of each transformer (transformer_latent [C', T, h, w] with the concatenated conditioning,
transformer_timestep, transformer_out; transformer_2_* for a14b), video [F, 3, H, W], index.json.

Usage: make-wan-i2v-references.py MODE HIGH.gguf CONFIG WAN21_FOLDER START OUT [--low LOW.gguf] [--last LAST]
       [--frames 17] [--height 480] [--width 832] [--steps 20]
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

PROMPT = "The cat slowly turns its head and blinks, soft window light, cinematic"
NEGATIVE = "blurry, low quality, distorted"


def save(folder, index, name, tensor):
    array = tensor.detach().to(torch.float32).cpu().numpy()
    array.astype("<f4").tofile(os.path.join(folder, name + ".f32"))
    index[name] = list(array.shape)


def patch_single_file_conversion():
    """diffusers' Wan single-file conversion leaves FLF2V's image position table (img_emb.emb_pos) unmapped: map it to pos_embed."""
    import diffusers.loaders.single_file_model as single_file_model

    for entry in single_file_model.SINGLE_FILE_LOADABLE_CLASSES.values():
        original = entry.get("checkpoint_mapping_fn")
        if original is None or original.__name__ != "convert_wan_transformer_to_diffusers":
            continue

        def mapped(checkpoint, _original=original, **kwargs):
            converted = _original(checkpoint, **kwargs)
            if "img_emb.emb_pos" in converted:
                converted["condition_embedder.image_embedder.pos_embed"] = converted.pop("img_emb.emb_pos")
            return converted

        mapped.__name__ = original.__name__
        entry["checkpoint_mapping_fn"] = mapped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["i2v", "flf2v", "a14b"])
    parser.add_argument("high")
    parser.add_argument("config")
    parser.add_argument("components")
    parser.add_argument("start")
    parser.add_argument("out")
    parser.add_argument("--low")
    parser.add_argument("--last")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--guidance-2", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from diffusers import AutoencoderKLWan, GGUFQuantizationConfig, UniPCMultistepScheduler, WanImageToVideoPipeline, WanTransformer3DModel
    import diffusers.pipelines.wan.pipeline_wan as t2v_module
    import diffusers.pipelines.wan.pipeline_wan_i2v as i2v_module
    from PIL import Image
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

    i2v_module.prompt_clean = t2v_module.prompt_clean  # the I2V module calls ftfy without importing it
    patch_single_file_conversion()
    torch.set_grad_enabled(False)
    # Plain f32 on CUDA: TF32 convolutions alone move CLIP ViT-H's states by 1e-2 relative.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = os.path.join(args.out, "reference")
    os.makedirs(folder, exist_ok=True)
    index = {}
    quantization = GGUFQuantizationConfig(compute_dtype=torch.float32)
    high = WanTransformer3DModel.from_single_file(args.high, quantization_config=quantization, config=os.path.join(args.config, "transformer"),
                                                  torch_dtype=torch.float32)
    low = None
    if args.mode == "a14b":
        low = WanTransformer3DModel.from_single_file(args.low, quantization_config=quantization, config=os.path.join(args.config, "transformer_2"),
                                                     torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.components, "tokenizer"))
    text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(args.components, "text_encoder"), torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(os.path.join(args.components, "vae"), torch_dtype=torch.float32)
    images = {}
    if args.mode != "a14b":
        images = {"image_encoder": CLIPVisionModel.from_pretrained(os.path.join(args.config, "image_encoder"), torch_dtype=torch.float32),
                  "image_processor": CLIPImageProcessor.from_pretrained(os.path.join(args.config, "image_processor"))}
    with open(os.path.join(args.config, "scheduler", "scheduler_config.json")) as f:
        scheduler = UniPCMultistepScheduler.from_config(json.load(f))
    with open(os.path.join(args.config, "model_index.json")) as f:
        boundary = json.load(f).get("boundary_ratio")
    experts = {"transformer_2": low, "boundary_ratio": boundary} if low is not None else {}
    pipe = WanImageToVideoPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler, transformer=high, **images, **experts)
    pipe.enable_model_cpu_offload()
    start = Image.open(args.start).convert("RGB")
    last = Image.open(args.last).convert("RGB") if args.last else None
    shutil.copy(args.start, os.path.join(folder, "start.png"))
    if last is not None:
        shutil.copy(args.last, os.path.join(folder, "last.png"))
    embeds, negative = pipe.encode_prompt(PROMPT, NEGATIVE, do_classifier_free_guidance=True, max_sequence_length=512)
    save(folder, index, "text_hidden", embeds[0])
    save(folder, index, "negative_hidden", negative[0])
    image_embeds = None
    if images:
        image_embeds = pipe.encode_image([start] if last is None else [start, last], "cuda")
        save(folder, index, "image_hidden", image_embeds.reshape(-1, image_embeds.shape[-1]))
    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()

    latent_frames = (args.frames - 1) // 4 + 1
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(1, 16, latent_frames, args.height // 8, args.width // 8, generator=generator)
    save(folder, index, "noise", noise[0])

    captured = {}

    def capture(name):
        def hook(module, inputs, kwargs, output):
            if name in captured:
                return
            captured[name] = True
            save(folder, index, name + "_latent", kwargs["hidden_states"][0])
            save(folder, index, name + "_timestep", kwargs["timestep"].reshape(-1)[:1].float())
            save(folder, index, name + "_out", output[0][0])
        return hook

    high.register_forward_hook(capture("transformer"), with_kwargs=True)
    if low is not None:
        low.register_forward_hook(capture("transformer_2"), with_kwargs=True)
    start_time = time.time()
    extra = {"guidance_scale_2": args.guidance_2} if low is not None and args.guidance_2 is not None else {}
    video = pipe(image=start, last_image=last, prompt_embeds=embeds, negative_prompt_embeds=negative, height=args.height,
                 width=args.width, num_frames=args.frames, num_inference_steps=args.steps, guidance_scale=args.guidance, latents=noise.clone(),
                 output_type="np", **extra).frames[0]
    elapsed = time.time() - start_time
    save(folder, index, "video", torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2))
    meta = {"tensors": index, "prompt": PROMPT, "negative_prompt": NEGATIVE, "frames": args.frames, "height": args.height, "width": args.width,
            "steps": args.steps, "guidance_scale": args.guidance, "seed": args.seed, "seconds": elapsed, "image": "reference/start.png", "mode": args.mode}
    if last is not None:
        meta["last_image"] = "reference/last.png"
    if low is not None:
        meta["guidance_scale_2"] = args.guidance_2 if args.guidance_2 is not None else args.guidance
        meta["boundary_ratio"] = boundary
    with open(os.path.join(folder, "index.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {args.out} ({elapsed:.1f} s for the pipeline)")


if __name__ == "__main__":
    main()
