#!/usr/bin/env python3
"""Diffusers references of HunyuanVideo text-to-video running a transformer GGUF file, for the video-generation tests.

The transformer comes from a GGUF file (city96, dequantized on the fly by diffusers' GGUF support, computed in f32 on CUDA), configured
from the HunyuanVideo diffusers folder, which also gives the Llama text encoder (run in f32 on the CPU: 8B parameters do not fit the
GPU in f32), CLIP-L and the VAE (f32 on the CPU, decoded in diffusers' temporal tiles). TF32 is off. Writes to OUT/reference:
text_hidden [256, 4096], text_mask, pooled, noise, the first transformer evaluation (transformer_latent, transformer_timestep,
transformer_out), the final latents, video [F, 3, H, W], index.json.

Usage: make-hunyuan-video-references.py TRANSFORMER.gguf FOLDER OUT [--frames 33] [--height 480] [--width 832] [--steps 20]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

PROMPT = "A cat walks on the grass, realistic style, warm sunlight"


def save(folder, index, name, tensor):
    array = tensor.detach().to(torch.float32).cpu().numpy()
    array.astype("<f4").tofile(os.path.join(folder, name + ".f32"))
    index[name] = list(array.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transformer")
    parser.add_argument("folder")
    parser.add_argument("out")
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from diffusers import AutoencoderKLHunyuanVideo, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
    from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, LlamaModel

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = os.path.join(args.out, "reference")
    os.makedirs(folder, exist_ok=True)
    index = {}
    transformer = HunyuanVideoTransformer3DModel.from_single_file(args.transformer, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float32),
                                                                  config=os.path.join(args.folder, "transformer"), torch_dtype=torch.float32)
    text = LlamaModel.from_pretrained(os.path.join(args.folder, "text_encoder"), torch_dtype=torch.float32)
    clip = CLIPTextModel.from_pretrained(os.path.join(args.folder, "text_encoder_2"), torch_dtype=torch.float32)
    vae = AutoencoderKLHunyuanVideo.from_pretrained(os.path.join(args.folder, "vae"), torch_dtype=torch.float32)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(args.folder, "scheduler"))
    pipe = HunyuanVideoPipeline(text_encoder=text, tokenizer=AutoTokenizer.from_pretrained(os.path.join(args.folder, "tokenizer")), transformer=transformer,
                                vae=vae, scheduler=scheduler, text_encoder_2=clip, tokenizer_2=CLIPTokenizer.from_pretrained(os.path.join(args.folder, "tokenizer_2")))
    # The text encoders on the CPU in f32.
    embeds, pooled, mask = pipe.encode_prompt(PROMPT, device="cpu", max_sequence_length=256)
    save(folder, index, "text_hidden", embeds[0])
    save(folder, index, "text_mask", mask[0])
    save(folder, index, "pooled", pooled[0])
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    del text, clip
    transformer.to("cuda")
    vae.to("cuda")

    latent_frames = (args.frames - 1) // 4 + 1
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(1, 16, latent_frames, args.height // 8, args.width // 8, generator=generator)
    save(folder, index, "noise", noise[0])
    captured = {}

    def hook(module, inputs, kwargs, output):
        if captured:
            return
        captured["done"] = True
        save(folder, index, "transformer_latent", kwargs["hidden_states"][0])
        save(folder, index, "transformer_timestep", kwargs["timestep"].reshape(-1)[:1].float())
        save(folder, index, "transformer_out", output[0][0])

    transformer.register_forward_hook(hook, with_kwargs=True)
    start = time.time()
    latents = pipe(prompt_embeds=embeds.cuda(), pooled_prompt_embeds=pooled.cuda(), prompt_attention_mask=mask.cuda(), height=args.height, width=args.width,
                   num_frames=args.frames, num_inference_steps=args.steps, guidance_scale=args.guidance, latents=noise.clone().cuda(),
                   output_type="latent").frames
    elapsed = time.time() - start
    save(folder, index, "latents", latents[0])
    # The VAE decodes on the CPU in f32 (diffusers' temporal tiles, no spatial tiling): a 5-latent-frame tile at 832×480 does not fit
    # the 32 GB GPU in f32.
    transformer.to("cpu")
    vae.to("cpu")
    torch.cuda.empty_cache()
    decoded = vae.decode(latents.cpu().to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
    video = pipe.video_processor.postprocess_video(decoded, output_type="np")[0]
    save(folder, index, "video", torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2))
    with open(os.path.join(folder, "index.json"), "w") as f:
        json.dump({"tensors": index, "prompt": PROMPT, "frames": args.frames, "height": args.height, "width": args.width, "steps": args.steps,
                   "guidance_scale": args.guidance, "seed": args.seed, "seconds": elapsed}, f, indent=1)
    print(f"wrote {args.out} ({elapsed:.1f} s for the pipeline)")


if __name__ == "__main__":
    main()
