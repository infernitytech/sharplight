#!/usr/bin/env python3
"""Creates a tiny random LTX-Video-shaped pipeline and diffusers reference outputs for the video-generation tests.

The layout of LTX-Video 0.9.0 with very small widths: LTXVideoTransformer3DModel (1×1×1 patches, RMS-normed blocks with
six-fold adaptive modulation, fractional (frame, row, column) RoPE, cross-attention to T5 tokens with a key mask), a T5 encoder
(run without a mask, as the LTX pipeline does), the non-causal 3-D VAE decoder (AutoencoderKLLTXVideo: 4×4 pixel patching,
depth-to-space 2×2×2 upsamplers, per-channel latent statistics) and the flow-matching Euler scheduler with the resolution-dependent
shift and the terminal stretch.

Written to OUT (default ~/models/video/tiny-random-ltx):

  diffusers/           diffusers folder
  reference/*.f32      raw little-endian f32 reference tensors described by reference/index.json

The T5 tokenizer comes from --tokenizer (a T5 tokenizer folder, e.g. a FLUX.1 diffusers folder's tokenizer_2/) or the hub.
Requires torch, diffusers, transformers and safetensors.
"""

import argparse
import json
import os

import numpy as np
import torch

PROMPTS = ["a cat walking on grass", "A painting of a FOX, in the style of Van Gogh!! 123", ""]
PROMPT = "a red panda eating bamboo in a forest, cinematic"
NEGATIVE = "blurry, low quality"
HEIGHT = 64
WIDTH = 96
FRAMES = 17
STEPS = 4
GUIDANCE = 3.0
MAX_SEQUENCE = 128
FRAME_RATE = 25


class Reference:
    def __init__(self, folder):
        self.folder = folder
        self.index = {}
        os.makedirs(folder, exist_ok=True)

    def add(self, name, tensor):
        array = tensor.detach().to(torch.float32).cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor, np.float32)
        array.astype("<f4").tofile(os.path.join(self.folder, name + ".f32"))
        self.index[name] = list(array.shape)

    def save(self, extra):
        with open(os.path.join(self.folder, "index.json"), "w") as f:
            json.dump({"tensors": self.index, **extra}, f, indent=1)


def build_models():
    from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
    from transformers import T5Config, T5EncoderModel

    torch.manual_seed(4)
    transformer = LTXVideoTransformer3DModel(
        in_channels=8, out_channels=8, patch_size=1, patch_size_t=1, num_attention_heads=2, attention_head_dim=12, cross_attention_dim=24,
        num_layers=2, caption_channels=32)
    generator = torch.Generator().manual_seed(6)
    vae = AutoencoderKLLTXVideo(
        in_channels=3, out_channels=3, latent_channels=8, block_out_channels=(8, 16, 32, 32), layers_per_block=(1, 1, 1, 1, 1),
        decoder_block_out_channels=(8, 16, 32, 32), decoder_layers_per_block=(1, 1, 1, 1, 1), decoder_spatio_temporal_scaling=(True, True, True, False),
        spatio_temporal_scaling=(True, True, True, False), patch_size=4, patch_size_t=1, decoder_causal=False, encoder_causal=True,
        resnet_norm_eps=1e-6, scaling_factor=1.0)
    with torch.no_grad():
        vae.latents_mean.copy_(torch.randn(8, generator=generator) * 0.5)
        vae.latents_std.copy_(torch.rand(8, generator=generator) + 0.5)
    text = T5EncoderModel(T5Config(
        vocab_size=32128, d_model=32, d_kv=4, d_ff=64, num_layers=2, num_heads=8, feed_forward_proj="gated-gelu",
        relative_attention_num_buckets=32, relative_attention_max_distance=128, dense_act_fn="gelu_new", is_gated_act=True))
    with torch.no_grad():
        for model in (transformer, vae, text):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
    return transformer.eval(), vae.eval(), text.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/video/tiny-random-ltx"))
    parser.add_argument("--tokenizer", default=os.path.expanduser("~/models/flux/FLUX.1-schnell/tokenizer_2"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    from diffusers import FlowMatchEulerDiscreteScheduler, LTXPipeline
    from transformers import AutoTokenizer

    transformer, vae, text = build_models()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if os.path.isdir(args.tokenizer) else "google/t5-v1_1-xxl")
    scheduler = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=1024, base_shift=0.95, max_image_seq_len=4096, max_shift=2.05, num_train_timesteps=1000, shift=1.0,
        shift_terminal=0.1, use_dynamic_shifting=True)
    pipe = LTXPipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=transformer)
    pipe.save_pretrained(os.path.join(args.out, "diffusers"), safe_serialization=True)
    ref = Reference(os.path.join(args.out, "reference"))
    extra = {"height": HEIGHT, "width": WIDTH, "frames": FRAMES, "steps": STEPS, "guidance_scale": GUIDANCE, "prompt": PROMPT,
             "negative_prompt": NEGATIVE, "max_sequence_length": MAX_SEQUENCE, "frame_rate": FRAME_RATE}

    tokens = {}
    for i, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, padding="max_length", max_length=MAX_SEQUENCE, truncation=True, add_special_tokens=True)
        tokens[prompt] = [int(t) for t, m in zip(ids["input_ids"], ids["attention_mask"]) if m]
        embeds, _ = pipe._get_t5_prompt_embeds(prompt, max_sequence_length=MAX_SEQUENCE, device="cpu", dtype=torch.float32)
        ref.add(f"text_hidden_{i}", embeds[0])
    extra["tokens"] = tokens

    z = transformer.config.in_channels
    latent_frames = (FRAMES - 1) // 8 + 1
    lh, lw = HEIGHT // 32, WIDTH // 32
    g = torch.Generator().manual_seed(12)
    latent = torch.randn(1, z, latent_frames, lh, lw, generator=g)
    context = torch.randn(1, MAX_SEQUENCE, 32, generator=g)
    mask = torch.zeros(1, MAX_SEQUENCE)
    mask[:, :9] = 1
    packed = pipe._pack_latents(latent)
    out = transformer(packed, context, torch.tensor([700.0]), mask, num_frames=latent_frames, height=lh, width=lw,
                      rope_interpolation_scale=(8 / FRAME_RATE, 32, 32), return_dict=False)[0]
    ref.add("transformer_latent", latent[0])
    ref.add("transformer_timestep", torch.tensor([700.0]))
    ref.add("transformer_context", context[0])
    ref.add("transformer_context_mask", mask[0])
    ref.add("transformer_out", pipe._unpack_latents(out, latent_frames, lh, lw)[0])

    vae_latent = torch.randn(1, z, latent_frames, lh, lw, generator=g)
    ref.add("vae_latent", vae_latent[0])
    denormalized = pipe._denormalize_latents(vae_latent, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)
    ref.add("vae_decoded", vae.decode(denormalized, return_dict=False)[0][0])

    noise = torch.randn(1, z, latent_frames, lh, lw, generator=g)
    ref.add("noise", noise[0])
    video = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, height=HEIGHT, width=WIDTH, num_frames=FRAMES, frame_rate=FRAME_RATE,
                 num_inference_steps=STEPS, guidance_scale=GUIDANCE, latents=pipe._pack_latents(noise.clone()), output_type="np",
                 max_sequence_length=MAX_SEQUENCE).frames[0]
    ref.add("video", torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2))
    ref.save(extra)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
