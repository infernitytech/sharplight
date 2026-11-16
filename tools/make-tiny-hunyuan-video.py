#!/usr/bin/env python3
"""Creates a tiny random HunyuanVideo pipeline and diffusers reference outputs for the video-generation tests.

Writes OUT (default ~/models/video/tiny-random-hunyuan-video):

  diffusers/           diffusers folder: HunyuanVideoTransformer3DModel (1 dual-stream and 1 single-stream block, token refiner,
                       guidance embedding), a 4-layer LlamaModel text encoder with the real Llama-3 tokenizer (prompt template and
                       crop as the pipeline), a 2-layer CLIPTextModel (pooled) with the real CLIP tokenizer, AutoencoderKLHunyuanVideo
                       (causal 3-D VAE, 8× spatial and 4× temporal compression), FlowMatchEulerDiscreteScheduler (shift 7)
  reference/*.f32      raw little-endian f32 reference tensors, described by reference/index.json

The tokenizers come from --tokenizers (a HunyuanVideo diffusers folder with tokenizer/ and tokenizer_2/), so the token ids are real.
The clip has 25 frames (7 latent frames), so diffusers decodes it in overlapping temporal tiles as it does for real clips.
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch

PROMPT = "a red panda eating bamboo in a forest, cinematic"
HEIGHT, WIDTH, FRAMES, STEPS, GUIDANCE = 32, 48, 25, 4, 6.0


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


def perturb(model, scale=0.05):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * scale)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/video/tiny-random-hunyuan-video"))
    parser.add_argument("--tokenizers", default=os.path.expanduser("~/models/video/HunyuanVideo"))
    args = parser.parse_args()

    from diffusers import AutoencoderKLHunyuanVideo, FlowMatchEulerDiscreteScheduler, HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
    from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, LlamaConfig, LlamaModel

    torch.set_grad_enabled(False)
    torch.manual_seed(3)
    transformer = perturb(HunyuanVideoTransformer3DModel(
        in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=16, num_layers=1, num_single_layers=1, num_refiner_layers=1,
        mlp_ratio=2.0, patch_size=2, patch_size_t=1, qk_norm="rms_norm", guidance_embeds=True, text_embed_dim=32, pooled_projection_dim=16,
        rope_theta=256.0, rope_axes_dim=(4, 6, 6)))
    vae = perturb(AutoencoderKLHunyuanVideo(
        in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 16, 16, 16), layers_per_block=1, norm_num_groups=4,
        scaling_factor=0.476986, spatial_compression_ratio=8, temporal_compression_ratio=4, mid_block_add_attention=True), 0.02)
    text = perturb(LlamaModel(LlamaConfig(
        vocab_size=128320, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, rms_norm_eps=1e-5, rope_theta=500000.0, max_position_embeddings=8192, bos_token_id=128000, eos_token_id=128001)))
    clip = perturb(CLIPTextModel(CLIPTextConfig(
        vocab_size=49408, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=77,
        hidden_act="quick_gelu", projection_dim=16, bos_token_id=0, eos_token_id=2, pad_token_id=1)))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.tokenizers, "tokenizer"))
    tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(args.tokenizers, "tokenizer_2"))
    scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
    pipe = HunyuanVideoPipeline(text_encoder=text, tokenizer=tokenizer, transformer=transformer, vae=vae, scheduler=scheduler,
                                text_encoder_2=clip, tokenizer_2=tokenizer_2)
    shutil.rmtree(os.path.join(args.out, "diffusers"), ignore_errors=True)
    pipe.save_pretrained(os.path.join(args.out, "diffusers"), safe_serialization=True)
    ref = Reference(os.path.join(args.out, "reference"))
    extra = {"height": HEIGHT, "width": WIDTH, "frames": FRAMES, "steps": STEPS, "guidance_scale": GUIDANCE, "prompt": PROMPT}

    # Text: the templated Llama states after the crop with their mask, and CLIP's pooled output.
    embeds, pooled, mask = pipe.encode_prompt(PROMPT, device="cpu", max_sequence_length=256)
    ref.add("text_hidden", embeds[0])
    ref.add("text_mask", mask[0])
    ref.add("pooled", pooled[0])
    extra["text_tokens"] = int(mask[0].sum())

    # Transformer on random inputs with the real conditioning.
    g = torch.Generator().manual_seed(11)
    latent_frames = (FRAMES - 1) // 4 + 1
    lh, lw = HEIGHT // 8, WIDTH // 8
    latent = torch.randn(1, 4, latent_frames, lh, lw, generator=g)
    timestep = torch.tensor([700.0])
    guidance = torch.tensor([GUIDANCE * 1000.0])
    ref.add("transformer_latent", latent[0])
    ref.add("transformer_out", transformer(latent, timestep, embeds, mask, pooled, guidance, return_dict=False)[0][0])

    # VAE: decode scaled latents as the pipeline does (temporal tiles for more than 4 latent frames).
    vae_latent = torch.randn(1, 4, latent_frames, lh, lw, generator=g)
    ref.add("vae_latent", vae_latent[0])
    ref.add("vae_decoded", vae.decode(vae_latent / vae.config.scaling_factor, return_dict=False)[0][0].clamp(-1, 1) * 0.5 + 0.5)

    noise = torch.randn(1, 4, latent_frames, lh, lw, generator=g)
    ref.add("noise", noise[0])
    video = pipe(prompt=PROMPT, height=HEIGHT, width=WIDTH, num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                 latents=noise.clone(), output_type="np").frames[0]
    ref.add("video", torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2))
    ref.save(extra)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
