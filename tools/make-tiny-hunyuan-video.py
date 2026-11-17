#!/usr/bin/env python3
"""Creates a tiny random HunyuanVideo pipeline and diffusers reference outputs for the video-generation tests.

Writes OUT (default ~/models/video/tiny-random-hunyuan-video):

  diffusers/           diffusers folder: HunyuanVideoTransformer3DModel (1 dual-stream and 1 single-stream block, token refiner,
                       guidance embedding), a 4-layer LlamaModel text encoder with the real Llama-3 tokenizer (prompt template and
                       crop as the pipeline), a 2-layer CLIPTextModel (pooled) with the real CLIP tokenizer, AutoencoderKLHunyuanVideo
                       (causal 3-D VAE, 8× spatial and 4× temporal compression), FlowMatchEulerDiscreteScheduler (shift 7)
  single/hunyuan-video.safetensors   the transformer under the original HunyuanVideo names (fused q/k/v, the single blocks' fused
                       linear1, the final modulation as (shift, scale)), as GGUF and ComfyUI files store it
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


PLAIN = [
    ("time_text_embed.timestep_embedder.linear_1.", "time_in.mlp.0."), ("time_text_embed.timestep_embedder.linear_2.", "time_in.mlp.2."),
    ("time_text_embed.guidance_embedder.linear_1.", "guidance_in.mlp.0."), ("time_text_embed.guidance_embedder.linear_2.", "guidance_in.mlp.2."),
    ("time_text_embed.text_embedder.linear_1.", "vector_in.in_layer."), ("time_text_embed.text_embedder.linear_2.", "vector_in.out_layer."),
    ("context_embedder.time_text_embed.timestep_embedder.linear_1.", "txt_in.t_embedder.mlp.0."),
    ("context_embedder.time_text_embed.timestep_embedder.linear_2.", "txt_in.t_embedder.mlp.2."),
    ("context_embedder.time_text_embed.text_embedder.linear_1.", "txt_in.c_embedder.linear_1."),
    ("context_embedder.time_text_embed.text_embedder.linear_2.", "txt_in.c_embedder.linear_2."),
    ("context_embedder.proj_in.", "txt_in.input_embedder."), ("x_embedder.proj.", "img_in.proj."), ("proj_out.", "final_layer.linear."),
]
REFINER = {"norm1": "norm1", "norm2": "norm2", "attn.to_out.0": "self_attn_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2",
           "norm_out.linear": "adaLN_modulation.1"}
DOUBLE = {"norm1.linear": "img_mod.linear", "norm1_context.linear": "txt_mod.linear", "attn.norm_q": "img_attn_q_norm", "attn.norm_k": "img_attn_k_norm",
          "attn.norm_added_q": "txt_attn_q_norm", "attn.norm_added_k": "txt_attn_k_norm", "attn.to_out.0": "img_attn_proj", "attn.to_add_out": "txt_attn_proj",
          "ff.net.0.proj": "img_mlp.fc1", "ff.net.2": "img_mlp.fc2", "ff_context.net.0.proj": "txt_mlp.fc1", "ff_context.net.2": "txt_mlp.fc2"}
SINGLE = {"norm.linear": "modulation.linear", "attn.norm_q": "q_norm", "attn.norm_k": "k_norm", "proj_out": "linear2"}


def to_original(state):
    """diffusers HunyuanVideoTransformer3DModel names → the original names (fused projections, (shift, scale) final modulation)."""
    out = {}
    for kind in ("weight", "bias"):
        half = state[f"norm_out.linear.{kind}"].shape[0] // 2
        out[f"final_layer.adaLN_modulation.1.{kind}"] = torch.cat([state[f"norm_out.linear.{kind}"][half:], state[f"norm_out.linear.{kind}"][:half]])
    for name, value in state.items():
        if name.startswith("norm_out.linear."):
            continue
        for diffusers, original in PLAIN:
            if name.startswith(diffusers) and name.count(".") == diffusers.count("."):
                out[original + name[len(diffusers):]] = value
                break
    layers = {}
    for name in state:
        parts = name.split(".")
        if name.startswith("context_embedder.token_refiner.refiner_blocks."):
            layers[("refiner", int(parts[3]))] = True
        elif name.startswith("transformer_blocks."):
            layers[("double", int(parts[1]))] = True
        elif name.startswith("single_transformer_blocks."):
            layers[("single", int(parts[1]))] = True
    for (kind, i) in layers:
        if kind == "refiner":
            src, dst, table = f"context_embedder.token_refiner.refiner_blocks.{i}.", f"txt_in.individual_token_refiner.blocks.{i}.", REFINER
            fused = {"self_attn_qkv": ["attn.to_q", "attn.to_k", "attn.to_v"]}
        elif kind == "double":
            src, dst, table = f"transformer_blocks.{i}.", f"double_blocks.{i}.", DOUBLE
            fused = {"img_attn_qkv": ["attn.to_q", "attn.to_k", "attn.to_v"], "txt_attn_qkv": ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"]}
        else:
            src, dst, table = f"single_transformer_blocks.{i}.", f"single_blocks.{i}.", SINGLE
            fused = {"linear1": ["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"]}
        for module, original in table.items():
            for kind_ in ("weight", "bias"):
                if src + module + "." + kind_ in state:
                    out[dst + original + "." + kind_] = state[src + module + "." + kind_]
        for original, modules in fused.items():
            for kind_ in ("weight", "bias"):
                out[dst + original + "." + kind_] = torch.cat([state[src + m + "." + kind_] for m in modules])
    return {k: v.contiguous() for k, v in out.items()}


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
    from safetensors.torch import save_file
    os.makedirs(os.path.join(args.out, "single"), exist_ok=True)
    save_file(to_original(transformer.state_dict()), os.path.join(args.out, "single", "hunyuan-video.safetensors"))
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
