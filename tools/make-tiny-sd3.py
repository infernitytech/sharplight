#!/usr/bin/env python3
"""Creates tiny random SD3-shaped pipelines and diffusers reference outputs for the image-generation tests.

Two variants share the layout of Stable Diffusion 3 with very small widths (an MMDiT of joint-attention blocks whose last block only
updates the image stream, 2D sin-cos position embeddings cropped from a fixed grid, CLIP-L and OpenCLIP-bigG-like text encoders with
projections whose penultimate hidden states are joined per token and whose pooled embeddings are concatenated, a T5 encoder whose
hidden states follow the CLIP ones along the sequence, a 16-channel VAE without quant convolutions, the flow-matching Euler
scheduler with shift 3):

  sd3   no q/k norms, no second attention                          (like SD3 Medium)
  sd35  RMS q/k norms, a second image-only attention in blocks 0-1 (like SD3.5 Medium, MMDiT-X)

Written to OUT/{variant} (default OUT ~/models/sd3/tiny-random-sd3):

  diffusers/                   diffusers folder (model_index.json, transformer/, vae/, text_encoder(_2,_3)/, tokenizer(_2,_3)/, scheduler/)
  single/sd3.safetensors       the transformer under the original MMDiT names (model.diffusion_model.*)
  single/sd3-Q8_0.gguf         the transformer as a transformer-only GGUF (city96 layout: matrices Q8_0, the rest F16 / F32)
  reference/*.f32              raw little-endian f32 reference tensors, described by reference/index.json

The Q8_0 references (transformer_q8_out, q8_image) come from the diffusers transformer with the dequantized GGUF weights.
Requires torch, diffusers, transformers, safetensors and gguf. The tokenizers come from --tokenizers (a FLUX.1 diffusers folder
with tokenizer/ and tokenizer_2/ (T5)) or the Hugging Face hub.
"""

import argparse
import copy
import json
import os

import numpy as np
import torch

PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123", ""]
PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"
HEIGHT = 64
WIDTH = 96
STEPS = 4
GUIDANCE = 5.0
T5_LENGTH = 256


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


def build_models(variant):
    from diffusers import AutoencoderKL, SD3Transformer2DModel
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection, T5Config, T5EncoderModel

    sd35 = variant == "sd35"
    torch.manual_seed(3 if sd35 else 2)
    transformer = SD3Transformer2DModel(
        sample_size=32, patch_size=2, in_channels=16, num_layers=3, attention_head_dim=32, num_attention_heads=2, joint_attention_dim=192,
        caption_projection_dim=64, pooled_projection_dim=80, out_channels=16, pos_embed_max_size=24,
        dual_attention_layers=(0, 1) if sd35 else (), qk_norm="rms_norm" if sd35 else None)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=16, layers_per_block=2, norm_num_groups=32, block_out_channels=(32, 32, 64, 64),
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4, scaling_factor=1.5305, shift_factor=0.0609,
        use_quant_conv=False, use_post_quant_conv=False)
    text = CLIPTextModelWithProjection(CLIPTextConfig(
        vocab_size=49408, hidden_size=64, intermediate_size=128, num_attention_heads=1, num_hidden_layers=2, max_position_embeddings=77,
        hidden_act="quick_gelu", projection_dim=32, bos_token_id=49406, eos_token_id=49407, pad_token_id=1))
    text_2 = CLIPTextModelWithProjection(CLIPTextConfig(
        vocab_size=49408, hidden_size=128, intermediate_size=256, num_attention_heads=2, num_hidden_layers=3, max_position_embeddings=77,
        hidden_act="gelu", projection_dim=48, bos_token_id=49406, eos_token_id=49407, pad_token_id=1))
    text_3 = T5EncoderModel(T5Config(
        vocab_size=32128, d_model=192, d_kv=4, d_ff=64, num_layers=2, num_heads=8, feed_forward_proj="gated-gelu",
        relative_attention_num_buckets=32, relative_attention_max_distance=128, dense_act_fn="gelu_new", is_gated_act=True))
    with torch.no_grad():
        for model in (transformer, vae, text, text_2, text_3):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
    return transformer.eval(), vae.eval(), text.eval(), text_2.eval(), text_3.eval()


def to_original(state, layers):
    """diffusers SD3Transformer2DModel weights → original MMDiT names: fused q/k/v, (shift, scale) final / last context modulation."""
    out = {}

    def put(name, value):
        out[name] = value.contiguous()

    def swap(value):
        scale, shift = value.chunk(2, dim=0)
        return torch.cat([shift, scale], dim=0)

    put("pos_embed", state["pos_embed.pos_embed"])
    for suffix in ("weight", "bias"):
        put(f"x_embedder.proj.{suffix}", state[f"pos_embed.proj.{suffix}"])
        put(f"t_embedder.mlp.0.{suffix}", state[f"time_text_embed.timestep_embedder.linear_1.{suffix}"])
        put(f"t_embedder.mlp.2.{suffix}", state[f"time_text_embed.timestep_embedder.linear_2.{suffix}"])
        put(f"y_embedder.mlp.0.{suffix}", state[f"time_text_embed.text_embedder.linear_1.{suffix}"])
        put(f"y_embedder.mlp.2.{suffix}", state[f"time_text_embed.text_embedder.linear_2.{suffix}"])
        put(f"context_embedder.{suffix}", state[f"context_embedder.{suffix}"])
        put(f"final_layer.adaLN_modulation.1.{suffix}", swap(state[f"norm_out.linear.{suffix}"]))
        put(f"final_layer.linear.{suffix}", state[f"proj_out.{suffix}"])
        for i in range(layers):
            s = f"transformer_blocks.{i}."
            j = f"joint_blocks.{i}."
            last = i == layers - 1
            put(j + f"x_block.adaLN_modulation.1.{suffix}", state[s + f"norm1.linear.{suffix}"])
            context_mod = state[s + f"norm1_context.linear.{suffix}"]
            put(j + f"context_block.adaLN_modulation.1.{suffix}", swap(context_mod) if last else context_mod)
            put(j + f"x_block.attn.qkv.{suffix}", torch.cat([state[s + f"attn.to_{x}.{suffix}"] for x in "qkv"], dim=0))
            put(j + f"context_block.attn.qkv.{suffix}", torch.cat([state[s + f"attn.add_{x}_proj.{suffix}"] for x in "qkv"], dim=0))
            put(j + f"x_block.attn.proj.{suffix}", state[s + f"attn.to_out.0.{suffix}"])
            put(j + f"x_block.mlp.fc1.{suffix}", state[s + f"ff.net.0.proj.{suffix}"])
            put(j + f"x_block.mlp.fc2.{suffix}", state[s + f"ff.net.2.{suffix}"])
            if not last:
                put(j + f"context_block.attn.proj.{suffix}", state[s + f"attn.to_add_out.{suffix}"])
                put(j + f"context_block.mlp.fc1.{suffix}", state[s + f"ff_context.net.0.proj.{suffix}"])
                put(j + f"context_block.mlp.fc2.{suffix}", state[s + f"ff_context.net.2.{suffix}"])
            if s + f"attn2.to_q.{suffix}" in state:
                put(j + f"x_block.attn2.qkv.{suffix}", torch.cat([state[s + f"attn2.to_{x}.{suffix}"] for x in "qkv"], dim=0))
                put(j + f"x_block.attn2.proj.{suffix}", state[s + f"attn2.to_out.0.{suffix}"])
    for i in range(layers):
        s = f"transformer_blocks.{i}."
        j = f"joint_blocks.{i}."
        for diffusers_name, original in (("attn.norm_q", "x_block.attn.ln_q"), ("attn.norm_k", "x_block.attn.ln_k"),
                                         ("attn.norm_added_q", "context_block.attn.ln_q"), ("attn.norm_added_k", "context_block.attn.ln_k"),
                                         ("attn2.norm_q", "x_block.attn2.ln_q"), ("attn2.norm_k", "x_block.attn2.ln_k")):
            if s + diffusers_name + ".weight" in state:
                put(j + original + ".weight", state[s + diffusers_name + ".weight"])
    assert sum(v.numel() for v in out.values()) == sum(v.numel() for v in state.values())
    return out


def from_original(original, template, layers):
    """Original names → diffusers names (the inverse of to_original), for loading dequantized weights into the reference model."""
    state = {}
    for suffix in ("weight", "bias"):
        state[f"pos_embed.proj.{suffix}"] = original[f"x_embedder.proj.{suffix}"]
        state[f"time_text_embed.timestep_embedder.linear_1.{suffix}"] = original[f"t_embedder.mlp.0.{suffix}"]
        state[f"time_text_embed.timestep_embedder.linear_2.{suffix}"] = original[f"t_embedder.mlp.2.{suffix}"]
        state[f"time_text_embed.text_embedder.linear_1.{suffix}"] = original[f"y_embedder.mlp.0.{suffix}"]
        state[f"time_text_embed.text_embedder.linear_2.{suffix}"] = original[f"y_embedder.mlp.2.{suffix}"]
        state[f"context_embedder.{suffix}"] = original[f"context_embedder.{suffix}"]
        shift, scale = original[f"final_layer.adaLN_modulation.1.{suffix}"].chunk(2, dim=0)
        state[f"norm_out.linear.{suffix}"] = torch.cat([scale, shift], dim=0)
        state[f"proj_out.{suffix}"] = original[f"final_layer.linear.{suffix}"]
        for i in range(layers):
            s = f"transformer_blocks.{i}."
            j = f"joint_blocks.{i}."
            last = i == layers - 1
            state[s + f"norm1.linear.{suffix}"] = original[j + f"x_block.adaLN_modulation.1.{suffix}"]
            context_mod = original[j + f"context_block.adaLN_modulation.1.{suffix}"]
            if last:
                shift, scale = context_mod.chunk(2, dim=0)
                context_mod = torch.cat([scale, shift], dim=0)
            state[s + f"norm1_context.linear.{suffix}"] = context_mod
            for k, x in enumerate("qkv"):
                state[s + f"attn.to_{x}.{suffix}"] = original[j + f"x_block.attn.qkv.{suffix}"].chunk(3, dim=0)[k]
                state[s + f"attn.add_{x}_proj.{suffix}"] = original[j + f"context_block.attn.qkv.{suffix}"].chunk(3, dim=0)[k]
                if j + f"x_block.attn2.qkv.{suffix}" in original:
                    state[s + f"attn2.to_{x}.{suffix}"] = original[j + f"x_block.attn2.qkv.{suffix}"].chunk(3, dim=0)[k]
            state[s + f"attn.to_out.0.{suffix}"] = original[j + f"x_block.attn.proj.{suffix}"]
            state[s + f"ff.net.0.proj.{suffix}"] = original[j + f"x_block.mlp.fc1.{suffix}"]
            state[s + f"ff.net.2.{suffix}"] = original[j + f"x_block.mlp.fc2.{suffix}"]
            if not last:
                state[s + f"attn.to_add_out.{suffix}"] = original[j + f"context_block.attn.proj.{suffix}"]
                state[s + f"ff_context.net.0.proj.{suffix}"] = original[j + f"context_block.mlp.fc1.{suffix}"]
                state[s + f"ff_context.net.2.{suffix}"] = original[j + f"context_block.mlp.fc2.{suffix}"]
            if j + f"x_block.attn2.proj.{suffix}" in original:
                state[s + f"attn2.to_out.0.{suffix}"] = original[j + f"x_block.attn2.proj.{suffix}"]
    state["pos_embed.pos_embed"] = original["pos_embed"]
    for i in range(layers):
        s = f"transformer_blocks.{i}."
        j = f"joint_blocks.{i}."
        for diffusers_name, name in (("attn.norm_q", "x_block.attn.ln_q"), ("attn.norm_k", "x_block.attn.ln_k"),
                                     ("attn.norm_added_q", "context_block.attn.ln_q"), ("attn.norm_added_k", "context_block.attn.ln_k"),
                                     ("attn2.norm_q", "x_block.attn2.ln_q"), ("attn2.norm_k", "x_block.attn2.ln_k")):
            if j + name + ".weight" in original:
                state[s + diffusers_name + ".weight"] = original[j + name + ".weight"]
    for name, value in template.items():
        assert state[name].shape == value.shape, (name, state[name].shape, value.shape)
    return state


def write_gguf(path, original):
    """Writes a city96-style transformer GGUF (2-D weights with rows of 32k values Q8_0, the rest F16 / F32) and returns the dequantized weights."""
    import gguf

    writer = gguf.GGUFWriter(path, "sd3")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    dequantized = {}
    for name, tensor in original.items():
        data = tensor.to(torch.float32).numpy()
        if data.ndim == 2 and data.shape[1] % 32 == 0 and data.shape[0] > 64:
            q = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(name, q, raw_shape=q.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            dequantized[name] = torch.from_numpy(gguf.quants.dequantize(q, gguf.GGMLQuantizationType.Q8_0).copy())
        elif name == "pos_embed":
            writer.add_tensor(name, data)
            dequantized[name] = torch.from_numpy(data.copy())
        else:
            half = data.astype(np.float16)
            writer.add_tensor(name, half)
            dequantized[name] = torch.from_numpy(half.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return dequantized


def make_variant(out, variant, tokenizers):
    from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
    from safetensors.torch import save_file
    from transformers import CLIPTokenizer, T5TokenizerFast

    transformer, vae, text, text_2, text_3 = build_models(variant)
    source = tokenizers or "black-forest-labs/FLUX.1-schnell"
    tokenizer = CLIPTokenizer.from_pretrained(source, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(source, subfolder="tokenizer", pad_token="!")
    tokenizer_3 = T5TokenizerFast.from_pretrained(source, subfolder="tokenizer_2")
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    pipe = StableDiffusion3Pipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, text_encoder_2=text_2,
                                    tokenizer_2=tokenizer_2, text_encoder_3=text_3, tokenizer_3=tokenizer_3, transformer=transformer)
    root = os.path.join(out, variant)
    pipe.save_pretrained(os.path.join(root, "diffusers"), safe_serialization=True)

    state = transformer.state_dict()
    layers = transformer.config.num_layers
    original = to_original(state, layers)
    os.makedirs(os.path.join(root, "single"), exist_ok=True)
    single = {"model.diffusion_model." + k: v.to(torch.float32) for k, v in original.items()}
    save_file(single, os.path.join(root, "single", "sd3.safetensors"))
    dequantized = write_gguf(os.path.join(root, "single", "sd3-Q8_0.gguf"), original)

    ref = Reference(os.path.join(root, "reference"))
    tokens = {}
    tokens_2 = {}
    tokens_3 = {}
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
            ids_2 = tokenizer_2(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
            ids_3 = tokenizer_3(prompt, padding="max_length", max_length=T5_LENGTH, truncation=True, add_special_tokens=True,
                                return_tensors="pt").input_ids
            tokens[prompt] = ids[0].tolist()
            tokens_2[prompt] = ids_2[0].tolist()
            tokens_3[prompt] = ids_3[0].tolist()
            o = text(ids, output_hidden_states=True)
            ref.add(f"text_pooled_{i}", o.text_embeds[0])
            ref.add(f"text_hidden_{i}", o.hidden_states[-2][0])
            o = text_2(ids_2, output_hidden_states=True)
            ref.add(f"text2_pooled_{i}", o.text_embeds[0])
            ref.add(f"text2_hidden_{i}", o.hidden_states[-2][0])
            ref.add(f"text3_hidden_{i}", text_3(ids_3)[0][0])

        g = torch.Generator().manual_seed(1)
        lh, lw = HEIGHT // 8, WIDTH // 8
        latent = torch.randn(1, 16, lh, lw, generator=g)
        context = torch.randn(1, 77 + T5_LENGTH, 192, generator=g)
        pooled = torch.randn(1, 80, generator=g)
        timestep = torch.tensor([731.7])
        kwargs = dict(encoder_hidden_states=context, pooled_projections=pooled, timestep=timestep, return_dict=False)
        ref.add("transformer_latent", latent)
        ref.add("transformer_context", context)
        ref.add("transformer_pooled", pooled)
        ref.add("transformer_timestep", timestep)
        ref.add("transformer_out", transformer(latent, **kwargs)[0])

        quantized = copy.deepcopy(transformer)
        quantized.load_state_dict(from_original(dequantized, state, layers))
        quantized.eval()
        ref.add("transformer_q8_out", quantized(latent, **kwargs)[0])

        vae_latent = torch.randn(1, 16, lh, lw, generator=g)
        ref.add("vae_latent", vae_latent)
        ref.add("vae_decoded", vae.decode(vae_latent).sample)
        image = torch.rand(1, 3, HEIGHT, WIDTH, generator=g) * 2 - 1
        ref.add("vae_image", image)
        ref.add("vae_encoded_mean", vae.encode(image).latent_dist.mean)
        noise = torch.randn(1, 16, lh, lw, generator=torch.Generator().manual_seed(42))
        ref.add("noise", noise)

        image = pipe(PROMPT, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS, guidance_scale=GUIDANCE, latents=noise.clone(),
                     max_sequence_length=T5_LENGTH, output_type="pt").images
        ref.add("image", image[0])
        timesteps = [float(t) for t in pipe.scheduler.timesteps]
        pipe.transformer = quantized
        image = pipe(PROMPT, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS, guidance_scale=GUIDANCE, latents=noise.clone(),
                     max_sequence_length=T5_LENGTH, output_type="pt").images
        ref.add("q8_image", image[0])

    ref.save({"tokens": tokens, "tokens_2": tokens_2, "tokens_3": tokens_3, "prompt": PROMPT, "steps": STEPS, "guidance_scale": GUIDANCE,
              "height": HEIGHT, "width": WIDTH, "t5_length": T5_LENGTH, "timesteps": timesteps})
    print("written", root)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/sd3/tiny-random-sd3"))
    parser.add_argument("--tokenizers", default=None, help="a FLUX.1 diffusers folder with tokenizer/ (CLIP) and tokenizer_2/ (T5)")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    for variant in ("sd3", "sd35"):
        make_variant(args.out, variant, args.tokenizers)


if __name__ == "__main__":
    main()
