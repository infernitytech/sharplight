#!/usr/bin/env python3
"""Creates tiny random FLUX.1-shaped pipelines and diffusers reference outputs for the image-generation tests.

Two variants share the layout of FLUX.1 with very small widths (a dual-stream / single-stream MMDiT with 3-axis RoPE, a CLIP text
encoder whose pooled output conditions the modulation, a T5 encoder whose hidden states are the context, a 16-channel VAE
without quant convolutions, the flow-matching Euler scheduler):

  dev      guidance embedding, dynamic (resolution-dependent) timestep shift, T5 context 512   (like FLUX.1-dev)
  schnell  no guidance embedding, shift 1, T5 context 256                                     (like FLUX.1-schnell)

Written to OUT/{variant} (default OUT ~/models/flux/tiny-random-flux):

  diffusers/                   diffusers folder (model_index.json, transformer/, vae/, text_encoder/, text_encoder_2/, tokenizer(_2)/, scheduler/)
  single/flux.safetensors      the transformer under the original (Black Forest Labs) tensor names
  single/flux-Q8_0.gguf        the same as a transformer-only GGUF (city96 layout: matrices Q8_0, the rest F16)
  reference/*.f32              raw little-endian f32 reference tensors, described by reference/index.json

The Q8_0 references (transformer_q8_out, q8_image) come from the diffusers transformer with the dequantized GGUF weights.
Requires torch, diffusers, transformers, safetensors and gguf. The tokenizers come from --tokenizers (a FLUX.1 diffusers folder
with tokenizer/ and tokenizer_2/) or the Hugging Face hub.
"""

import argparse
import json
import os

import numpy as np
import torch

PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123", ""]
PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"
HEIGHT = 64
WIDTH = 96
STEPS = 4


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


def build_models(guidance):
    from diffusers import AutoencoderKL, FluxTransformer2DModel
    from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel

    torch.manual_seed(0 if guidance else 1)
    transformer = FluxTransformer2DModel(
        patch_size=1, in_channels=64, num_layers=2, num_single_layers=2, attention_head_dim=32, num_attention_heads=2,
        joint_attention_dim=32, pooled_projection_dim=32, guidance_embeds=guidance, axes_dims_rope=(4, 14, 14))
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=16, layers_per_block=2, norm_num_groups=8, block_out_channels=(16, 16, 32, 32),
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4, scaling_factor=0.3611, shift_factor=0.1159,
        use_quant_conv=False, use_post_quant_conv=False)
    text = CLIPTextModel(CLIPTextConfig(
        vocab_size=49408, hidden_size=32, intermediate_size=64, num_attention_heads=4, num_hidden_layers=2, max_position_embeddings=77,
        hidden_act="quick_gelu", bos_token_id=0, eos_token_id=2, pad_token_id=1))
    text_2 = T5EncoderModel(T5Config(
        vocab_size=32128, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4, feed_forward_proj="gated-gelu",
        relative_attention_num_buckets=32, relative_attention_max_distance=128, dense_act_fn="gelu_new", is_gated_act=True))
    with torch.no_grad():
        for model in (transformer, vae, text, text_2):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
    return transformer.eval(), vae.eval(), text.eval(), text_2.eval()


def to_bfl(state, single_layers, double_layers):
    """diffusers FluxTransformer2DModel weights → original (BFL) names: fused q/k/v, (shift, scale) final modulation."""
    out = {}

    def put(name, value):
        out[name] = value.contiguous()

    for suffix in ("weight", "bias"):
        put(f"img_in.{suffix}", state[f"x_embedder.{suffix}"])
        put(f"txt_in.{suffix}", state[f"context_embedder.{suffix}"])
        put(f"time_in.in_layer.{suffix}", state[f"time_text_embed.timestep_embedder.linear_1.{suffix}"])
        put(f"time_in.out_layer.{suffix}", state[f"time_text_embed.timestep_embedder.linear_2.{suffix}"])
        put(f"vector_in.in_layer.{suffix}", state[f"time_text_embed.text_embedder.linear_1.{suffix}"])
        put(f"vector_in.out_layer.{suffix}", state[f"time_text_embed.text_embedder.linear_2.{suffix}"])
        if f"time_text_embed.guidance_embedder.linear_1.{suffix}" in state:
            put(f"guidance_in.in_layer.{suffix}", state[f"time_text_embed.guidance_embedder.linear_1.{suffix}"])
            put(f"guidance_in.out_layer.{suffix}", state[f"time_text_embed.guidance_embedder.linear_2.{suffix}"])
        scale, shift = state[f"norm_out.linear.{suffix}"].chunk(2, dim=0)
        put(f"final_layer.adaLN_modulation.1.{suffix}", torch.cat([shift, scale], dim=0))
        put(f"final_layer.linear.{suffix}", state[f"proj_out.{suffix}"])
        for i in range(double_layers):
            s = f"transformer_blocks.{i}."
            d = f"double_blocks.{i}."
            put(d + f"img_mod.lin.{suffix}", state[s + f"norm1.linear.{suffix}"])
            put(d + f"txt_mod.lin.{suffix}", state[s + f"norm1_context.linear.{suffix}"])
            put(d + f"img_attn.qkv.{suffix}", torch.cat([state[s + f"attn.to_{x}.{suffix}"] for x in "qkv"], dim=0))
            put(d + f"txt_attn.qkv.{suffix}", torch.cat([state[s + f"attn.add_{x}_proj.{suffix}"] for x in "qkv"], dim=0))
            put(d + f"img_attn.proj.{suffix}", state[s + f"attn.to_out.0.{suffix}"])
            put(d + f"txt_attn.proj.{suffix}", state[s + f"attn.to_add_out.{suffix}"])
            put(d + f"img_mlp.0.{suffix}", state[s + f"ff.net.0.proj.{suffix}"])
            put(d + f"img_mlp.2.{suffix}", state[s + f"ff.net.2.{suffix}"])
            put(d + f"txt_mlp.0.{suffix}", state[s + f"ff_context.net.0.proj.{suffix}"])
            put(d + f"txt_mlp.2.{suffix}", state[s + f"ff_context.net.2.{suffix}"])
        for i in range(single_layers):
            s = f"single_transformer_blocks.{i}."
            d = f"single_blocks.{i}."
            put(d + f"modulation.lin.{suffix}", state[s + f"norm.linear.{suffix}"])
            put(d + f"linear1.{suffix}", torch.cat([state[s + f"attn.to_{x}.{suffix}"] for x in "qkv"] + [state[s + f"proj_mlp.{suffix}"]], dim=0))
            put(d + f"linear2.{suffix}", state[s + f"proj_out.{suffix}"])
    for i in range(double_layers):
        s = f"transformer_blocks.{i}.attn."
        d = f"double_blocks.{i}."
        put(d + "img_attn.norm.query_norm.scale", state[s + "norm_q.weight"])
        put(d + "img_attn.norm.key_norm.scale", state[s + "norm_k.weight"])
        put(d + "txt_attn.norm.query_norm.scale", state[s + "norm_added_q.weight"])
        put(d + "txt_attn.norm.key_norm.scale", state[s + "norm_added_k.weight"])
    for i in range(single_layers):
        s = f"single_transformer_blocks.{i}.attn."
        put(f"single_blocks.{i}.norm.query_norm.scale", state[s + "norm_q.weight"])
        put(f"single_blocks.{i}.norm.key_norm.scale", state[s + "norm_k.weight"])
    return out


def write_gguf(path, bfl):
    """Writes a city96-style transformer GGUF (2-D weights with rows of 32k values Q8_0, the rest F16) and returns the dequantized weights."""
    import gguf

    writer = gguf.GGUFWriter(path, "flux")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    dequantized = {}
    for name, tensor in bfl.items():
        data = tensor.to(torch.float32).numpy()
        if data.ndim == 2 and data.shape[1] % 32 == 0 and data.shape[0] > 64:
            q = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(name, q, raw_shape=q.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            dequantized[name] = torch.from_numpy(gguf.quants.dequantize(q, gguf.GGMLQuantizationType.Q8_0).copy())
        else:
            half = data.astype(np.float16)
            writer.add_tensor(name, half)
            dequantized[name] = torch.from_numpy(half.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return dequantized


def from_bfl(bfl, template):
    """Original names → diffusers names (the inverse of to_bfl), for loading dequantized weights into the reference model."""
    state = {}
    single = sum(1 for k in template if k.startswith("single_transformer_blocks.") and k.endswith("norm.linear.weight"))
    double = sum(1 for k in template if k.startswith("transformer_blocks.") and k.endswith("norm1.linear.weight"))
    mapping = {}
    for suffix in ("weight", "bias"):
        mapping[f"x_embedder.{suffix}"] = f"img_in.{suffix}"
        mapping[f"context_embedder.{suffix}"] = f"txt_in.{suffix}"
        mapping[f"time_text_embed.timestep_embedder.linear_1.{suffix}"] = f"time_in.in_layer.{suffix}"
        mapping[f"time_text_embed.timestep_embedder.linear_2.{suffix}"] = f"time_in.out_layer.{suffix}"
        mapping[f"time_text_embed.text_embedder.linear_1.{suffix}"] = f"vector_in.in_layer.{suffix}"
        mapping[f"time_text_embed.text_embedder.linear_2.{suffix}"] = f"vector_in.out_layer.{suffix}"
        mapping[f"time_text_embed.guidance_embedder.linear_1.{suffix}"] = f"guidance_in.in_layer.{suffix}"
        mapping[f"time_text_embed.guidance_embedder.linear_2.{suffix}"] = f"guidance_in.out_layer.{suffix}"
        mapping[f"proj_out.{suffix}"] = f"final_layer.linear.{suffix}"
        for i in range(double):
            s = f"transformer_blocks.{i}."
            d = f"double_blocks.{i}."
            mapping[s + f"norm1.linear.{suffix}"] = d + f"img_mod.lin.{suffix}"
            mapping[s + f"norm1_context.linear.{suffix}"] = d + f"txt_mod.lin.{suffix}"
            mapping[s + f"attn.to_out.0.{suffix}"] = d + f"img_attn.proj.{suffix}"
            mapping[s + f"attn.to_add_out.{suffix}"] = d + f"txt_attn.proj.{suffix}"
            mapping[s + f"ff.net.0.proj.{suffix}"] = d + f"img_mlp.0.{suffix}"
            mapping[s + f"ff.net.2.{suffix}"] = d + f"img_mlp.2.{suffix}"
            mapping[s + f"ff_context.net.0.proj.{suffix}"] = d + f"txt_mlp.0.{suffix}"
            mapping[s + f"ff_context.net.2.{suffix}"] = d + f"txt_mlp.2.{suffix}"
            for j, x in enumerate("qkv"):
                mapping[s + f"attn.to_{x}.{suffix}"] = (d + f"img_attn.qkv.{suffix}", j, 3)
                mapping[s + f"attn.add_{x}_proj.{suffix}"] = (d + f"txt_attn.qkv.{suffix}", j, 3)
        for i in range(single):
            s = f"single_transformer_blocks.{i}."
            d = f"single_blocks.{i}."
            mapping[s + f"norm.linear.{suffix}"] = d + f"modulation.lin.{suffix}"
            mapping[s + f"proj_out.{suffix}"] = d + f"linear2.{suffix}"
            for x in ("q", "k", "v", "mlp"):
                mapping[s + (f"attn.to_{x}.{suffix}" if x != "mlp" else f"proj_mlp.{suffix}")] = (d + f"linear1.{suffix}", x)
    for i in range(double):
        s = f"transformer_blocks.{i}.attn."
        d = f"double_blocks.{i}."
        mapping[s + "norm_q.weight"] = d + "img_attn.norm.query_norm.scale"
        mapping[s + "norm_k.weight"] = d + "img_attn.norm.key_norm.scale"
        mapping[s + "norm_added_q.weight"] = d + "txt_attn.norm.query_norm.scale"
        mapping[s + "norm_added_k.weight"] = d + "txt_attn.norm.key_norm.scale"
    for i in range(single):
        s = f"single_transformer_blocks.{i}.attn."
        mapping[s + "norm_q.weight"] = f"single_blocks.{i}.norm.query_norm.scale"
        mapping[s + "norm_k.weight"] = f"single_blocks.{i}.norm.key_norm.scale"
    for name, shape_source in template.items():
        if name.startswith("norm_out.linear."):
            suffix = name.rsplit(".", 1)[1]
            shift, scale = bfl[f"final_layer.adaLN_modulation.1.{suffix}"].chunk(2, dim=0)
            state[name] = torch.cat([scale, shift], dim=0)
            continue
        target = mapping[name]
        if isinstance(target, str):
            state[name] = bfl[target]
        elif len(target) == 3:
            state[name] = bfl[target[0]].chunk(3, dim=0)[target[1]]
        else:
            fused = bfl[target[0]]
            inner = template[name.replace("proj_mlp", "attn.to_q").replace("attn.to_k", "attn.to_q").replace("attn.to_v", "attn.to_q")].shape[0]
            offsets = {"q": 0, "k": inner, "v": 2 * inner, "mlp": 3 * inner}
            start = offsets[target[1]]
            length = shape_source.shape[0]
            state[name] = fused[start:start + length]
        assert state[name].shape == shape_source.shape, (name, state[name].shape, shape_source.shape)
    return state


def pack(latents):
    from diffusers import FluxPipeline
    b, c, h, w = latents.shape
    return FluxPipeline._pack_latents(latents, b, c, h, w)


def make_variant(out, variant, tokenizers):
    from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
    from safetensors.torch import save_file
    from transformers import CLIPTokenizer, T5TokenizerFast

    dev = variant == "dev"
    transformer, vae, text, text_2 = build_models(dev)
    source = tokenizers or "black-forest-labs/FLUX.1-schnell"
    tokenizer = CLIPTokenizer.from_pretrained(source, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(source, subfolder="tokenizer_2")
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=3.0 if dev else 1.0, use_dynamic_shifting=dev, base_shift=0.5, max_shift=1.15,
        base_image_seq_len=256, max_image_seq_len=4096)
    pipe = FluxPipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, text_encoder_2=text_2,
                        tokenizer_2=tokenizer_2, transformer=transformer)
    root = os.path.join(out, variant)
    pipe.save_pretrained(os.path.join(root, "diffusers"), safe_serialization=True)
    context_length = 512 if dev else 256
    guidance = 3.5 if dev else 0.0

    state = transformer.state_dict()
    bfl = to_bfl(state, 2, 2)
    os.makedirs(os.path.join(root, "single"), exist_ok=True)
    save_file({k: v.to(torch.float32) for k, v in bfl.items()}, os.path.join(root, "single", "flux.safetensors"))
    dequantized = write_gguf(os.path.join(root, "single", "flux-Q8_0.gguf"), bfl)

    ref = Reference(os.path.join(root, "reference"))
    tokens = {}
    tokens_2 = {}
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
            ids_2 = tokenizer_2(prompt, padding="max_length", max_length=context_length, truncation=True, return_tensors="pt").input_ids
            tokens[prompt] = ids[0].tolist()
            tokens_2[prompt] = ids_2[0].tolist()
            ref.add(f"text_pooled_{i}", text(ids, output_hidden_states=False).pooler_output[0])
            ref.add(f"text2_hidden_{i}", text_2(ids_2, output_hidden_states=False)[0][0])

        g = torch.Generator().manual_seed(1)
        lh, lw = HEIGHT // 8, WIDTH // 8
        latent = torch.randn(1, 16, lh, lw, generator=g)
        context = torch.randn(1, context_length, 32, generator=g)
        pooled = torch.randn(1, 32, generator=g)
        timestep = torch.tensor([0.7317])
        img_ids = FluxPipeline._prepare_latent_image_ids(1, lh // 2, lw // 2, "cpu", torch.float32)
        txt_ids = torch.zeros(context_length, 3)
        kwargs = dict(encoder_hidden_states=context, pooled_projections=pooled, timestep=timestep, img_ids=img_ids, txt_ids=txt_ids,
                      guidance=torch.tensor([guidance]) if dev else None, return_dict=False)
        ref.add("transformer_latent", latent)
        ref.add("transformer_context", context)
        ref.add("transformer_pooled", pooled)
        ref.add("transformer_timestep", timestep * 1000)
        ref.add("transformer_guidance", torch.tensor([guidance]))
        prediction = transformer(pack(latent), **kwargs)[0]
        ref.add("transformer_out", FluxPipeline._unpack_latents(prediction, HEIGHT, WIDTH, 8))

        quantized = FluxTransformer2DModelFrom(transformer, from_bfl(dequantized, state))
        prediction = quantized(pack(latent), **kwargs)[0]
        ref.add("transformer_q8_out", FluxPipeline._unpack_latents(prediction, HEIGHT, WIDTH, 8))

        vae_latent = torch.randn(1, 16, lh, lw, generator=g)
        ref.add("vae_latent", vae_latent)
        ref.add("vae_decoded", vae.decode(vae_latent).sample)
        image = torch.rand(1, 3, HEIGHT, WIDTH, generator=g) * 2 - 1
        ref.add("vae_image", image)
        ref.add("vae_encoded_mean", vae.encode(image).latent_dist.mean)
        noise = torch.randn(1, 16, lh, lw, generator=torch.Generator().manual_seed(42))
        ref.add("noise", noise)

        image = pipe(PROMPT, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS, guidance_scale=guidance, latents=pack(noise.clone()),
                     max_sequence_length=context_length, output_type="pt").images
        ref.add("image", image[0])
        timesteps = [float(t) for t in pipe.scheduler.timesteps]
        pipe.transformer = quantized
        image = pipe(PROMPT, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS, guidance_scale=guidance, latents=pack(noise.clone()),
                     max_sequence_length=context_length, output_type="pt").images
        ref.add("q8_image", image[0])

    ref.save({"tokens": tokens, "tokens_2": tokens_2, "prompt": PROMPT, "steps": STEPS, "guidance_scale": guidance, "height": HEIGHT,
              "width": WIDTH, "context_length": context_length, "timesteps": timesteps})
    print("written", root)


def FluxTransformer2DModelFrom(template, state):
    import copy
    model = copy.deepcopy(template)
    model.load_state_dict(state)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/flux/tiny-random-flux"))
    parser.add_argument("--tokenizers", default=None, help="a FLUX.1 diffusers folder with tokenizer/ and tokenizer_2/")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    for variant in ("dev", "schnell"):
        make_variant(args.out, variant, args.tokenizers)


if __name__ == "__main__":
    main()
