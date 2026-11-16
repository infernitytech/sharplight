#!/usr/bin/env python3
"""Creates a tiny random Chroma-shaped pipeline and diffusers reference outputs for the image-generation tests.

The layout of Chroma1 (a FLUX.1-schnell derivative) with very small widths: dual-stream / single-stream MMDiT blocks with 3-axis RoPE
whose modulation comes from the "distilled guidance" approximator (an MLP over sinusoidal timestep / index embeddings) instead of
per-block projections, no CLIP, a T5 encoder run with the prompt's attention mask, the transformer's text attention masked except
for the first padding token, a 16-channel VAE, the flow-matching Euler scheduler and real classifier-free guidance.

Written to OUT (default ~/models/chroma/tiny-random-chroma):

  diffusers/                   diffusers folder (model_index.json, transformer/, vae/, text_encoder/, tokenizer/, scheduler/)
  single/chroma.safetensors    the transformer under the original names (FLUX names plus distilled_guidance_layer.*)
  single/chroma-Q8_0.gguf      the same as a transformer-only GGUF (matrices Q8_0, the rest F16)
  reference/*.f32              raw little-endian f32 reference tensors, described by reference/index.json

Requires torch, diffusers, transformers, safetensors and gguf. The T5 tokenizer comes from --tokenizers (a FLUX.1 diffusers folder
with tokenizer_2/) or the Hugging Face hub.
"""

import argparse
import copy
import json
import os

import numpy as np
import torch

PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123", ""]
PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"
NEGATIVE = "blurry, low quality"
HEIGHT = 64
WIDTH = 96
STEPS = 4
GUIDANCE = 4.0
T5_LENGTH = 512


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
    from diffusers import AutoencoderKL, ChromaTransformer2DModel
    from transformers import T5Config, T5EncoderModel

    torch.manual_seed(5)
    transformer = ChromaTransformer2DModel(
        patch_size=1, in_channels=64, num_layers=2, num_single_layers=2, attention_head_dim=32, num_attention_heads=2,
        joint_attention_dim=32, axes_dims_rope=(4, 14, 14), approximator_num_channels=64, approximator_hidden_dim=96, approximator_layers=2)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=16, layers_per_block=2, norm_num_groups=32, block_out_channels=(32, 32, 64, 64),
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4, scaling_factor=0.3611, shift_factor=0.1159,
        use_quant_conv=False, use_post_quant_conv=False)
    text = T5EncoderModel(T5Config(
        vocab_size=32128, d_model=32, d_kv=4, d_ff=64, num_layers=2, num_heads=8, feed_forward_proj="gated-gelu",
        relative_attention_num_buckets=32, relative_attention_max_distance=128, dense_act_fn="gelu_new", is_gated_act=True))
    with torch.no_grad():
        for model in (transformer, vae, text):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
    return transformer.eval(), vae.eval(), text.eval()


def to_original(state, single_layers, double_layers, approximator_layers):
    """diffusers ChromaTransformer2DModel weights → original names (FLUX names without modulation, distilled_guidance_layer.*)."""
    out = {}

    def put(name, value):
        out[name] = value.contiguous()

    for suffix in ("weight", "bias"):
        put(f"img_in.{suffix}", state[f"x_embedder.{suffix}"])
        put(f"txt_in.{suffix}", state[f"context_embedder.{suffix}"])
        put(f"final_layer.linear.{suffix}", state[f"proj_out.{suffix}"])
        put(f"distilled_guidance_layer.in_proj.{suffix}", state[f"distilled_guidance_layer.in_proj.{suffix}"])
        put(f"distilled_guidance_layer.out_proj.{suffix}", state[f"distilled_guidance_layer.out_proj.{suffix}"])
        for i in range(approximator_layers):
            put(f"distilled_guidance_layer.layers.{i}.in_layer.{suffix}", state[f"distilled_guidance_layer.layers.{i}.linear_1.{suffix}"])
            put(f"distilled_guidance_layer.layers.{i}.out_layer.{suffix}", state[f"distilled_guidance_layer.layers.{i}.linear_2.{suffix}"])
        for i in range(double_layers):
            s = f"transformer_blocks.{i}."
            d = f"double_blocks.{i}."
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
            put(f"single_blocks.{i}.linear1.{suffix}",
                torch.cat([state[s + f"attn.to_{x}.{suffix}"] for x in "qkv"] + [state[s + f"proj_mlp.{suffix}"]], dim=0))
            put(f"single_blocks.{i}.linear2.{suffix}", state[s + f"proj_out.{suffix}"])
    for i in range(approximator_layers):
        put(f"distilled_guidance_layer.norms.{i}.scale", state[f"distilled_guidance_layer.norms.{i}.weight"])
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
    assert sum(v.numel() for v in out.values()) == sum(v.numel() for v in state.values())
    return out


def from_original(original, template):
    """Loads original-name weights into a copy of the diffusers model (the inverse of to_original)."""
    state = template.state_dict()
    probe = {k: torch.zeros_like(v) for k, v in state.items()}
    # Map by construction: run to_original on index tensors to find where every diffusers element lands.
    single = sum(1 for k in state if k.startswith("single_transformer_blocks.") and k.endswith("proj_out.weight"))
    double = sum(1 for k in state if k.startswith("transformer_blocks.") and k.endswith("attn.to_q.weight"))
    approximator = sum(1 for k in state if k.startswith("distilled_guidance_layer.norms."))
    offset = 0
    for k, v in state.items():
        probe[k] = torch.arange(offset, offset + v.numel(), dtype=torch.float64).reshape(v.shape)
        offset += v.numel()
    mapped = to_original(probe, single, double, approximator)
    flat = torch.empty(offset, dtype=torch.float32)
    for name, positions in mapped.items():
        flat[positions.reshape(-1).long()] = original[name].reshape(-1).to(torch.float32)
    result = {}
    offset = 0
    for k, v in state.items():
        result[k] = flat[offset:offset + v.numel()].reshape(v.shape)
        offset += v.numel()
    model = copy.deepcopy(template)
    model.load_state_dict(result)
    return model.eval()


def write_gguf(path, original):
    """Writes a transformer-only GGUF (2-D weights with rows of 32k values Q8_0, the rest F16) and returns the dequantized weights."""
    import gguf

    writer = gguf.GGUFWriter(path, "chroma")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    dequantized = {}
    for name, tensor in original.items():
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


def use_boolean_mask(pipe):
    """Makes the pipeline's transformer mask boolean (padding keys excluded), as the original Chroma code and ComfyUI mask them.

    diffusers builds a float 0/1 mask, which scaled_dot_product_attention adds to the scores instead of masking (+1 on the visible
    pairs); Sharplight follows the original semantics, so the references do too.
    """
    original = pipe._prepare_attention_mask

    def prepare(*args, **kwargs):
        mask = original(*args, **kwargs)
        return None if mask is None else mask.bool()

    pipe._prepare_attention_mask = prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/chroma/tiny-random-chroma"))
    parser.add_argument("--tokenizers", default=None, help="a FLUX.1 diffusers folder with tokenizer_2/ (T5)")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    from diffusers import ChromaPipeline, FlowMatchEulerDiscreteScheduler
    from safetensors.torch import save_file
    from transformers import T5TokenizerFast

    transformer, vae, text = build_models()
    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizers or "black-forest-labs/FLUX.1-schnell", subfolder="tokenizer_2")
    # Chroma1-HD's scheduler (static shift 3), which transformer-only files default to.
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False, base_shift=0.5, max_shift=1.15,
                                                base_image_seq_len=256, max_image_seq_len=4096)
    pipe = ChromaPipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=transformer)
    use_boolean_mask(pipe)
    root = args.out
    pipe.save_pretrained(os.path.join(root, "diffusers"), safe_serialization=True)

    state = transformer.state_dict()
    original = to_original(state, 2, 2, 2)
    os.makedirs(os.path.join(root, "single"), exist_ok=True)
    save_file({k: v.to(torch.float32) for k, v in original.items()}, os.path.join(root, "single", "chroma.safetensors"))
    dequantized = write_gguf(os.path.join(root, "single", "chroma-Q8_0.gguf"), original)
    quantized = from_original(dequantized, transformer)

    ref = Reference(os.path.join(root, "reference"))
    tokens = {}
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            t = tokenizer(prompt, padding="max_length", max_length=T5_LENGTH, truncation=True, return_tensors="pt")
            tokens[prompt] = t.input_ids[0].tolist()
            ref.add(f"text_hidden_{i}", text(t.input_ids, attention_mask=t.attention_mask)[0][0])

        g = torch.Generator().manual_seed(1)
        lh, lw = HEIGHT // 8, WIDTH // 8
        latent = torch.randn(1, 16, lh, lw, generator=g)
        context = torch.randn(1, T5_LENGTH, 32, generator=g)
        valid = 7
        mask = (torch.arange(T5_LENGTH) <= valid).to(torch.float32)[None]
        timestep = torch.tensor([0.7317])
        img_ids = ChromaPipeline._prepare_latent_image_ids(1, lh // 2, lw // 2, "cpu", torch.float32)
        txt_ids = torch.zeros(T5_LENGTH, 3)
        full_mask = pipe._prepare_attention_mask(1, img_ids.shape[0], torch.float32, mask)
        packed = ChromaPipeline._pack_latents(latent, 1, 16, lh, lw)
        kwargs = dict(encoder_hidden_states=context, timestep=timestep, img_ids=img_ids, txt_ids=txt_ids, attention_mask=full_mask, return_dict=False)
        ref.add("transformer_latent", latent)
        ref.add("transformer_context", context)
        ref.add("transformer_context_mask", mask)
        ref.add("transformer_timestep", timestep * 1000)
        ref.add("transformer_out", ChromaPipeline._unpack_latents(transformer(packed, **kwargs)[0], HEIGHT, WIDTH, 8))
        ref.add("transformer_q8_out", ChromaPipeline._unpack_latents(quantized(packed, **kwargs)[0], HEIGHT, WIDTH, 8))

        vae_latent = torch.randn(1, 16, lh, lw, generator=g)
        ref.add("vae_latent", vae_latent)
        ref.add("vae_decoded", vae.decode(vae_latent).sample)
        noise = torch.randn(1, 16, lh, lw, generator=torch.Generator().manual_seed(42))
        ref.add("noise", noise)
        packed_noise = ChromaPipeline._pack_latents(noise, 1, 16, lh, lw)
        image = pipe(PROMPT, negative_prompt=NEGATIVE, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                     latents=packed_noise.clone(), max_sequence_length=T5_LENGTH, output_type="pt").images
        ref.add("image", image[0])
        timesteps = [float(t) for t in pipe.scheduler.timesteps]
        pipe.transformer = quantized
        image = pipe(PROMPT, negative_prompt=NEGATIVE, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                     latents=packed_noise.clone(), max_sequence_length=T5_LENGTH, output_type="pt").images
        ref.add("q8_image", image[0])

    ref.save({"tokens": tokens, "prompt": PROMPT, "negative_prompt": NEGATIVE, "steps": STEPS, "guidance_scale": GUIDANCE, "height": HEIGHT,
              "width": WIDTH, "t5_length": T5_LENGTH, "timesteps": timesteps})
    print("written", root)


if __name__ == "__main__":
    main()
