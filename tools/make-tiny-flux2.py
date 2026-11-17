#!/usr/bin/env python3
"""Creates a tiny random FLUX.2-Klein-shaped pipeline and diffusers reference outputs for the image-generation tests.

The layout of FLUX.2 Klein with very small widths: dual- and single-stream blocks with shared modulations, SwiGLU feed-forwards and 4-axis
RoPE (θ 2000; text on the fourth axis), a Qwen3 text encoder (28 layers so that hidden states 9, 18 and 27 exist) whose hidden states entering
those layers are concatenated per token over the chat-templated prompt (thinking disabled) of 512 tokens, the 2×2-patchified KL VAE latents
normalized by its batch-norm statistics, and the flow-matching Euler scheduler with FLUX.2's empirical shift; step-distilled (no guidance).
The image is 64x96 (24 tokens).

Written to OUT (default ~/models/flux2/tiny-random-flux2):

  diffusers/                       diffusers folder (model_index.json, transformer/, vae/, text_encoder/, tokenizer/, scheduler/)
  single/flux2.safetensors         the transformer under the original (Black Forest Labs) names
  single/flux2-Q8_0.gguf           the same as a transformer-only GGUF (flux architecture id, as unsloth's; matrices Q8_0)
  reference/*.f32                  raw little-endian f32 reference tensors, described by reference/index.json

Requires torch, diffusers, transformers, gguf and safetensors; the tokenizer comes from --tokenizer (a FLUX.2 Klein folder with tokenizer/).
"""

import argparse
import copy
import json
import os

import numpy as np
import torch

PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123"]
PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"
HEIGHT = 64
WIDTH = 96
STEPS = 4
LENGTH = 512


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


def to_original(state, blocks, single_blocks):
    """diffusers Flux2Transformer2DModel names -> the original names (fused q/k/v, (shift, scale) final modulation)."""
    out = {}
    rename = {"x_embedder.weight": "img_in.weight", "context_embedder.weight": "txt_in.weight",
              "time_guidance_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
              "time_guidance_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
              "double_stream_modulation_img.linear.weight": "double_stream_modulation_img.lin.weight",
              "double_stream_modulation_txt.linear.weight": "double_stream_modulation_txt.lin.weight",
              "single_stream_modulation.linear.weight": "single_stream_modulation.lin.weight",
              "proj_out.weight": "final_layer.linear.weight"}
    for k, v in rename.items():
        out[v] = state[k]
    scale, shift = state["norm_out.linear.weight"].chunk(2, 0)
    out["final_layer.adaLN_modulation.1.weight"] = torch.cat([shift, scale], 0)
    for i in range(blocks):
        p, o = f"transformer_blocks.{i}.", f"double_blocks.{i}."
        out[o + "img_attn.qkv.weight"] = torch.cat([state[p + f"attn.to_{x}.weight"] for x in "qkv"], 0)
        out[o + "txt_attn.qkv.weight"] = torch.cat([state[p + f"attn.add_{x}_proj.weight"] for x in "qkv"], 0)
        out[o + "img_attn.proj.weight"] = state[p + "attn.to_out.0.weight"]
        out[o + "txt_attn.proj.weight"] = state[p + "attn.to_add_out.weight"]
        out[o + "img_attn.norm.query_norm.scale"] = state[p + "attn.norm_q.weight"]
        out[o + "img_attn.norm.key_norm.scale"] = state[p + "attn.norm_k.weight"]
        out[o + "txt_attn.norm.query_norm.scale"] = state[p + "attn.norm_added_q.weight"]
        out[o + "txt_attn.norm.key_norm.scale"] = state[p + "attn.norm_added_k.weight"]
        out[o + "img_mlp.0.weight"] = state[p + "ff.linear_in.weight"]
        out[o + "img_mlp.2.weight"] = state[p + "ff.linear_out.weight"]
        out[o + "txt_mlp.0.weight"] = state[p + "ff_context.linear_in.weight"]
        out[o + "txt_mlp.2.weight"] = state[p + "ff_context.linear_out.weight"]
    for i in range(single_blocks):
        p, o = f"single_transformer_blocks.{i}.", f"single_blocks.{i}."
        out[o + "linear1.weight"] = state[p + "attn.to_qkv_mlp_proj.weight"]
        out[o + "linear2.weight"] = state[p + "attn.to_out.weight"]
        out[o + "norm.query_norm.scale"] = state[p + "attn.norm_q.weight"]
        out[o + "norm.key_norm.scale"] = state[p + "attn.norm_k.weight"]
    return out


def write_gguf(path, original):
    import gguf

    writer = gguf.GGUFWriter(path, "flux")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    dequantized = {}
    for name, tensor in original.items():
        data = tensor.to(torch.float32).numpy()
        if data.ndim == 2 and data.shape[1] % 32 == 0 and data.shape[0] > 64:
            q = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(name, q, raw_shape=q.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            dequantized[name] = torch.from_numpy(gguf.quants.dequantize(q, gguf.GGMLQuantizationType.Q8_0).copy())
        else:
            writer.add_tensor(name, data)
            dequantized[name] = torch.from_numpy(data.copy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return dequantized


def from_original(original, state, blocks, single_blocks):
    """The inverse of to_original over the dequantized tensors."""
    back = {}
    rename = {"img_in.weight": "x_embedder.weight", "txt_in.weight": "context_embedder.weight",
              "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
              "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
              "double_stream_modulation_img.lin.weight": "double_stream_modulation_img.linear.weight",
              "double_stream_modulation_txt.lin.weight": "double_stream_modulation_txt.linear.weight",
              "single_stream_modulation.lin.weight": "single_stream_modulation.linear.weight",
              "final_layer.linear.weight": "proj_out.weight"}
    for k, v in rename.items():
        back[v] = original[k]
    shift, scale = original["final_layer.adaLN_modulation.1.weight"].chunk(2, 0)
    back["norm_out.linear.weight"] = torch.cat([scale, shift], 0)
    for i in range(blocks):
        p, o = f"transformer_blocks.{i}.", f"double_blocks.{i}."
        for x, part in zip("qkv", original[o + "img_attn.qkv.weight"].chunk(3, 0)):
            back[p + f"attn.to_{x}.weight"] = part
        for x, part in zip("qkv", original[o + "txt_attn.qkv.weight"].chunk(3, 0)):
            back[p + f"attn.add_{x}_proj.weight"] = part
        back[p + "attn.to_out.0.weight"] = original[o + "img_attn.proj.weight"]
        back[p + "attn.to_add_out.weight"] = original[o + "txt_attn.proj.weight"]
        back[p + "attn.norm_q.weight"] = original[o + "img_attn.norm.query_norm.scale"]
        back[p + "attn.norm_k.weight"] = original[o + "img_attn.norm.key_norm.scale"]
        back[p + "attn.norm_added_q.weight"] = original[o + "txt_attn.norm.query_norm.scale"]
        back[p + "attn.norm_added_k.weight"] = original[o + "txt_attn.norm.key_norm.scale"]
        back[p + "ff.linear_in.weight"] = original[o + "img_mlp.0.weight"]
        back[p + "ff.linear_out.weight"] = original[o + "img_mlp.2.weight"]
        back[p + "ff_context.linear_in.weight"] = original[o + "txt_mlp.0.weight"]
        back[p + "ff_context.linear_out.weight"] = original[o + "txt_mlp.2.weight"]
    for i in range(single_blocks):
        p, o = f"single_transformer_blocks.{i}.", f"single_blocks.{i}."
        back[p + "attn.to_qkv_mlp_proj.weight"] = original[o + "linear1.weight"]
        back[p + "attn.to_out.weight"] = original[o + "linear2.weight"]
        back[p + "attn.norm_q.weight"] = original[o + "norm.query_norm.scale"]
        back[p + "attn.norm_k.weight"] = original[o + "norm.key_norm.scale"]
    return {k: back[k].reshape(v.shape) for k, v in state.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/flux2/tiny-random-flux2"))
    parser.add_argument("--tokenizer", default=os.path.expanduser("~/models/flux2/FLUX.2-klein-4B"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(7)
    blocks, single_blocks = 2, 2
    transformer = Flux2Transformer2DModel(patch_size=1, in_channels=16, num_layers=blocks, num_single_layers=single_blocks, attention_head_dim=32,
                                          num_attention_heads=4, joint_attention_dim=96, mlp_ratio=3.0, axes_dims_rope=(8, 8, 8, 8), rope_theta=2000,
                                          guidance_embeds=False)
    vae = AutoencoderKLFlux2(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(32, 32, 64, 64), layers_per_block=1,
                             norm_num_groups=32, down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, subfolder="tokenizer")
    text = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=28, num_attention_heads=2,
                                        num_key_value_heads=1, head_dim=16, max_position_embeddings=1024, rope_theta=1000000.0, rms_norm_eps=1e-6,
                                        tie_word_embeddings=True))
    with torch.no_grad():
        for model in (transformer, vae, text):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        vae.bn.running_mean.copy_(torch.randn(16) * 0.3)
        vae.bn.running_var.copy_(torch.rand(16) + 0.5)
    transformer.eval()
    text.eval()
    vae.eval()
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
                                                base_image_seq_len=256, max_image_seq_len=4096, time_shift_type="exponential")
    pipe = Flux2KleinPipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=transformer, is_distilled=True)
    root = args.out
    pipe.save_pretrained(os.path.join(root, "diffusers"), safe_serialization=True)
    os.makedirs(os.path.join(root, "single"), exist_ok=True)
    state = {k: v.contiguous() for k, v in transformer.state_dict().items()}
    original = to_original(state, blocks, single_blocks)
    save_file({k: v.contiguous() for k, v in original.items()}, os.path.join(root, "single", "flux2.safetensors"))
    dequantized = write_gguf(os.path.join(root, "single", "flux2-Q8_0.gguf"), original)
    quantized = copy.deepcopy(transformer)
    quantized.load_state_dict(from_original(dequantized, state, blocks, single_blocks))
    quantized.eval()

    ref = Reference(os.path.join(root, "reference"))
    tokens, lengths = {}, {}
    for i, prompt in enumerate(PROMPTS):
        templated = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        t = tokenizer(templated, padding="max_length", max_length=LENGTH, truncation=True, return_tensors="pt")
        tokens[prompt] = t.input_ids[0].tolist()
        lengths[prompt] = int(t.attention_mask[0].sum())
        ref.add(f"text_hidden_{i}", pipe.encode_prompt(prompt, device="cpu", max_sequence_length=LENGTH)[0][0])
    template = tokenizer.apply_chat_template([{"role": "user", "content": "{prompt}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    g = torch.Generator().manual_seed(1)
    lh, lw = HEIGHT // 16, WIDTH // 16
    latent = torch.randn(1, 16, lh, lw, generator=g)
    context = torch.randn(1, LENGTH, 96, generator=g)
    sigma = 0.7317
    text_ids = Flux2KleinPipeline._prepare_text_ids(context)
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(latent)

    def run(model):
        out = model(hidden_states=Flux2KleinPipeline._pack_latents(latent), encoder_hidden_states=context, timestep=torch.tensor([sigma]),
                    img_ids=latent_ids, txt_ids=text_ids, return_dict=False)[0]
        return out.permute(0, 2, 1).reshape(1, 16, lh, lw)

    ref.add("transformer_latent", latent)
    ref.add("transformer_context", context)
    ref.add("transformer_timestep", torch.tensor([sigma * 1000]))
    ref.add("transformer_out", run(transformer))
    ref.add("transformer_q8_out", run(quantized))

    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    vae_latent = torch.randn(1, 16, lh, lw, generator=g)
    ref.add("vae_latent", vae_latent)
    ref.add("vae_decoded", vae.decode(Flux2KleinPipeline._unpatchify_latents(vae_latent * std + mean)).sample)
    image_in = torch.rand(1, 3, HEIGHT, WIDTH, generator=g)
    ref.add("vae_image", image_in)
    encoded = Flux2KleinPipeline._patchify_latents(vae.encode(image_in * 2 - 1).latent_dist.mean)
    ref.add("vae_encoded", (encoded - mean) / std)
    noise = torch.randn(1, 16, lh, lw, generator=torch.Generator().manual_seed(42))
    ref.add("noise", noise)
    kwargs = dict(height=HEIGHT, width=WIDTH, num_inference_steps=STEPS, guidance_scale=1.0, max_sequence_length=LENGTH, output_type="pt")
    ref.add("image", pipe(prompt=PROMPT, latents=noise.clone(), **kwargs).images[0])
    timesteps = [float(t) for t in pipe.scheduler.timesteps]
    # Editing: one reference image (its tokens at the time coordinate 10) of the output's size, from the same noise.
    import PIL.Image

    pixels = (torch.rand(HEIGHT, WIDTH, 3, generator=torch.Generator().manual_seed(5)) * 255).round().to(torch.uint8)
    ref.add("edit_input", pixels.permute(2, 0, 1).to(torch.float32) / 255)
    reference = PIL.Image.fromarray(pixels.numpy())
    ref.add("edit_image", pipe(prompt=PROMPT, image=[reference], latents=noise.clone(), **kwargs).images[0])
    pipe.transformer = quantized
    ref.add("q8_image", pipe(prompt=PROMPT, latents=noise.clone(), **kwargs).images[0])
    ref.save({"tokens": tokens, "lengths": lengths, "template": template, "prompt": PROMPT, "steps": STEPS, "height": HEIGHT, "width": WIDTH,
              "length": LENGTH, "timesteps": timesteps})
    print("written", root)


if __name__ == "__main__":
    main()
