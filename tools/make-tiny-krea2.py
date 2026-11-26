#!/usr/bin/env python3
"""Creates a tiny random Krea-2-shaped pipeline and diffusers reference outputs for the image-generation tests.

The layout of Krea 2 with very small widths: a single-stream DiT over [text, image] (grouped-query attention with q/k RMS norms, a sigmoid
output gate and a text key-padding mask; one shared time modulation plus a learned table per block; SwiGLU; zero-centred RMS norms
everywhere; 3-axis RoPE with the text at the origin), a text-fusion stage (blocks across the tapped encoder layers of every token, a linear
projector over the layers, refiner blocks across the tokens), a Qwen3-VL text encoder tapped at several layers, the Qwen-Image VAE and the
flow-matching Euler scheduler with the exponential dynamic shift. The image is 64x96 (4x6 = 24 image tokens of 2x2 latent patches).

Written to OUT (default ~/models/krea2/tiny-random-krea2):

  diffusers/                 diffusers folder (model_index.json, transformer/, vae/, text_encoder/, tokenizer/, scheduler/)
  single/krea2-raw.safetensors   the transformer in the original Krea names (as the release's turbo.safetensors and ComfyUI's files)
  single/krea2-raw-Q8_0.gguf     the same as a GGUF (architecture krea2, matrices Q8_0)
  reference/*.f32            raw little-endian f32 reference tensors, described by reference/index.json

Requires torch, diffusers (with Krea2Pipeline), transformers, gguf and safetensors; the tokenizer comes from --tokenizer (a Krea 2 diffusers
folder with tokenizer/) or the Hugging Face hub.
"""

import argparse
import copy
import json
import os

import numpy as np
import torch

PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123"]
PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"
NEGATIVE = ""
HEIGHT = 64
WIDTH = 96
STEPS = 4
# Krea 2's convention: cond + g * (cond - uncond), the usual guidance scale 1 + g.
KREA_GUIDANCE = 3.0
LENGTH = 512
# Every second of six layers, ending one before the last (the released pipelines: every third of 36, 2 ... 35).
SELECT_LAYERS = (1, 3, 5)

# Original Krea names (turbo.safetensors) <- diffusers names, the inverse of diffusers' convert_krea2_transformer_checkpoint_to_diffusers.
PREFIXES = [
    ("img_in.", "first."),
    ("time_embed.linear_1.", "tmlp.0."),
    ("time_embed.linear_2.", "tmlp.2."),
    ("time_mod_proj.", "tproj.1."),
    ("txt_in.norm.weight", "txtmlp.0.scale"),
    ("txt_in.linear_1.", "txtmlp.1."),
    ("txt_in.linear_2.", "txtmlp.3."),
    ("text_fusion.", "txtfusion."),
    ("transformer_blocks.", "blocks."),
    ("final_layer.linear.", "last.linear."),
    ("final_layer.norm.weight", "last.norm.scale"),
    ("final_layer.scale_shift_table", "last.modulation.lin"),
]
PARTS = [
    (".attn.to_q.", ".attn.wq."),
    (".attn.to_k.", ".attn.wk."),
    (".attn.to_v.", ".attn.wv."),
    (".attn.to_out.0.", ".attn.wo."),
    (".attn.to_gate.", ".attn.gate."),
    (".attn.norm_q.weight", ".attn.qknorm.qnorm.scale"),
    (".attn.norm_k.weight", ".attn.qknorm.knorm.scale"),
    (".ff.", ".mlp."),
    (".norm1.weight", ".prenorm.scale"),
    (".norm2.weight", ".postnorm.scale"),
    (".scale_shift_table", ".mod.lin"),
]


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


def original_name(name):
    for new, old in PREFIXES:
        if name.startswith(new):
            name = old + name[len(new):]
            break
    for new, old in PARTS:
        name = name.replace(new, old)
    return name


def to_original(state):
    """diffusers names -> the original Krea names (the blocks' modulation tables flattened, as released)."""
    out = {}
    for name, value in state.items():
        key = original_name(name)
        if key.startswith("blocks.") and key.endswith(".mod.lin"):
            value = value.reshape(-1)
        out[key] = value.contiguous()
    return out


def write_gguf(path, state):
    import gguf

    writer = gguf.GGUFWriter(path, "krea2")
    dequantized = {}
    for name, tensor in state.items():
        data = tensor.to(torch.float32).numpy()
        if data.ndim == 2 and data.shape[1] % 32 == 0 and data.shape[0] > 32:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/krea2/tiny-random-krea2"))
    parser.add_argument("--tokenizer", default=None, help="a Krea 2 diffusers folder with tokenizer/")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Pipeline, Krea2Transformer2DModel
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLModel

    torch.manual_seed(7)
    transformer = Krea2Transformer2DModel(in_channels=64, num_layers=2, attention_head_dim=32, num_attention_heads=4, num_key_value_heads=2,
                                          intermediate_size=96, timestep_embed_dim=256, text_hidden_dim=64, num_text_layers=len(SELECT_LAYERS),
                                          text_num_attention_heads=4, text_num_key_value_heads=2, text_intermediate_size=80,
                                          num_layerwise_text_blocks=2, num_refiner_text_blocks=1, axes_dims_rope=(8, 12, 12), rope_theta=1000.0)
    rng = torch.Generator().manual_seed(3)
    latents_mean = (torch.randn(16, generator=rng) * 0.5).tolist()
    latents_std = (torch.rand(16, generator=rng) + 0.75).tolist()
    vae = AutoencoderKLQwenImage(base_dim=32, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1, attn_scales=[],
                                 temperal_downsample=[False, True, True], latents_mean=latents_mean, latents_std=latents_std)
    source = args.tokenizer or "krea/Krea-2-Turbo"  # gated: accept the terms and set HF_TOKEN, or pass a local folder
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(source, "tokenizer")) if os.path.isdir(source) \
        else AutoTokenizer.from_pretrained(source, subfolder="tokenizer")
    config = Qwen3VLConfig(
        text_config={"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 6, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 16, "vocab_size": 151936, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
                     "rope_parameters": {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True, "rope_theta": 5000000.0}},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 64, "patch_size": 16,
                       "deepstack_visual_indexes": [0], "num_position_embeddings": 64},
        image_token_id=151655, video_token_id=151656, vision_start_token_id=151652, vision_end_token_id=151653, tie_word_embeddings=True)
    text = Qwen3VLModel(config)
    with torch.no_grad():
        for model in (transformer, vae, text):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
    transformer.eval()
    text.eval()
    vae.eval()
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
                                                base_image_seq_len=256, max_image_seq_len=6400, time_shift_type="exponential")
    pipe = Krea2Pipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=transformer,
                         text_encoder_select_layers=SELECT_LAYERS, is_distilled=False, patch_size=2)
    root = args.out
    pipe.save_pretrained(os.path.join(root, "diffusers"), safe_serialization=True)
    os.makedirs(os.path.join(root, "single"), exist_ok=True)
    state = {k: v.contiguous() for k, v in transformer.state_dict().items()}
    original = to_original(state)
    save_file(original, os.path.join(root, "single", "krea2-raw.safetensors"))
    dequantized = write_gguf(os.path.join(root, "single", "krea2-raw-Q8_0.gguf"), original)
    quantized = copy.deepcopy(transformer)
    quantized.load_state_dict({name: dequantized[original_name(name)].reshape(value.shape) for name, value in state.items()})
    quantized.eval()

    ref = Reference(os.path.join(root, "reference"))
    skip = pipe.prompt_template_encode_start_idx
    tokens, lengths = {}, {}
    for i, prompt in enumerate(PROMPTS):
        hidden, mask = pipe.get_text_hidden_states(prompt, LENGTH)
        # The pipeline's layout is [template + prompt, padding, assistant turn]; Sharplight's [template + prompt + assistant turn, padding]
        # (the text rows have no positions in the transformer and padding is masked, so only the order of the valid rows differs).
        text_ids = tokenizer([pipe.prompt_template_encode_prefix + prompt], truncation=True, padding="max_length",
                             max_length=LENGTH + skip - pipe.prompt_template_encode_num_suffix_tokens, return_tensors="pt")
        suffix = tokenizer([pipe.prompt_template_encode_suffix], return_tensors="pt")
        ids = torch.cat([text_ids.input_ids, suffix.input_ids], 1)[0]
        valid = torch.cat([text_ids.attention_mask, suffix.attention_mask], 1)[0].bool()
        tokens[prompt] = ids[valid].tolist()
        lengths[prompt] = int(valid.sum())
        ref.add(f"text_hidden_{i}", hidden[0][mask[0]].reshape(int(mask[0].sum()), -1))

    g = torch.Generator().manual_seed(1)
    lh, lw = HEIGHT // 8, WIDTH // 8
    latent = torch.randn(1, 16, lh, lw, generator=g)
    context = torch.randn(LENGTH, len(SELECT_LAYERS), 64, generator=g)
    valid = 37
    mask = torch.arange(LENGTH) < valid
    sigma = 0.7317
    packed = pipe._pack_latents(latent, 1, 16, lh, lw)
    positions = Krea2Pipeline.prepare_position_ids(LENGTH, lh // 2, lw // 2, "cpu")

    def run(model):
        out = model(hidden_states=packed, encoder_hidden_states=context[None], timestep=torch.tensor([sigma]), position_ids=positions,
                    encoder_attention_mask=mask[None], return_dict=False)[0]
        out = out.view(1, lh // 2, lw // 2, 16, 2, 2).permute(0, 3, 1, 4, 2, 5)
        return out.reshape(1, 16, lh, lw)

    ref.add("transformer_latent", latent)
    ref.add("transformer_context", context.reshape(1, LENGTH, -1))
    ref.add("transformer_context_mask", mask.to(torch.float32)[None])
    ref.add("transformer_timestep", torch.tensor([sigma * 1000]))
    ref.add("transformer_out", run(transformer))
    ref.add("transformer_q8_out", run(quantized))

    vae_latent = torch.randn(1, 16, lh, lw, generator=g)
    ref.add("vae_latent", vae_latent)
    mean = torch.tensor(latents_mean).view(1, 16, 1, 1, 1)
    std = torch.tensor(latents_std).view(1, 16, 1, 1, 1)
    ref.add("vae_decoded", vae.decode(vae_latent[:, :, None] * std + mean).sample[:, :, 0])
    image_in = torch.rand(1, 3, HEIGHT, WIDTH, generator=g)
    ref.add("vae_image", image_in)
    encoded = vae.encode((image_in * 2 - 1)[:, :, None]).latent_dist.mean
    ref.add("vae_encoded", ((encoded - mean) / std)[:, :, 0])
    noise = torch.randn(1, 16, lh, lw, generator=torch.Generator().manual_seed(42))
    ref.add("noise", noise)
    noise_packed = pipe._pack_latents(noise, 1, 16, lh, lw)
    kwargs = dict(negative_prompt=NEGATIVE, guidance_scale=KREA_GUIDANCE, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS, output_type="pt")
    image = pipe(PROMPT, latents=noise_packed.clone(), **kwargs).images
    ref.add("image", image[0])
    timesteps = [float(t) for t in pipe.scheduler.timesteps]
    pipe.transformer = quantized
    image = pipe(PROMPT, latents=noise_packed.clone(), **kwargs).images
    ref.add("q8_image", image[0])
    template = pipe.prompt_template_encode_prefix + "{prompt}" + pipe.prompt_template_encode_suffix
    ref.save({"tokens": tokens, "lengths": lengths, "template": template, "drop": skip, "prompt": PROMPT, "negative": NEGATIVE,
              "guidance": 1 + KREA_GUIDANCE, "steps": STEPS, "height": HEIGHT, "width": WIDTH, "length": LENGTH, "timesteps": timesteps})
    print("written", root)


if __name__ == "__main__":
    main()
