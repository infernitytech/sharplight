#!/usr/bin/env python3
"""Creates a tiny random Qwen-Image-2.1-shaped pipeline and diffusers reference outputs for the image-generation tests.

The layout of Qwen-Image 2.1 with very small widths: a single-stream DiT over [text, image] with block-causal attention (text causal,
the image block bidirectional and attending to all text), one shared modulation projection (scale and tanh gate, no shift; text tokens
modulated from t = 0), zero-centred RMS-normed text input MLP, SwiGLU, 3-axis RoPE with the image frame position after the text; a
Qwen3-VL text encoder whose last hidden states before the final norm, minus the system prompt's tokens, are the text features; the
residual 16x VAE (four input / output channels, separate decoder width); the flow-matching Euler scheduler with the exponential dynamic
shift. The image is 64x96 (24 image tokens).

Written to OUT (default ~/models/qwen-image21/tiny-random-qwen-image21):

  diffusers/                          diffusers folder (model_index.json, transformer/, vae/, text_encoder/, processor/, scheduler/)
  single/qwen-image21.safetensors     the transformer as ComfyUI's single file (model.diffusion_model. prefix, fused img_mlp.gate_up)
  single/qwen-image21-Q8_0.gguf       the same as an unsloth-style GGUF without metadata (matrices Q8_0)
  reference/*.f32                     raw little-endian f32 reference tensors, described by reference/index.json

Requires torch, diffusers, transformers, gguf and safetensors; the processor comes from --processor (a Qwen-Image 2.1 folder with
processor/) or the Hugging Face hub.
"""

import argparse
import copy
import json
import os

import numpy as np
import torch

PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123"]
PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"
NEGATIVE = " "
HEIGHT = 64
WIDTH = 96
STEPS = 4
GUIDANCE = 4.0
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


def to_comfy(state):
    """diffusers names -> ComfyUI's (model.diffusion_model. prefix, img_mlp.gate_layer and img_mlp.proj fused as gate_up, gate first)."""
    out = {}
    for name, value in state.items():
        if ".img_mlp.proj." in name:
            continue
        if ".img_mlp.gate_layer." in name:
            out["model.diffusion_model." + name.replace("gate_layer", "gate_up")] = torch.cat([value, state[name.replace("gate_layer", "proj")]], 0)
            continue
        out["model.diffusion_model." + name] = value
    return out


def write_gguf(path, state):
    import gguf

    writer = gguf.GGUFWriter(path, "qwen_image21")
    dequantized = {}
    for name, tensor in state.items():
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


def from_comfy(original, template):
    state = {}
    for name in template:
        if ".img_mlp.gate_layer." in name or ".img_mlp.proj." in name:
            fused = original["model.diffusion_model." + name.replace("gate_layer", "gate_up").replace(".proj.", ".gate_up.")]
            state[name] = fused.chunk(2, 0)[0 if "gate_layer" in name else 1]
        else:
            state[name] = original["model.diffusion_model." + name].reshape(template[name].shape)
    return state


class TextProcessor:
    """What the text-to-image path of QwenImage21Pipeline needs of its Qwen3-VL processor (call, chat template, tokenizer), for
    environments without torchvision, which transformers' Qwen3-VL video processor requires."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text=None, **kwargs):
        kwargs.pop("images", None)
        return self.tokenizer(text, **kwargs)

    def apply_chat_template(self, messages, **kwargs):
        flat = [{"role": m["role"], "content": "".join(c["text"] for c in m["content"])} for m in messages]
        result = self.tokenizer.apply_chat_template(flat, **kwargs)
        # The processor returns a batch of one conversation.
        return [result] if isinstance(result, list) and result and isinstance(result[0], int) else result

    def save_pretrained(self, folder, **kwargs):
        self.tokenizer.save_pretrained(folder)


def load_processor(source):
    """The pipeline's Qwen3-VL processor, or the text-only stand-in when torchvision is missing."""
    from transformers import AutoProcessor, AutoTokenizer

    try:
        return AutoProcessor.from_pretrained(source, subfolder="processor")
    except ImportError:
        folder = os.path.join(source, "processor") if os.path.isdir(source) else source
        tokenizer = AutoTokenizer.from_pretrained(folder) if os.path.isdir(source) else AutoTokenizer.from_pretrained(source, subfolder="processor")
        return TextProcessor(tokenizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/qwen-image21/tiny-random-qwen-image21"))
    parser.add_argument("--processor", default=None, help="a Qwen-Image 2.1 diffusers folder with processor/")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline, QwenImage21Transformer2DModel
    from safetensors.torch import save_file
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(7)
    transformer = QwenImage21Transformer2DModel(patch_size=1, in_channels=16, out_channels=16, num_layers=2, attention_head_dim=32,
                                                num_attention_heads=4, context_in_dim=64, mlp_ratio=3, axes_dims_rope=(4, 14, 14))
    rng = torch.Generator().manual_seed(3)
    latents_mean = (torch.randn(16, generator=rng) * 0.5).tolist()
    latents_std = (torch.rand(16, generator=rng) + 0.75).tolist()
    vae = AutoencoderKLQwenImage21(base_dim=16, decoder_base_dim=24, z_dim=16, dim_mult=[1, 1, 2, 2, 2], num_res_blocks=1,
                                   temperal_downsample=[False, True, True, True], latents_mean=latents_mean, latents_std=latents_std)
    processor = load_processor(args.processor or "Qwen/Qwen-Image-2.1")
    config = Qwen3VLConfig(
        text_config={"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 16, "vocab_size": 151936, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
                     "rope_parameters": {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True, "rope_theta": 5000000.0}},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 64, "patch_size": 16,
                       "deepstack_visual_indexes": [0], "num_position_embeddings": 64},
        image_token_id=151655, video_token_id=151656, vision_start_token_id=151652, vision_end_token_id=151653, tie_word_embeddings=False)
    text = Qwen3VLForConditionalGeneration(config)
    with torch.no_grad():
        for model in (transformer, vae, text):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
    transformer.eval()
    text.eval()
    vae.eval()
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=True, base_shift=0.5, max_shift=0.9,
                                                base_image_seq_len=256, max_image_seq_len=8192, shift_terminal=0.02,
                                                time_shift_type="exponential")
    pipe = QwenImage21Pipeline(scheduler=scheduler, vae=vae, text_encoder=text, processor=processor, transformer=transformer)
    root = args.out
    pipe.save_pretrained(os.path.join(root, "diffusers"), safe_serialization=True)
    if isinstance(processor, TextProcessor):
        # diffusers saves only real processors: the stand-in's tokenizer goes where the pipeline keeps it.
        processor.tokenizer.save_pretrained(os.path.join(root, "diffusers", "processor"))
    os.makedirs(os.path.join(root, "single"), exist_ok=True)
    state = {k: v.contiguous() for k, v in transformer.state_dict().items()}
    comfy = to_comfy(state)
    save_file({k: v.contiguous() for k, v in comfy.items()}, os.path.join(root, "single", "qwen-image21.safetensors"))
    dequantized = write_gguf(os.path.join(root, "single", "qwen-image21-Q8_0.gguf"), comfy)
    quantized = copy.deepcopy(transformer)
    quantized.load_state_dict(from_comfy(dequantized, state))
    quantized.eval()

    ref = Reference(os.path.join(root, "reference"))
    tokenizer = processor.tokenizer
    drop = pipe._drop_idx
    tokens, lengths = {}, {}
    for i, prompt in enumerate(PROMPTS):
        t = tokenizer([pipe.prompt_template_t2i.format(prompt)], padding="max_length", max_length=LENGTH + drop, truncation=True, return_tensors="pt")
        tokens[prompt] = t.input_ids[0].tolist()
        lengths[prompt] = int(t.attention_mask[0].sum())
        embeds = pipe.encode_prompt(prompt)[0]
        ref.add(f"text_hidden_{i}", embeds[0])

    g = torch.Generator().manual_seed(1)
    lh, lw = HEIGHT // 16, WIDTH // 16
    latent = torch.randn(1, 16, lh, lw, generator=g)
    context = torch.randn(LENGTH, 64, generator=g)
    valid = 37
    mask = (torch.arange(LENGTH) < valid).to(torch.float32)
    sigma = 0.7317
    packed = QwenImage21Pipeline._pack_latents(latent, 1, 16, lh, lw)
    img_mask = torch.tensor([[False] * valid + [True] * (lh * lw // 4)])

    def run(model):
        out = model(hidden_states=packed, encoder_hidden_states=context[None, :valid], timestep=torch.tensor([sigma]), img_shapes=[[(1, lh, lw)]],
                    img_mask=img_mask, return_dict=False)[0]
        return out[:, -lh * lw:].transpose(1, 2).reshape(1, 16, lh, lw)

    ref.add("transformer_latent", latent)
    ref.add("transformer_context", context[None])
    ref.add("transformer_context_mask", mask[None])
    ref.add("transformer_timestep", torch.tensor([sigma * 1000]))
    ref.add("transformer_out", run(transformer))
    ref.add("transformer_q8_out", run(quantized))

    vae_latent = torch.randn(1, 16, lh, lw, generator=g)
    ref.add("vae_latent", vae_latent)
    mean = torch.tensor(latents_mean).view(1, 16, 1, 1, 1)
    std = torch.tensor(latents_std).view(1, 16, 1, 1, 1)
    ref.add("vae_decoded", vae.decode(vae_latent[:, :, None] * std + mean).sample[:, :3, 0])
    image_in = torch.rand(1, 3, HEIGHT, WIDTH, generator=g)
    ref.add("vae_image", image_in)
    rgba = torch.cat([image_in, torch.ones(1, 1, HEIGHT, WIDTH)], 1) * 2 - 1
    encoded = vae.encode(rgba[:, :, None]).latent_dist.mean
    ref.add("vae_encoded", ((encoded - mean) / std)[:, :, 0])
    noise = torch.randn(1, 16, lh, lw, generator=torch.Generator().manual_seed(42))
    ref.add("noise", noise)
    noise_packed = QwenImage21Pipeline._pack_latents(noise, 1, 16, lh, lw)
    kwargs = dict(negative_prompt=NEGATIVE, true_cfg_scale=GUIDANCE, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS, output_type="pt")
    image = pipe(PROMPT, latents=noise_packed.clone(), **kwargs).images
    ref.add("image", image[0][:3])
    timesteps = [float(t) for t in pipe.scheduler.timesteps]
    pipe.transformer = quantized
    image = pipe(PROMPT, latents=noise_packed.clone(), **kwargs).images
    ref.add("q8_image", image[0][:3])
    template = pipe.prompt_template_t2i.replace("{}", "{prompt}")
    ref.save({"tokens": tokens, "lengths": lengths, "template": template, "drop": drop, "prompt": PROMPT, "negative": NEGATIVE,
              "guidance": GUIDANCE, "steps": STEPS, "height": HEIGHT, "width": WIDTH, "length": LENGTH, "timesteps": timesteps})
    print("written", root, "drop", drop)


if __name__ == "__main__":
    main()
