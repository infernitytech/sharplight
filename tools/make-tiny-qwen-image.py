#!/usr/bin/env python3
"""Creates a tiny random Qwen-Image-shaped pipeline and diffusers reference outputs for the image-generation tests.

The layout of Qwen-Image with very small widths: a dual-stream MMDiT (per-stream modulation, q/k RMS norms, joint attention over
[text, image], 3-axis RoPE over centred image positions with the text after them, RMS-normed text input), a Qwen2.5-VL text encoder
whose last (final-normed) hidden states over the system-templated prompt, minus the template's first 34 tokens, are the text features,
the causal 3-D VAE of Qwen-Image (Wan 2.1 layout, per-channel latent normalization) and the flow-matching Euler scheduler with the
exponential dynamic shift and the terminal stretch; classifier-free guidance with diffusers' norm rescaling.
The image is 64×96 (24 image tokens, a non-square RoPE grid).

Written to OUT (default ~/models/qwen-image/tiny-random-qwen-image):

  diffusers/                       diffusers folder (model_index.json, transformer/, vae/, text_encoder/, tokenizer/, scheduler/)
  single/qwen-image.safetensors    the transformer as a single file (ComfyUI's names: the diffusers ones under model.diffusion_model.)
  single/qwen-image-Q8_0.gguf      the same as a transformer-only GGUF (city96's qwen_image architecture id and names; matrices Q8_0)
  reference/*.f32                  raw little-endian f32 reference tensors, described by reference/index.json

Requires torch, diffusers, transformers, gguf and safetensors. The tokenizer comes from --tokenizer (a Qwen-Image diffusers folder with
tokenizer/) or the Hugging Face hub.
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
DROP = 34


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


def write_gguf(path, state):
    """Writes a transformer-only GGUF (2-D weights with rows of 32k values Q8_0, the rest F32) and returns the dequantized weights."""
    import gguf

    writer = gguf.GGUFWriter(path, "qwen_image")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/qwen-image/tiny-random-qwen-image"))
    parser.add_argument("--tokenizer", default=None, help="a Qwen-Image diffusers folder with tokenizer/")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImagePipeline, QwenImageTransformer2DModel
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(7)
    transformer = QwenImageTransformer2DModel(patch_size=2, in_channels=64, out_channels=16, num_layers=2, attention_head_dim=32,
                                              num_attention_heads=4, joint_attention_dim=64, axes_dims_rope=(4, 14, 14))
    rng = torch.Generator().manual_seed(3)
    latents_mean = (torch.randn(16, generator=rng) * 0.5).tolist()
    latents_std = (torch.rand(16, generator=rng) + 0.75).tolist()
    vae = AutoencoderKLQwenImage(base_dim=32, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1, attn_scales=[],
                                 temperal_downsample=[False, True, True], latents_mean=latents_mean, latents_std=latents_std)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or "Qwen/Qwen-Image", subfolder="tokenizer")
    config = Qwen2_5_VLConfig(
        text_config={"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "vocab_size": 152064, "max_position_embeddings": 2048, "rms_norm_eps": 1e-6,
                     "rope_parameters": {"rope_type": "default", "mrope_section": [2, 3, 3], "rope_theta": 1000000.0}},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 64},
        vocab_size=152064, vision_start_token_id=151652, vision_end_token_id=151653, vision_token_id=151654, image_token_id=151655,
        video_token_id=151656, tie_word_embeddings=False)
    text = Qwen2_5_VLForConditionalGeneration(config)
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
    pipe = QwenImagePipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=transformer)
    root = args.out
    pipe.save_pretrained(os.path.join(root, "diffusers"), safe_serialization=True)
    os.makedirs(os.path.join(root, "single"), exist_ok=True)
    state = {k: v.contiguous() for k, v in transformer.state_dict().items()}
    save_file({"model.diffusion_model." + k: v for k, v in state.items()}, os.path.join(root, "single", "qwen-image.safetensors"))
    dequantized = write_gguf(os.path.join(root, "single", "qwen-image-Q8_0.gguf"), state)
    quantized = copy.deepcopy(transformer)
    quantized.load_state_dict({k: dequantized[k].reshape(v.shape) for k, v in state.items()})
    quantized.eval()

    ref = Reference(os.path.join(root, "reference"))
    tokens = {}
    lengths = {}
    template = pipe.prompt_template_encode
    for i, prompt in enumerate(PROMPTS):
        t = tokenizer([template.format(prompt)], padding="max_length", max_length=LENGTH + DROP, truncation=True, return_tensors="pt")
        tokens[prompt] = t.input_ids[0].tolist()
        lengths[prompt] = int(t.attention_mask[0].sum())
        embeds, _ = pipe.encode_prompt(prompt, max_sequence_length=LENGTH)
        ref.add(f"text_hidden_{i}", embeds[0])

    g = torch.Generator().manual_seed(1)
    lh, lw = HEIGHT // 8, WIDTH // 8
    latent = torch.randn(1, 16, lh, lw, generator=g)
    context = torch.randn(LENGTH, 64, generator=g)
    valid = 37
    mask = (torch.arange(LENGTH) < valid).to(torch.float32)
    sigma = 0.7317
    packed = QwenImagePipeline._pack_latents(latent, 1, 16, lh, lw)
    shapes = [[(1, lh // 2, lw // 2)]]

    def run(model):
        out = model(hidden_states=packed, encoder_hidden_states=context[None, :valid], encoder_hidden_states_mask=None,
                    timestep=torch.tensor([sigma]), img_shapes=shapes, return_dict=False)[0]
        return QwenImagePipeline._unpack_latents(out, HEIGHT, WIDTH, 8)[:, :, 0]

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
    ref.add("vae_decoded", vae.decode(vae_latent[:, :, None] * std + mean).sample[:, :, 0])
    image_in = torch.rand(1, 3, HEIGHT, WIDTH, generator=g)
    ref.add("vae_image", image_in)
    encoded = vae.encode(image_in[:, :, None] * 2 - 1).latent_dist.mean
    ref.add("vae_encoded", ((encoded - mean) / std)[:, :, 0])
    noise = torch.randn(1, 16, lh, lw, generator=torch.Generator().manual_seed(42))
    ref.add("noise", noise)
    noise_packed = QwenImagePipeline._pack_latents(noise, 1, 16, lh, lw)
    image = pipe(PROMPT, negative_prompt=NEGATIVE, true_cfg_scale=GUIDANCE, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
                 latents=noise_packed.clone(), max_sequence_length=LENGTH, output_type="pt").images
    ref.add("image", image[0])
    timesteps = [float(t) for t in pipe.scheduler.timesteps]
    pipe.transformer = quantized
    image = pipe(PROMPT, negative_prompt=NEGATIVE, true_cfg_scale=GUIDANCE, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
                 latents=noise_packed.clone(), max_sequence_length=LENGTH, output_type="pt").images
    ref.add("q8_image", image[0])
    ref.save({"tokens": tokens, "lengths": lengths, "template": template, "drop": DROP, "prompt": PROMPT, "negative": NEGATIVE,
              "guidance": GUIDANCE, "steps": STEPS, "height": HEIGHT, "width": WIDTH, "length": LENGTH, "timesteps": timesteps})
    print("written", root)


if __name__ == "__main__":
    main()
