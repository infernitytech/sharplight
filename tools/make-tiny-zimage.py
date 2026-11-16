#!/usr/bin/env python3
"""Creates a tiny random Z-Image-shaped pipeline and diffusers reference outputs for the image-generation tests.

The layout of Z-Image-Turbo with very small widths: a single-stream DiT (modulated refiner blocks over the image tokens, unmodulated ones
over the caption tokens, joint blocks over both; tanh-gated sandwich-normed RMS blocks, SwiGLU, 3-axis RoPE with θ 256, image and caption
tokens padded to multiples of 32 with learned pad tokens), a Qwen3 text encoder whose second-to-last hidden states over the
chat-templated prompt are the caption features, a 16-channel VAE and the flow-matching Euler scheduler with shift 3, no guidance.
The image is 64×96 (24 image tokens: the padding is exercised).

Written to OUT (default ~/models/zimage/tiny-random-zimage):

  diffusers/              diffusers folder (model_index.json, transformer/, vae/, text_encoder/, tokenizer/, scheduler/)
  single/zimage.safetensors   the transformer under its own (diffusers) names in one file
  reference/*.f32         raw little-endian f32 reference tensors, described by reference/index.json

Requires torch, diffusers, transformers and safetensors. The tokenizer comes from --tokenizer (a Z-Image diffusers folder with
tokenizer/) or the Hugging Face hub.
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/zimage/tiny-random-zimage"))
    parser.add_argument("--tokenizer", default=None, help="a Z-Image diffusers folder with tokenizer/")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

    torch.manual_seed(7)
    transformer = ZImageTransformer2DModel(all_patch_size=(2,), all_f_patch_size=(1,), in_channels=16, dim=128, n_layers=2, n_refiner_layers=1,
                                           n_heads=2, n_kv_heads=2, norm_eps=1e-5, qk_norm=True, cap_feat_dim=64, rope_theta=256.0, t_scale=1000.0,
                                           axes_dims=[16, 24, 24], axes_lens=[1024, 64, 64])
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=16, layers_per_block=2, norm_num_groups=32, block_out_channels=(32, 32, 64, 64),
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4, scaling_factor=0.3611, shift_factor=0.1159,
        use_quant_conv=False, use_post_quant_conv=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or "Tongyi-MAI/Z-Image-Turbo", subfolder="tokenizer")
    text = Qwen3Model(Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=2,
                                  num_key_value_heads=1, head_dim=32, max_position_embeddings=1024, rope_theta=1000000.0, rms_norm_eps=1e-6,
                                  tie_word_embeddings=True))
    with torch.no_grad():
        for model in (transformer, vae, text):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        # The pad tokens start at zero; give them values.
        transformer.x_pad_token.copy_(torch.randn_like(transformer.x_pad_token) * 0.5)
        transformer.cap_pad_token.copy_(torch.randn_like(transformer.cap_pad_token) * 0.5)
    transformer.eval()
    text.eval()
    vae.eval()
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False)
    pipe = ZImagePipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=transformer)
    root = args.out
    pipe.save_pretrained(os.path.join(root, "diffusers"), safe_serialization=True)
    os.makedirs(os.path.join(root, "single"), exist_ok=True)
    save_file({k: v.contiguous() for k, v in transformer.state_dict().items()}, os.path.join(root, "single", "zimage.safetensors"))

    ref = Reference(os.path.join(root, "reference"))
    tokens = {}
    lengths = {}
    for i, prompt in enumerate(PROMPTS):
        templated = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        t = tokenizer([templated], padding="max_length", max_length=LENGTH, truncation=True, return_tensors="pt")
        tokens[prompt] = t.input_ids[0].tolist()
        lengths[prompt] = int(t.attention_mask[0].sum())
        hidden = text(input_ids=t.input_ids, attention_mask=t.attention_mask.bool(), output_hidden_states=True).hidden_states[-2][0]
        ref.add(f"text_hidden_{i}", hidden)
    template = tokenizer.apply_chat_template([{"role": "user", "content": "{prompt}"}], tokenize=False, add_generation_prompt=True, enable_thinking=True)

    g = torch.Generator().manual_seed(1)
    lh, lw = HEIGHT // 8, WIDTH // 8
    latent = torch.randn(1, 16, lh, lw, generator=g)
    context = torch.randn(LENGTH, 64, generator=g)
    valid = 37
    mask = (torch.arange(LENGTH) < valid).to(torch.float32)
    sigma = 0.7317
    out = transformer([latent[0].unsqueeze(1)], torch.tensor([1 - sigma]), [context[:valid]], return_dict=False)[0]
    ref.add("transformer_latent", latent)
    ref.add("transformer_context", context[None])
    ref.add("transformer_context_mask", mask[None])
    ref.add("transformer_timestep", torch.tensor([sigma * 1000]))
    ref.add("transformer_out", -out[0].squeeze(1)[None])

    vae_latent = torch.randn(1, 16, lh, lw, generator=g)
    ref.add("vae_latent", vae_latent)
    ref.add("vae_decoded", vae.decode(vae_latent).sample)
    noise = torch.randn(1, 16, lh, lw, generator=torch.Generator().manual_seed(42))
    ref.add("noise", noise)
    image = pipe(PROMPT, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS, guidance_scale=0.0, latents=noise.clone(),
                 max_sequence_length=LENGTH, output_type="pt").images
    ref.add("image", image[0])
    ref.save({"tokens": tokens, "lengths": lengths, "template": template, "prompt": PROMPT, "steps": STEPS, "height": HEIGHT, "width": WIDTH,
              "length": LENGTH, "timesteps": [float(t) for t in pipe.scheduler.timesteps]})
    print("written", root)


if __name__ == "__main__":
    main()
