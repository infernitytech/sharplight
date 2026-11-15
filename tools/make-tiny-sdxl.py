#!/usr/bin/env python3
"""Creates a tiny random SDXL-shaped pipeline and diffusers reference outputs for the image-generation tests.

The model keeps the SDXL layout with very small widths: two CLIP text encoders (the second with a text projection, both read at
the penultimate layer without the final norm), a UNet whose first level has no attention, transformer depths per level, and the
"text_time" added conditioning (pooled text embedding + sinusoidal size / crop embeddings), and a four-level VAE. Written to OUT
(default ~/models/sd/tiny-random-sdxl):

  diffusers/                 diffusers folder (model_index.json, unet/, vae/, text_encoder/, text_encoder_2/, tokenizer(_2)/, scheduler/)
  reference/*.f32            raw little-endian f32 reference tensors, described by reference/index.json

Requires torch, diffusers, transformers and safetensors. The CLIP tokenizer files come from --tokenizer (a folder with vocab.json
and merges.txt) or the Hugging Face hub.
"""

import argparse
import json
import os

import numpy as np
import torch

PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123", ""]
PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"
NEGATIVE = "blurry, low quality"


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
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=2, block_out_channels=(32, 32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        transformer_layers_per_block=(1, 1, 2), attention_head_dim=(2, 2, 4), cross_attention_dim=64, norm_num_groups=8,
        use_linear_projection=True, addition_embed_type="text_time", addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=32 + 6 * 8)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, layers_per_block=2, norm_num_groups=8, block_out_channels=(16, 16, 32, 32),
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4, scaling_factor=0.13025)
    common = dict(vocab_size=49408, hidden_size=32, intermediate_size=64, num_attention_heads=4, max_position_embeddings=77,
                  bos_token_id=49406, eos_token_id=49407)
    text = CLIPTextModel(CLIPTextConfig(num_hidden_layers=3, hidden_act="quick_gelu", pad_token_id=49407, **common))
    text_2 = CLIPTextModelWithProjection(CLIPTextConfig(num_hidden_layers=3, hidden_act="gelu", projection_dim=32, pad_token_id=0, **common))
    with torch.no_grad():
        for model in (unet, vae, text, text_2):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
    return unet.eval(), vae.eval(), text.eval(), text_2.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/sd/tiny-random-sdxl"))
    parser.add_argument("--tokenizer", default=None, help="folder with the CLIP vocab.json and merges.txt")
    args = parser.parse_args()

    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, StableDiffusionXLPipeline
    from transformers import CLIPTokenizer

    out = args.out
    unet, vae, text, text_2 = build_models()
    source = args.tokenizer or "stable-diffusion-v1-5/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(source, subfolder=None if args.tokenizer else "tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(source, subfolder=None if args.tokenizer else "tokenizer", pad_token="!")
    scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
                                       steps_offset=1, timestep_spacing="leading", interpolation_type="linear")
    pipe = StableDiffusionXLPipeline(vae=vae, text_encoder=text, text_encoder_2=text_2, tokenizer=tokenizer, tokenizer_2=tokenizer_2,
                                     unet=unet, scheduler=scheduler, force_zeros_for_empty_prompt=True)
    pipe.save_pretrained(os.path.join(out, "diffusers"), safe_serialization=True)


    ref = Reference(os.path.join(out, "reference"))
    tokens = {}
    tokens_2 = {}
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
            ids_2 = tokenizer_2(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
            tokens[prompt] = ids[0].tolist()
            tokens_2[prompt] = ids_2[0].tolist()
            ref.add(f"text_hidden_{i}", text(ids, output_hidden_states=True).hidden_states[-2][0])
            out_2 = text_2(ids_2, output_hidden_states=True)
            ref.add(f"text2_hidden_{i}", out_2.hidden_states[-2][0])
            ref.add(f"text2_pooled_{i}", out_2.text_embeds[0])

        g = torch.Generator().manual_seed(1)
        latent = torch.randn(2, 4, 8, 8, generator=g)
        context = torch.randn(2, 77, 64, generator=g)
        pooled = torch.randn(2, 32, generator=g)
        time_ids = torch.tensor([[64.0, 64.0, 0.0, 0.0, 64.0, 64.0], [96.0, 80.0, 8.0, 4.0, 64.0, 64.0]])
        timesteps = torch.tensor([999.0, 321.5])
        ref.add("unet_latent", latent)
        ref.add("unet_context", context)
        ref.add("unet_pooled", pooled)
        ref.add("unet_time_ids", time_ids)
        ref.add("unet_timesteps", timesteps)
        ref.add("unet_out", unet(latent, timesteps, encoder_hidden_states=context,
                                 added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample)

        vae_latent = torch.randn(1, 4, 8, 8, generator=g)
        ref.add("vae_latent", vae_latent)
        ref.add("vae_decoded", vae.decode(vae_latent).sample)
        image = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
        ref.add("vae_image", image)
        ref.add("vae_encoded_mean", vae.encode(image).latent_dist.mean)
        noise = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(42))
        ref.add("noise", noise)

    config = pipe.scheduler.config
    samplers = {
        "euler": EulerDiscreteScheduler.from_config(config),
        "ddim": DDIMScheduler.from_config(config),
        "dpm++2m-karras": DPMSolverMultistepScheduler.from_config(config, algorithm_type="dpmsolver++", solver_order=2,
                                                                  use_karras_sigmas=True),
    }
    runs = {}
    for name, sampler in samplers.items():
        pipe.scheduler = sampler
        for negative, suffix in ((NEGATIVE, ""), (None, "_noneg")):
            with torch.no_grad():
                image = pipe(PROMPT, negative_prompt=negative, height=64, width=64, num_inference_steps=6, guidance_scale=7.5,
                             latents=noise.clone(), output_type="pt").images
            ref.add(f"{name}{suffix}_image", image[0])
        runs[name] = {"timesteps": [float(t) for t in sampler.timesteps]}

    ref.save({"tokens": tokens, "tokens_2": tokens_2, "prompt": PROMPT, "negative_prompt": NEGATIVE, "steps": 6, "guidance_scale": 7.5,
              "height": 64, "width": 64, "samplers": runs})
    print("written to", out)


if __name__ == "__main__":
    main()
