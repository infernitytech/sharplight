#!/usr/bin/env python3
"""Creates tiny random Stable Diffusion 1.x-shaped models and diffusers reference outputs for the image-generation tests.

The models keep the SD 1.5 layout (four UNet levels with cross-attention in the first three, a four-level VAE, a CLIP text
encoder with the real CLIP vocabulary) with very small widths, so the Sharplight importers see the same tensor names as in
real checkpoints. Written to OUT (default ~/models/sd/tiny-random-sd):

  diffusers/                 diffusers folder (model_index.json, unet/, vae/, text_encoder/, tokenizer/, scheduler/)
  tiny-sd.safetensors        the same weights as a single-file checkpoint with the original (LDM) tensor names
  reference/*.f32            raw little-endian f32 reference tensors, described by reference/index.json

Requires torch, diffusers, transformers and safetensors, plus diffusers' scripts/convert_diffusers_to_original_stable_diffusion.py
(pass --convert-script, or it is downloaded). The CLIP tokenizer files are taken from --tokenizer (a folder with vocab.json and
merges.txt) or downloaded from the Hugging Face hub.
"""

import argparse
import importlib.util
import json
import os
import shutil
import urllib.request

import numpy as np
import torch

CONVERT_URL = "https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_diffusers_to_original_stable_diffusion.py"
PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123", ""]


def load_convert_script(path, scratch):
    if path is None:
        path = os.path.join(scratch, "convert_diffusers_to_original_stable_diffusion.py")
        if not os.path.exists(path):
            urllib.request.urlretrieve(CONVERT_URL, path)
    spec = importlib.util.spec_from_file_location("convert_sd", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_models():
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(32, 32, 64, 64),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32, attention_head_dim=4, norm_num_groups=8)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, layers_per_block=2, norm_num_groups=8,
        block_out_channels=(16, 16, 32, 32),
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4)
    text = CLIPTextModel(CLIPTextConfig(
        vocab_size=49408, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=77, hidden_act="quick_gelu", bos_token_id=49406, eos_token_id=49407, pad_token_id=49407))
    # Random init leaves norms at identity and some biases at zero; perturb everything so every weight matters.
    with torch.no_grad():
        for model in (unet, vae, text):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
    return unet.eval(), vae.eval(), text.eval()


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
    parser.add_argument("--out", default=os.path.expanduser("~/models/sd/tiny-random-sd"))
    parser.add_argument("--tokenizer", default=None, help="folder with the CLIP vocab.json and merges.txt")
    parser.add_argument("--convert-script", default=None)
    args = parser.parse_args()

    from diffusers import (DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                           PNDMScheduler, StableDiffusionPipeline)
    from transformers import CLIPTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.out
    diffusers_dir = os.path.join(out, "diffusers")
    os.makedirs(out, exist_ok=True)
    unet, vae, text = build_models()
    tokenizer = (CLIPTokenizer.from_pretrained(args.tokenizer) if args.tokenizer
                 else CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer"))
    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
                              set_alpha_to_one=False, skip_prk_steps=True, steps_offset=1)
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=text, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
                                   safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.save_pretrained(diffusers_dir, safe_serialization=True)

    convert = load_convert_script(args.convert_script, out)
    state = {}
    state.update({"model.diffusion_model." + k: v for k, v in convert.convert_unet_state_dict(unet.state_dict()).items()})
    state.update({"first_stage_model." + k: v for k, v in convert.convert_vae_state_dict(vae.state_dict()).items()})
    state.update({"cond_stage_model.transformer." + k: v for k, v in text.state_dict().items()
                  if not k.endswith("position_ids")})
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in state.items()}, os.path.join(out, "tiny-sd.safetensors"))
    script_copy = os.path.join(out, "convert_diffusers_to_original_stable_diffusion.py")
    if os.path.exists(script_copy):
        os.remove(script_copy)

    ref = Reference(os.path.join(out, "reference"))
    tokens = {}
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
            tokens[prompt] = ids[0].tolist()
            ref.add(f"text_hidden_{i}", text(ids).last_hidden_state[0])

        g = torch.Generator().manual_seed(1)
        latent = torch.randn(2, 4, 8, 8, generator=g)
        context = torch.randn(2, 77, 32, generator=g)
        timesteps = torch.tensor([999.0, 321.5])
        ref.add("unet_latent", latent)
        ref.add("unet_context", context)
        ref.add("unet_timesteps", timesteps)
        ref.add("unet_out", unet(latent, timesteps, encoder_hidden_states=context).sample)

        vae_latent = torch.randn(1, 4, 8, 8, generator=g)
        ref.add("vae_latent", vae_latent)
        ref.add("vae_decoded", vae.decode(vae_latent).sample)
        image = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
        ref.add("vae_image", image)
        ref.add("vae_encoded_mean", vae.encode(image).latent_dist.mean)

    samplers = {
        "euler": EulerDiscreteScheduler.from_config(scheduler.config),
        "euler_a": EulerAncestralDiscreteScheduler.from_config(scheduler.config),
        "dpmpp2m_karras": DPMSolverMultistepScheduler.from_config(scheduler.config, use_karras_sigmas=True,
                                                                  algorithm_type="dpmsolver++", solver_order=2),
    }
    runs = {}
    for name, sampler in samplers.items():
        pipe.scheduler = sampler
        steps = []

        def record(p, step, timestep, kwargs):
            steps.append(kwargs["latents"].clone())
            return kwargs

        image = pipe(PROMPTS[0], negative_prompt="blurry", height=64, width=64, num_inference_steps=4, guidance_scale=7.5,
                     generator=torch.Generator().manual_seed(42), output_type="pt", callback_on_step_end=record).images
        for i, latents in enumerate(steps):
            ref.add(f"{name}_step{i}", latents)
        ref.add(f"{name}_image", image)
        sigmas = sampler.sigmas.tolist()
        runs[name] = {"sigmas": sigmas, "timesteps": [float(t) for t in sampler.timesteps]}

    ref.save({"tokens": tokens, "prompt": PROMPTS[0], "negative_prompt": "blurry", "seed": 42, "steps": 4,
              "guidance_scale": 7.5, "height": 64, "width": 64, "samplers": runs})
    print("written to", out)


if __name__ == "__main__":
    main()
