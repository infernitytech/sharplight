#!/usr/bin/env python3
"""Creates tiny random Stable Diffusion 1.x-shaped models and diffusers reference outputs for the image-generation tests.

The models keep the SD 1.5 layout (four UNet levels with cross-attention in the first three, a four-level VAE, a CLIP text
encoder with the real CLIP vocabulary) with very small widths, so the Sharplight importers see the same tensor names as in
real checkpoints. Written to OUT (default ~/models/sd/tiny-random-sd):

  diffusers/                 diffusers folder (model_index.json, unet/, vae/, text_encoder/, tokenizer/, scheduler/)
  tiny-sd.safetensors        the same weights as a single-file checkpoint with the original (LDM) tensor names
  tiny-sd.ckpt               the same single-file checkpoint as a PyTorch (pickle) file
  reference/*.f32            diffusers reference outputs (tools/make-diffusion-references.py, 64x64, 6 steps, run on the new folder),
                             described by reference/index.json

Weights and references are written together into OUT.staging and replace OUT as a whole at the end, so they cannot diverge (a run that
fails leaves the previous set as it was).

Requires torch, diffusers, transformers and safetensors, plus diffusers' scripts/convert_diffusers_to_original_stable_diffusion.py
(pass --convert-script, or it is downloaded). The CLIP tokenizer files are taken from --tokenizer (a folder with vocab.json and
merges.txt) or downloaded from the Hugging Face hub.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.request

import torch

CONVERT_URL = "https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_diffusers_to_original_stable_diffusion.py"


def load_convert_script(path, scratch):
    if path is None:
        path = os.path.join(scratch, "convert_diffusers_to_original_stable_diffusion.py")
        if not os.path.exists(path):
            urllib.request.urlretrieve(CONVERT_URL, path)
    spec = importlib.util.spec_from_file_location("convert_sd", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_models(family):
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel

    torch.manual_seed(0)
    # SD 2.x: linear transformer projections, a head count per level (diffusers' "attention_head_dim"), upcast attention,
    # an OpenCLIP-style GELU text encoder.
    sd2 = family == "sd2"
    unet = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(32, 32, 64, 64),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32, attention_head_dim=(2, 2, 4, 4) if sd2 else 4, norm_num_groups=8,
        use_linear_projection=sd2, upcast_attention=sd2)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, layers_per_block=2, norm_num_groups=8,
        block_out_channels=(16, 16, 32, 32),
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4)
    text = CLIPTextModel(CLIPTextConfig(
        vocab_size=49408, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=77, hidden_act="gelu" if sd2 else "quick_gelu", bos_token_id=49406, eos_token_id=49407, pad_token_id=49407))
    # Random init leaves norms at identity and some biases at zero; perturb everything so every weight matters.
    with torch.no_grad():
        for model in (unet, vae, text):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
    return unet.eval(), vae.eval(), text.eval()


def to_open_clip(state, layers):
    """transformers CLIPTextModel names -> the OpenCLIP text tower of original SD 2.x checkpoints (fused q/k/v), plus one random
    trailing block: SD 2.x conditions on the penultimate layer, so importers must drop the last one."""
    state = {k.removeprefix("text_model."): v for k, v in state.items()}   # newer transformers drop the prefix
    out = {
        "token_embedding.weight": state["embeddings.token_embedding.weight"],
        "positional_embedding": state["embeddings.position_embedding.weight"],
        "ln_final.weight": state["final_layer_norm.weight"],
        "ln_final.bias": state["final_layer_norm.bias"],
    }
    parts = {"layer_norm1": "ln_1", "layer_norm2": "ln_2", "self_attn.out_proj": "attn.out_proj", "mlp.fc1": "mlp.c_fc", "mlp.fc2": "mlp.c_proj"}
    for i in range(layers):
        src = f"encoder.layers.{i}."
        dst = f"transformer.resblocks.{i}."
        for kind in ("weight", "bias"):
            out[dst + "attn.in_proj_" + kind] = torch.cat([state[src + f"self_attn.{p}_proj.{kind}"] for p in "qkv"])
            for hf, oc in parts.items():
                out[dst + oc + "." + kind] = state[src + hf + "." + kind]
    last = f"transformer.resblocks.{layers - 1}."
    for k, v in list(out.items()):
        if k.startswith(last):
            out[k.replace(last, f"transformer.resblocks.{layers}.", 1)] = v + torch.randn_like(v) * 0.05
    return out



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=["sd1", "sd2"], default="sd1", help="sd2: SD 2.x layout, v-prediction")
    parser.add_argument("--out", default=None, help="default ~/models/sd/tiny-random-sd (sd1) or tiny-random-sd2 (sd2)")
    parser.add_argument("--tokenizer", default=None, help="folder with the CLIP vocab.json and merges.txt")
    parser.add_argument("--convert-script", default=None)
    args = parser.parse_args()

    from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                           PNDMScheduler, StableDiffusionPipeline)
    from transformers import CLIPTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sd2 = args.family == "sd2"
    final = os.path.abspath(args.out or os.path.expanduser("~/models/sd/tiny-random-sd2" if sd2 else "~/models/sd/tiny-random-sd"))
    # Everything is written to a staging folder first and moved into place at the end.
    out = final + ".staging"
    if os.path.exists(out):
        shutil.rmtree(out)
    diffusers_dir = os.path.join(out, "diffusers")
    os.makedirs(out)
    unet, vae, text = build_models(args.family)
    tokenizer = (CLIPTokenizer.from_pretrained(args.tokenizer) if args.tokenizer
                 else CLIPTokenizer.from_pretrained("sd2-community/stable-diffusion-2-1" if sd2 else "stable-diffusion-v1-5/stable-diffusion-v1-5",
                                                    subfolder="tokenizer"))
    if sd2:
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
                                  clip_sample=False, set_alpha_to_one=False, steps_offset=1, prediction_type="v_prediction")
    else:
        scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
                                  set_alpha_to_one=False, skip_prk_steps=True, steps_offset=1)
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=text, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
                                   safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.save_pretrained(diffusers_dir, safe_serialization=True)

    sys.dont_write_bytecode = True   # no __pycache__ of the downloaded convert script in the model folder
    convert = load_convert_script(args.convert_script, out)
    single_file = "tiny-sd2" if sd2 else "tiny-sd"
    state = {}
    state.update({"model.diffusion_model." + k: v for k, v in convert.convert_unet_state_dict(unet.state_dict()).items()})
    state.update({"first_stage_model." + k: v for k, v in convert.convert_vae_state_dict(vae.state_dict()).items()})
    if sd2:
        # OpenCLIP layout (fused q/k/v); the real SD 2.x encoder has one more layer, which the diffusers export drops.
        text_state = {k: v for k, v in text.state_dict().items() if not k.endswith("position_ids")}
        open_clip = to_open_clip(text_state, text.config.num_hidden_layers)
        state.update({"cond_stage_model.model." + k: v for k, v in open_clip.items()})
    else:
        state.update({"cond_stage_model.transformer." + k: v for k, v in text.state_dict().items()
                      if not k.endswith("position_ids")})
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in state.items()}, os.path.join(out, single_file + ".safetensors"))
    # The same weights as a Lightning-style PyTorch checkpoint (state dict nested under "state_dict").
    torch.save({"state_dict": {k: v.contiguous() for k, v in state.items()}, "global_step": 0, "epoch": 1}, os.path.join(out, single_file + ".ckpt"))
    script_copy = os.path.join(out, "convert_diffusers_to_original_stable_diffusion.py")
    if os.path.exists(script_copy):
        os.remove(script_copy)

    # The references come from tools/make-diffusion-references.py on the folder just written, in the same run: weights and references
    # are produced together and replace the previous set only as a whole (a failed run leaves the old set untouched).
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make-diffusion-references.py")
    subprocess.run([sys.executable, script, diffusers_dir, os.path.join(out, "reference"), "--size", "64", "--steps", "6"], check=True)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.replace(out, final)
    print("written to", final)


if __name__ == "__main__":
    main()
