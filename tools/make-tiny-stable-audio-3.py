#!/usr/bin/env python3
"""Creates two tiny random Stable Audio 3 checkpoints (the stable-audio-3 folder layout) and reference outputs of the stable-audio-3 package for the audio-generation tests.

Writes OUT (default ~/models/audio-models/tiny-random-stable-audio-3):

  small/      the layout of stable-audio-3-small-*: model_config.json, model.safetensors (a 2-layer diffusion transformer of width 64 with 4 memory rows,
              the SAME autoencoder with a 2-block chunked decoder with a midpoint shift, 8 latent channels, 16 samples per frame at 512 Hz) and t5gemma-b-b-ul2/
              (a 2-layer T5Gemma of width 32 with Gemma's real 256k-token tokenizer)
  medium/     the same with the options of stable-audio-3-medium-*: differential attention in the diffusion transformer, a sliding-window decoder whose
              last block has a sine-gated feed-forward layer
  reference/  raw little-endian f32 tensors of generations with seed 42 and a duration of 0.5 s (16 latent frames of the 208 the package generates):
              <variant>_<case>_noise ([frames, channels], the initial standard normal draw), <case>_step_noise ([steps, frames, channels], ping-pong only),
              <case>_latent ([frames, channels]) and <case>_wav ([samples, channels], decoded without the autoencoder's dither, clamped and cut to the duration);
              cases "euler" (6 Euler steps, guidance 3), "negative" (the same with a negative prompt) and "pingpong" (4 ping-pong steps without guidance)

The tokenizer comes from --tokenizer (the t5gemma-b-b-ul2 folder of a Stable Audio 3 checkpoint, or google/t5gemma-b-b-ul2 on the hub).
Requires the stable-audio-3 package (github.com/Stability-AI/stable-audio-3) and its dependencies.
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch

PROMPT = "a dog barking twice in a quiet room"
NEGATIVE = "music, noise"
SECONDS = 0.5
SEED = 42
TEXT = "t5gemma-b-b-ul2"


def model_config(medium):
    same = {
        "channels": 16, "c_mults": [2], "strides": [4], "latent_dim": 8, "transformer_depths": [2], "checkpointing": False, "differential": True,
        "dim_heads": 16, "variable_stride": True, "dyt": True, "chunk_size": 8, "chunk_midpoint_shift": not medium,
    }
    if medium:
        same["sliding_window"] = [1, 1]
    decoder = dict(same, out_channels=8, conv_mapping=True, mask_noise=0.1 if medium else 0.01, sinusoidal_blocks=[2 if medium else 0])
    encoder = dict(same, in_channels=8, mask_noise=0.0)
    return {
        "model_type": "diffusion_cond_inpaint",
        "sample_size": 4096,
        "sample_rate": 512,
        "audio_channels": 2,
        "model": {
            "pretransform": {
                "type": "autoencoder", "iterate_batch": False, "chunked": True, "enable_grad": False,
                "config": {
                    "pretransform": {"type": "patched", "enable_grad": False, "config": {"patch_size": 4, "channels": 2}},
                    "encoder": {"type": "taae_v2", "requires_grad": False, "config": encoder},
                    "decoder": {"type": "taae_v2", "requires_grad": False, "config": decoder},
                    "bottleneck": {"type": "softnorm", "requires_grad": False,
                                   "config": {"dim": 8, "noise_augment_dim": 0, "noise_regularize": True, "auto_scale": True, "freeze": True}},
                    "latent_dim": 8, "downsampling_ratio": 16, "io_channels": 2,
                },
            },
            "conditioning": {
                "configs": [
                    {"id": "prompt", "type": "t5gemma", "config": {"max_length": 24, "padding_mode": "learned", "subfolder": TEXT}},
                    {"id": "seconds_total", "type": "number", "config": {"min_val": 0, "max_val": 384, "fourier_features_type": "expo"}},
                ],
                "cond_dim": 32,
            },
            "diffusion": {
                "cross_attention_cond_ids": ["prompt", "seconds_total"],
                "global_cond_ids": ["seconds_total"],
                "local_add_cond_ids": ["inpaint_mask", "inpaint_masked_input"],
                "type": "dit",
                "diffusion_objective": "rectified_flow",
                "use_effective_length_for_schedule": True,
                "mask_padding_attention": True,
                "config": {
                    "io_channels": 8, "embed_dim": 64, "depth": 2, "num_heads": 2, "cond_token_dim": 32, "global_cond_dim": 32, "local_add_cond_dim": 9,
                    "global_cond_type": "adaLN", "timestep_features_type": "expo", "timestep_features_logsnr": False,
                    "attn_kwargs": {"qk_norm": "rms", "differential": medium}, "norm_type": "rms_norm", "norm_kwargs": {"force_fp32": True},
                    "ff_kwargs": {"mult": 2.0}, "num_memory_tokens": 4,
                },
            },
            "io_channels": 8,
        },
    }


def build_text_encoder(folder, tokenizer):
    from transformers import T5GemmaConfig, T5GemmaForConditionalGeneration, T5GemmaModuleConfig

    module = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=16, vocab_size=256000,
                  query_pre_attn_scalar=16, attn_logit_softcapping=50.0, final_logit_softcapping=30.0, sliding_window=4096, rms_norm_eps=1e-6,
                  hidden_activation="gelu_pytorch_tanh", layer_types=["sliding_attention", "full_attention"])
    config = T5GemmaConfig(encoder=T5GemmaModuleConfig(**module), decoder=T5GemmaModuleConfig(**module, cross_attention_hidden_size=32, is_decoder=True),
                           pad_token_id=0, eos_token_id=[1, 107])
    torch.manual_seed(1)
    model = T5GemmaForConditionalGeneration(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(torch.randn_like(p) * 0.2)
    model.to(torch.bfloat16).save_pretrained(folder)
    for f in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]:
        if os.path.exists(os.path.join(tokenizer, f)):
            shutil.copy(os.path.join(tokenizer, f), folder)


def build(out, variant, tokenizer):
    from safetensors.torch import save_file
    from stable_audio_3.factory import create_diffusion_cond_from_config
    from stable_audio_3.models import conditioners

    medium = variant == "medium"
    folder = os.path.join(out, variant)
    os.makedirs(os.path.join(folder, TEXT), exist_ok=True)
    build_text_encoder(os.path.join(folder, TEXT), tokenizer)
    config = model_config(medium)
    with open(os.path.join(folder, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    conditioners.T5GemmaConditioner.T5GEMMA_MODEL_DIMS["google/t5gemma-b-b-ul2"] = 32
    config["model"]["conditioning"]["configs"][0]["config"]["model_path"] = folder
    torch.manual_seed(2 if medium else 0)
    model = create_diffusion_cond_from_config(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("gamma", "weight_g")) or "norm" in name:
                p.copy_(1 + torch.randn_like(p) * 0.1)
            elif name.endswith("alpha"):
                p.copy_(torch.full_like(p, 0.7))
            elif name.endswith("running_std"):
                p.copy_(torch.full_like(p, 1.3))
            else:
                p.copy_(torch.randn_like(p) * (0.5 / max(p.shape[-1], 1) ** 0.5 if p.dim() > 1 else 0.05))
    state = {k: v.contiguous() for k, v in model.state_dict().items() if not k.endswith("inv_freq")}
    save_file(state, os.path.join(folder, "model.safetensors"))
    return folder


def reference(out, variant):
    from stable_audio_3 import model as sa3_model
    from stable_audio_3.loading_utils import load_diffusion_cond
    from stable_audio_3.models import bottleneck, conditioners

    folder = os.path.join(out, variant)
    refs = os.path.join(out, "reference")
    os.makedirs(refs, exist_ok=True)
    conditioners.T5GemmaConditioner.T5GEMMA_MODEL_DIMS["google/t5gemma-b-b-ul2"] = 32
    with open(os.path.join(folder, "model_config.json")) as f:
        config = json.load(f)
    config["model"]["conditioning"]["configs"][0]["config"]["model_path"] = folder
    from transformers import T5GemmaEncoderModel

    original = T5GemmaEncoderModel.from_pretrained.__func__
    T5GemmaEncoderModel.from_pretrained = classmethod(lambda cls, *a, **k: original(cls, *a, **dict(k, dtype=torch.float32)))
    model = load_diffusion_cond(config, os.path.join(folder, "model.safetensors"), device="cpu", model_half=False)
    model.use_lora = False
    model.lora_names = []
    sam = sa3_model.StableAudioModel(model, config, "cpu", False)
    for m in model.pretransform.modules():
        if isinstance(m, bottleneck.SoftNormBottleneck):
            m.noise_regularize = False
        if hasattr(m, "mask_noise"):
            m.mask_noise = 0.0

    for case, steps, cfg, sampler, negative in [("euler", 6, 3.0, "euler", None), ("negative", 6, 3.0, "euler", NEGATIVE), ("pingpong", 4, 1.0, "pingpong", None)]:
        draws, step_noise = [], []
        randn, randn_like = torch.randn, torch.randn_like

        def recording_randn(*a, **k):
            r = randn(*a, **k)
            draws.append(r[0].T.clone())
            return r

        def recording_randn_like(t, *a, **k):
            r = randn_like(t, *a, **k)
            step_noise.append(r[0].T.clone())
            return r

        torch.randn, torch.randn_like = recording_randn, recording_randn_like
        try:
            latents = sam.generate(prompt=PROMPT, negative_prompt=negative, duration=SECONDS, steps=steps, cfg_scale=cfg, seed=SEED, sampler_type=sampler,
                                   return_latents=True)
        finally:
            torch.randn, torch.randn_like = randn, randn_like
        with torch.no_grad():
            audio = model.pretransform.decode(latents.float(), chunked=None).clamp(-1, 1)[0, :, : int(SECONDS * 512)].T
        tensors = {"noise": draws[0], "latent": latents[0].T, "wav": audio}
        if sampler == "pingpong":
            tensors["step_noise"] = torch.stack(step_noise)
        for name, t in tensors.items():
            np.ascontiguousarray(t.detach().float().numpy().astype("<f4")).tofile(os.path.join(refs, f"{variant}_{case}_{name}.f32"))
        print(variant, case, {k: tuple(v.shape) for k, v in tensors.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/audio-models/tiny-random-stable-audio-3"))
    parser.add_argument("--tokenizer", default="google/t5gemma-b-b-ul2")
    args = parser.parse_args()
    torch.set_num_threads(4)
    for variant in ["small", "medium"]:
        build(args.out, variant, args.tokenizer)
        reference(args.out, variant)


if __name__ == "__main__":
    main()
