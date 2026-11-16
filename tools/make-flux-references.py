#!/usr/bin/env python3
"""Writes diffusers reference outputs of a FLUX.1 pipeline (schnell or dev) for the Sharplight image-generation tests.

For a diffusers FLUX folder (MODEL) this writes raw little-endian f32 tensors plus OUT/index.json:

  tokens / tokens_2                 CLIP and T5 token ids of a few prompts
  text_pooled_{i} / text2_hidden_{i}  CLIP pooled embeddings and T5 hidden states of those prompts
  transformer_*                     one transformer evaluation at SIZE on seeded random latents / context (prediction unpacked to
                                    [1, 16, SIZE/8, SIZE/8]; the timestep input is the model timestep σ·1000)
  vae_latent / vae_decoded          a VAE decode of seeded random latents (raw decoder output in [-1, 1], latents unscaled)
  noise_{size}                      unit Gaussian initial latents [1, 16, size/8, size/8] (seed 42) for the pipeline runs
  image_{size}                      the image ([3, H, W] in [0, 1]) of a txt2img run at each size from noise_{size}
  latents_{size}                    its final latents (model space, unpacked)

Weights are upcast to f32 (--dtype) and everything runs on --device (default cpu: the 12B transformer needs ~48 GB of RAM in f32).
--t5-gguf and --clip-file replace the text encoders by the companion files of a transformer-only checkpoint (same weights as
Sharplight loads them). --transformer-gguf replaces the folder's transformer with a transformer-only GGUF (dequantized on the fly by diffusers; e.g. a
FLUX.1-dev Q8_0 file with the text encoders and VAE of a FLUX.1-schnell folder, which are the same); --dev then also selects the
FLUX.1-dev scheduler (resolution-dependent shift).
Usage: make-flux-references.py MODEL OUT [--sizes 512 1024] [--steps N] [--guidance G] [--device cpu] [--threads 24]
       [--transformer-gguf FILE --dev]
"""

import argparse
import json
import os

import numpy as np
import torch

PROMPTS = ["a photo of a cat", "A painting of a FOX, in the style of Van Gogh!! 123", ""]
PROMPT = "a photograph of an astronaut riding a horse on the moon, detailed, 8k"


class Reference:
    def __init__(self, folder):
        self.folder = folder
        self.index = {}
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "index.json")
        self.extra = {}
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self.index = data.pop("tensors", {})
            self.extra = data

    def add(self, name, tensor):
        array = tensor.detach().to(torch.float32).cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor, np.float32)
        array.astype("<f4").tofile(os.path.join(self.folder, name + ".f32"))
        self.index[name] = list(array.shape)
        self.save({})

    def save(self, extra):
        self.extra.update(extra)
        with open(os.path.join(self.folder, "index.json"), "w") as f:
            json.dump({"tensors": self.index, **self.extra}, f, indent=1)


def load_t5_gguf(model, path, dtype):
    """Loads the dequantized weights of a llama.cpp T5 encoder GGUF (city96's t5-v1_1-xxl-encoder files) into a T5EncoderModel."""
    import gguf
    names = {"attn_q": "layer.0.SelfAttention.q", "attn_k": "layer.0.SelfAttention.k", "attn_v": "layer.0.SelfAttention.v",
             "attn_o": "layer.0.SelfAttention.o", "attn_norm": "layer.0.layer_norm", "ffn_gate": "layer.1.DenseReluDense.wi_0",
             "ffn_up": "layer.1.DenseReluDense.wi_1", "ffn_down": "layer.1.DenseReluDense.wo", "ffn_norm": "layer.1.layer_norm",
             "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias"}
    state = {}
    for t in gguf.GGUFReader(path).tensors:
        a = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape([int(x) for x in reversed(t.shape)])
        if t.name == "token_embd.weight":
            key = "shared.weight"
        elif t.name == "enc.output_norm.weight":
            key = "encoder.final_layer_norm.weight"
        else:
            parts = t.name.split(".")
            key = f"encoder.block.{parts[2]}.{names[parts[3]]}.weight"
        state[key] = torch.from_numpy(a.copy()).to(dtype)
    state["encoder.embed_tokens.weight"] = state["shared.weight"]
    missing = model.load_state_dict(state, strict=False)
    assert not missing.missing_keys, missing.missing_keys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--skip", nargs="*", default=[], help="parts to skip: text transformer vae images")
    parser.add_argument("--transformer-gguf", default=None)
    parser.add_argument("--dev", action="store_true", help="the GGUF transformer is FLUX.1-dev: guidance embedding, dynamic shift")
    parser.add_argument("--t5-gguf", default=None, help="replace the T5 encoder by a T5 encoder GGUF (dequantized; llama.cpp names)")
    parser.add_argument("--clip-file", default=None, help="replace the CLIP text encoder weights by a stand-alone file (clip_l.safetensors)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    from diffusers import FluxPipeline

    dtype = getattr(torch, args.dtype)
    extra = {}
    if args.transformer_gguf:
        import shutil
        import tempfile
        from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel, GGUFQuantizationConfig
        config_dir = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
        os.makedirs(os.path.join(config_dir, "transformer"))
        with open(os.path.join(args.model, "transformer", "config.json")) as f:
            config = json.load(f)
        config["guidance_embeds"] = args.dev
        with open(os.path.join(config_dir, "transformer", "config.json"), "w") as f:
            json.dump(config, f)
        extra["transformer"] = FluxTransformer2DModel.from_single_file(
            args.transformer_gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), torch_dtype=dtype, config=config_dir,
            subfolder="transformer")
        shutil.rmtree(config_dir)
        if args.dev:
            extra["scheduler"] = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True, base_shift=0.5,
                                                                 max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096)
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=dtype, **extra)
    if args.t5_gguf:
        load_t5_gguf(pipe.text_encoder_2, args.t5_gguf, dtype)
    if args.clip_file:
        from safetensors.torch import load_file
        state = {k: v.to(dtype) for k, v in load_file(args.clip_file).items()}
        missing = pipe.text_encoder.load_state_dict(state, strict=False)
        assert not [k for k in missing.missing_keys if "position_ids" not in k], missing.missing_keys
    pipe.to(args.device)
    dev = pipe.transformer.config.guidance_embeds
    steps = args.steps or (28 if dev else 4)
    guidance = args.guidance if args.guidance is not None else (3.5 if dev else 0.0)
    context_length = 512 if dev else 256
    ref = Reference(args.out)
    ref.save({"prompt": PROMPT, "steps": steps, "guidance_scale": guidance, "context_length": context_length})
    device = args.device

    if "text" not in args.skip:
        tokens, tokens_2 = {}, {}
        for i, prompt in enumerate(PROMPTS):
            ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
            ids_2 = pipe.tokenizer_2(prompt, padding="max_length", max_length=context_length, truncation=True, return_tensors="pt").input_ids
            tokens[prompt] = ids[0].tolist()
            tokens_2[prompt] = ids_2[0].tolist()
            ref.add(f"text_pooled_{i}", pipe.text_encoder(ids.to(device), output_hidden_states=False).pooler_output[0])
            ref.add(f"text2_hidden_{i}", pipe.text_encoder_2(ids_2.to(device), output_hidden_states=False)[0][0])
        ref.save({"tokens": tokens, "tokens_2": tokens_2})

    size = args.sizes[0]
    lh = lw = size // 8
    g = torch.Generator().manual_seed(1)
    latent = torch.randn(1, 16, lh, lw, generator=g)
    context = torch.randn(1, context_length, 4096, generator=g)
    pooled = torch.randn(1, 768, generator=g)
    if "transformer" not in args.skip:
        timestep = torch.tensor([0.75])
        img_ids = FluxPipeline._prepare_latent_image_ids(1, lh // 2, lw // 2, device, dtype)
        txt_ids = torch.zeros(context_length, 3, device=device, dtype=dtype)
        packed = FluxPipeline._pack_latents(latent, 1, 16, lh, lw)
        out = pipe.transformer(packed.to(device, dtype), encoder_hidden_states=context.to(device, dtype), pooled_projections=pooled.to(device, dtype),
                               timestep=timestep.to(device, dtype), img_ids=img_ids, txt_ids=txt_ids,
                               guidance=torch.tensor([guidance], device=device, dtype=dtype) if dev else None, return_dict=False)[0]
        ref.add("transformer_latent", latent)
        ref.add("transformer_context", context)
        ref.add("transformer_pooled", pooled)
        ref.add("transformer_timestep", timestep * 1000)
        ref.add("transformer_guidance", torch.tensor([guidance]))
        ref.add("transformer_out", FluxPipeline._unpack_latents(out.float().cpu(), size, size, 8))
        ref.save({"transformer_size": size})

    if "vae" not in args.skip:
        vae_latent = torch.randn(1, 16, lh, lw, generator=g)
        ref.add("vae_latent", vae_latent)
        ref.add("vae_decoded", pipe.vae.decode(vae_latent.to(device, dtype)).sample)
        ref.save({"vae_size": size})

    if "images" not in args.skip:
        for size in args.sizes:
            noise = torch.randn(1, 16, size // 8, size // 8, generator=torch.Generator().manual_seed(42))
            ref.add(f"noise_{size}", noise)
            packed = FluxPipeline._pack_latents(noise, 1, 16, size // 8, size // 8).to(device, dtype)
            result = pipe(PROMPT, height=size, width=size, num_inference_steps=steps, guidance_scale=guidance, latents=packed,
                          max_sequence_length=context_length, output_type="latent").images
            latents = FluxPipeline._unpack_latents(result.float().cpu(), size, size, 8)
            ref.add(f"latents_{size}", latents)
            decoded = pipe.vae.decode((latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor).to(device, dtype)).sample
            ref.add(f"image_{size}", (decoded[0].float() / 2 + 0.5).clamp(0, 1))
            ref.save({"timesteps_" + str(size): [float(t) for t in pipe.scheduler.timesteps], "sizes": args.sizes})
    print("written to", args.out)


if __name__ == "__main__":
    main()
