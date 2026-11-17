#!/usr/bin/env python3
"""Writes diffusers reference outputs of FLUX.1 Kontext (image editing) for the Sharplight image-generation tests.

MODEL is a FLUX.1 Kontext diffusers folder (configurations, tokenizers, scheduler; with --transformer-gguf, --t5-gguf and --clip-file its
transformer and text encoder weights are replaced by those files, the same ones Sharplight loads as companions). Writes raw little-endian f32
tensors plus OUT/index.json:

  edit_input                         the input image [3, S, S] in [0, 1] (IMAGE, center-cropped to a square and resized to S)
  transformer_*                      one transformer evaluation at S with seeded random latents, reference latents, text features, guidance
  noise_{S} / image_{S}              initial latents (seed 42) and the edited image of FluxKontextPipeline from them (S is one of the
                                     pipeline's preferred resolutions, so it keeps the input's size)

Usage: make-flux-kontext-references.py MODEL OUT IMAGE --transformer-gguf FILE --t5-gguf FILE --clip-file FILE [--size 1024] [--steps 20]
       [--guidance 2.5] [--device cuda]
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import torch

PROMPT = "Turn the horse into a zebra, keep everything else unchanged"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("image")
    parser.add_argument("--transformer-gguf", required=True)
    parser.add_argument("--t5-gguf", required=True)
    parser.add_argument("--clip-file", required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    import PIL.Image
    from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
    from safetensors.torch import load_file

    spec = importlib.util.spec_from_file_location("flux_refs", os.path.join(os.path.dirname(os.path.abspath(__file__)), "make-flux-references.py"))
    flux_refs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flux_refs)

    dtype = torch.float32
    transformer = FluxTransformer2DModel.from_single_file(args.transformer_gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                                                          torch_dtype=dtype, config=args.model, subfolder="transformer")
    pipe = FluxKontextPipeline.from_pretrained(args.model, transformer=transformer, torch_dtype=dtype)
    flux_refs.load_t5_gguf(pipe.text_encoder_2, args.t5_gguf, dtype)
    state = {k.removeprefix("text_model.") if not any(n.startswith("text_model.") for n in pipe.text_encoder.state_dict()) else k: v.to(dtype)
             for k, v in load_file(args.clip_file).items()}
    missing = pipe.text_encoder.load_state_dict(state, strict=False)
    assert not [k for k in missing.missing_keys if "position_ids" not in k], missing.missing_keys
    # The text encoders run on the CPU (the f32 T5 alone takes 19 GB); the transformer and the VAE on --device.
    embeds, pooled_embeds, _ = pipe.encode_prompt(prompt=PROMPT, prompt_2=None, device="cpu", max_sequence_length=512)
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    pipe.transformer.to(args.device)
    pipe.vae.to(args.device)
    device = args.device
    ref = flux_refs.Reference(args.out)
    size = args.size
    ref.save({"prompt": PROMPT, "steps": args.steps, "guidance_scale": args.guidance, "size": size})

    image = PIL.Image.open(args.image).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((size, size), PIL.Image.BICUBIC)
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).to(torch.float32) / 255
    ref.add("edit_input", pixels)

    lh = size // 8
    g = torch.Generator().manual_seed(1)
    latent = torch.randn(1, 16, lh, lh, generator=g)
    reference = torch.randn(1, 16, lh, lh, generator=g)
    context = torch.randn(1, 512, 4096, generator=g)
    pooled = torch.randn(1, 768, generator=g)
    timestep = torch.tensor([0.75])
    ids = FluxKontextPipeline._prepare_latent_image_ids(1, lh // 2, lh // 2, device, dtype)
    reference_ids = ids.clone()
    reference_ids[..., 0] = 1
    packed = torch.cat([FluxKontextPipeline._pack_latents(latent, 1, 16, lh, lh), FluxKontextPipeline._pack_latents(reference, 1, 16, lh, lh)], 1)
    out = pipe.transformer(packed.to(device), encoder_hidden_states=context.to(device), pooled_projections=pooled.to(device), timestep=timestep.to(device),
                           img_ids=torch.cat([ids, reference_ids], 0), txt_ids=torch.zeros(512, 3, device=device),
                           guidance=torch.tensor([args.guidance], device=device), return_dict=False)[0]
    ref.add("transformer_latent", latent)
    ref.add("transformer_reference", reference)
    ref.add("transformer_context", context)
    ref.add("transformer_pooled", pooled)
    ref.add("transformer_timestep", timestep * 1000)
    ref.add("transformer_guidance", torch.tensor([args.guidance]))
    ref.add("transformer_out", FluxKontextPipeline._unpack_latents(out[:, : (lh // 2) ** 2].float().cpu(), size, size, 8))
    del out
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    noise = torch.randn(1, 16, lh, lh, generator=torch.Generator().manual_seed(42))
    ref.add(f"noise_{size}", noise)
    result = pipe(image=image, prompt_embeds=embeds.to(device), pooled_prompt_embeds=pooled_embeds.to(device), height=size, width=size, num_inference_steps=args.steps, guidance_scale=args.guidance,
                  latents=FluxKontextPipeline._pack_latents(noise, 1, 16, lh, lh).to(device), max_sequence_length=512, output_type="pt").images
    ref.add(f"image_{size}", result[0])
    ref.save({"timesteps": [float(t) for t in pipe.scheduler.timesteps]})
    print("written to", args.out)


if __name__ == "__main__":
    main()
