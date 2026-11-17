#!/usr/bin/env python3
"""Diffusers references of LTX-Video single files (a transformer GGUF or safetensors plus a stand-alone VAE file), for the tests.

The transformer comes from the file (QuantStack / Lightricks, GGUF dequantized on the fly in f32 on CUDA) configured from a diffusers
configuration folder (CONFIG: transformer/config.json, vae/config.json), the VAE from its single file, T5 and the tokenizer from an
LTX-Video diffusers folder (T5_FOLDER). Text-to-video with diffusers' LTXConditionPipeline (no conditions: Lightricks' causal RoPE frame
coordinates), TF32 off, the initial noise fixed. Distilled models (--distilled): the 8-step timesteps of Lightricks' 13B distilled
configuration without guidance. Writes to OUT/reference: text_hidden, text_mask, noise [C, T, h, w], transformer_out (one evaluation on
the noise at the first timestep), video [F, 3, H, W], index.json.

Usage: make-ltx-single-file-references.py TRANSFORMER VAE CONFIG T5_FOLDER OUT [--distilled] [--frames 25] [--height 480] [--width 704]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

PROMPT = "A red fox trotting through fresh snow in a pine forest, morning light, cinematic"
NEGATIVE = "blurry, low quality, distorted"
DISTILLED_TIMESTEPS = [1000, 993, 987, 981, 975, 909, 725, 0.03]


def save(folder, index, name, tensor):
    array = tensor.detach().to(torch.float32).cpu().numpy()
    array.astype("<f4").tofile(os.path.join(folder, name + ".f32"))
    index[name] = list(array.shape)


def patch_vae_conversion():
    """Stand-alone VAE files store the decoder without the "vae." prefix that diffusers' LTX VAE conversion expects."""
    import diffusers.loaders.single_file_model as single_file_model

    entry = single_file_model.SINGLE_FILE_LOADABLE_CLASSES["AutoencoderKLLTXVideo"]
    original = entry["checkpoint_mapping_fn"]

    def mapped(checkpoint, **kwargs):
        prefixed = {(k if k.startswith("vae.") else "vae." + k): v for k, v in checkpoint.items()}
        return original(prefixed, **kwargs)

    entry["checkpoint_mapping_fn"] = mapped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transformer")
    parser.add_argument("vae")
    parser.add_argument("config")
    parser.add_argument("t5")
    parser.add_argument("out")
    parser.add_argument("--distilled", action="store_true")
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import diffusers.pipelines.ltx.pipeline_ltx_condition as condition_module
    from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, LTXConditionPipeline, LTXVideoTransformer3DModel
    from transformers import AutoTokenizer, T5EncoderModel

    patch_vae_conversion()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = os.path.join(args.out, "reference")
    os.makedirs(folder, exist_ok=True)
    index = {}
    quantization = GGUFQuantizationConfig(compute_dtype=torch.float32) if args.transformer.endswith(".gguf") else None
    transformer = LTXVideoTransformer3DModel.from_single_file(args.transformer, quantization_config=quantization, config=os.path.join(args.config, "transformer"),
                                                              torch_dtype=torch.float32)
    vae = AutoencoderKLLTXVideo.from_single_file(args.vae, config=os.path.join(args.config, "vae"), torch_dtype=torch.float32)
    text = T5EncoderModel.from_pretrained(os.path.join(args.t5, "text_encoder"), torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.t5, "tokenizer"))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(args.config, "scheduler"))
    pipe = LTXConditionPipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, transformer=transformer)
    pipe.enable_model_cpu_offload()
    guidance = 1.0 if args.distilled else args.guidance
    embeds, mask, negative, negative_mask = pipe.encode_prompt(PROMPT, NEGATIVE, do_classifier_free_guidance=guidance > 1, max_sequence_length=128)
    save(folder, index, "text_hidden", embeds[0])
    save(folder, index, "text_mask", mask[0])

    latent_frames = (args.frames - 1) // 8 + 1
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(1, 128, latent_frames, args.height // 32, args.width // 32, generator=generator)
    save(folder, index, "noise", noise[0])
    original = condition_module.randn_tensor
    condition_module.randn_tensor = lambda shape, generator=None, device=None, dtype=None, **kwargs: noise.to(device=device, dtype=dtype)

    # One evaluation at the first timestep with the pipeline's causal coordinates.
    t0 = DISTILLED_TIMESTEPS[0] if args.distilled else 1000.0
    ids = pipe._scale_video_ids(pipe._prepare_video_ids(1, latent_frames, args.height // 32, args.width // 32), 32, 8, 0).float()
    ids[:, 0] = ids[:, 0] / 25.0
    transformer.to("cuda")
    out = transformer(pipe._pack_latents(noise).cuda(), embeds.cuda(), torch.tensor([[t0]]).cuda(), mask.cuda(), video_coords=ids.cuda(), return_dict=False)[0]
    save(folder, index, "transformer_out", pipe._unpack_latents(out.cpu(), latent_frames, args.height // 32, args.width // 32)[0])
    transformer.to("cpu")
    torch.cuda.empty_cache()

    captured = {}

    def hook(module, inputs, kwargs, output):
        if "first" not in captured:
            captured["first"] = True
            save(folder, index, "pipeline_transformer_out", pipe._unpack_latents(output[0][:1].float().cpu(), latent_frames, args.height // 32, args.width // 32)[0])
            save(folder, index, "pipeline_timestep", kwargs["timestep"].reshape(-1)[:1].float())

    def on_step(pipeline, step, timestep, tensors):
        captured["latents"] = tensors["latents"]
        if step == 0:
            save(folder, index, "latents_step0", pipe._unpack_latents(tensors["latents"].float().cpu(), latent_frames, args.height // 32, args.width // 32)[0])
        return tensors

    transformer.register_forward_hook(hook, with_kwargs=True)
    start = time.time()
    extra = {"timesteps": DISTILLED_TIMESTEPS} if args.distilled else {"num_inference_steps": args.steps}
    extra["callback_on_step_end"] = on_step
    video = pipe(prompt_embeds=embeds, prompt_attention_mask=mask, negative_prompt_embeds=negative, negative_prompt_attention_mask=negative_mask,
                 height=args.height, width=args.width, num_frames=args.frames, guidance_scale=guidance, generator=generator, output_type="np",
                 **extra).frames[0]
    elapsed = time.time() - start
    condition_module.randn_tensor = original
    save(folder, index, "latents", pipe._unpack_latents(captured["latents"].float().cpu(), latent_frames, args.height // 32, args.width // 32)[0])
    save(folder, index, "video", torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2))
    steps = len(DISTILLED_TIMESTEPS) if args.distilled else args.steps
    with open(os.path.join(folder, "index.json"), "w") as f:
        json.dump({"tensors": index, "prompt": PROMPT, "negative_prompt": NEGATIVE, "frames": args.frames, "height": args.height, "width": args.width,
                   "steps": steps, "guidance_scale": guidance, "transformer_timestep": t0, "seed": args.seed, "seconds": elapsed}, f, indent=1)
    print(f"wrote {args.out} ({elapsed:.1f} s for the pipeline)")


if __name__ == "__main__":
    main()
