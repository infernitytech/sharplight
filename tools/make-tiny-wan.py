#!/usr/bin/env python3
"""Creates tiny random Wan-shaped video pipelines and diffusers reference outputs for the video-generation tests.

Variants (written to OUT/{variant}, default OUT ~/models/video/tiny-random-wan):

  t2v    Wan 2.1 layout: WanTransformer3DModel (3-D patches, (t, h, w) RoPE, cross-attention to umT5 tokens), the umT5 encoder,
         the 3-D causal VAE (AutoencoderKLWan, 16-channel style with z 4 here), text-to-video
  a14b   Wan 2.2 A14B layout: the t2v layout with two experts (transformer for the high-noise timesteps, transformer_2 below the
         boundary 0.875 · 1000, guidance 5 and 3)
  i2v    Wan 2.2 I2V-A14B layout: two experts that take the noisy latents concatenated with a first-frame mask and the encoded
         video of the start image followed by blank frames (4 + 4 + 4 channels), image-to-video only
  clip   Wan 2.1 I2V layout: one transformer over the concatenated channels (as i2v) that also attends to the penultimate hidden
         states of a CLIP vision encoder (image_dim, added k/v projections), image-to-video only
  flf2v  Wan 2.1 FLF2V layout: the clip layout conditioned on a first and a last frame (both images' CLIP states with a learned
         position table, the mask 1 on the first and the last frame)
  ti2v   Wan 2.2 TI2V layout: the same transformer with per-token timesteps (expand_timesteps), the residual VAE with 2×2
         patchification (is_residual, patch_size 2), text-to-video and image-to-video (the first latent frame from the image)

Each holds:

  diffusers/           diffusers folder (model_index.json, transformer/, vae/, text_encoder/, tokenizer/, scheduler/)
  single/wan.safetensors   the transformer under the original Wan tensor names (f32), as ComfyUI / the Wan repository store it
  single/wan-F16.gguf      the same as a transformer-only GGUF (city96 layout, F16: the tiny widths are not Q8_0 block multiples)
  reference/*.f32      raw little-endian f32 reference tensors, described by reference/index.json

The umT5 tokenizer comes from --tokenizer (a Wan diffusers folder's tokenizer/ directory) or google/umt5-xxl on the hub; the
encoder keeps the real 256384-token vocabulary so the tokenizer's ids are valid.
Requires torch, diffusers, transformers and safetensors.
"""

import argparse
import json
import os

import numpy as np
import torch

PROMPTS = ["a cat walking on grass", "Ein Hund, der im Schnee spielt!! 123", ""]
PROMPT = "a red panda eating bamboo in a forest, cinematic"
NEGATIVE = "blurry, low quality"
SIZES = {"t2v": (32, 48), "ti2v": (64, 96), "a14b": (32, 48), "i2v": (32, 48), "clip": (32, 48), "flf2v": (32, 48)}  # (height, width): the TI2V VAE compresses 16× spatially
FRAMES = 9
STEPS = 4
GUIDANCE = 5.0
GUIDANCE_2 = 3.0
MAX_SEQUENCE = 512
VARIANTS = ["t2v", "ti2v", "a14b", "i2v", "clip", "flf2v"]
CLIP_SIZE = 28  # 2×2 patches of 14 + the class token = 5 image tokens


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


def build_models(variant):
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from transformers import UMT5Config, UMT5EncoderModel

    ti2v = variant == "ti2v"
    torch.manual_seed(2 if ti2v else 3)
    z = 8 if ti2v else 4
    clip = variant in ("clip", "flf2v")
    image_tokens = (CLIP_SIZE // 14) ** 2 + 1
    transformer = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=3 * z if variant in ("i2v", "clip", "flf2v") else z, out_channels=z,
        text_dim=32, freq_dim=256, ffn_dim=32, num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6, rope_max_seq_len=32,
        image_dim=16 if clip else None, added_kv_proj_dim=24 if clip else None, pos_embed_seq_len=2 * image_tokens if variant == "flf2v" else None)
    generator = torch.Generator().manual_seed(5)
    vae = AutoencoderKLWan(
        base_dim=8, decoder_base_dim=12 if ti2v else None, z_dim=z, dim_mult=[1, 2, 4, 4], num_res_blocks=1, attn_scales=[],
        temperal_downsample=[False, True, True], latents_mean=(torch.randn(z, generator=generator) * 0.5).tolist(),
        latents_std=(torch.rand(z, generator=generator) + 0.5).tolist(), is_residual=ti2v, in_channels=12 if ti2v else 3,
        out_channels=12 if ti2v else 3, patch_size=2 if ti2v else None, scale_factor_spatial=16 if ti2v else 8)
    text = UMT5EncoderModel(UMT5Config(
        vocab_size=256384, d_model=32, d_kv=4, d_ff=64, num_layers=2, num_heads=8, feed_forward_proj="gated-gelu",
        relative_attention_num_buckets=32, relative_attention_max_distance=128, dense_act_fn="gelu_new", is_gated_act=True,
        pad_token_id=0, eos_token_id=1))
    with torch.no_grad():
        for model in (transformer, vae, text):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
    return transformer.eval(), vae.eval(), text.eval()


def build_image_encoder():
    """A tiny CLIP ViT (gelu, pre-norm, 3 layers of 16) and its processor (resize to 28×28 bicubic, CLIP normalization)."""
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection

    torch.manual_seed(13)
    encoder = CLIPVisionModelWithProjection(CLIPVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2, image_size=CLIP_SIZE, patch_size=14,
        projection_dim=8, hidden_act="gelu", layer_norm_eps=1e-5))
    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    processor = CLIPImageProcessor(size={"height": CLIP_SIZE, "width": CLIP_SIZE}, crop_size={"height": CLIP_SIZE, "width": CLIP_SIZE},
                                   do_center_crop=False, resample=3, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                   image_std=[0.26862954, 0.26130258, 0.27577711])
    return encoder.eval(), processor


RENAMES = [
    ("condition_embedder.image_embedder.norm1.", "img_emb.proj.0."),
    ("condition_embedder.image_embedder.ff.net.0.proj.", "img_emb.proj.1."),
    ("condition_embedder.image_embedder.ff.net.2.", "img_emb.proj.3."),
    ("condition_embedder.image_embedder.norm2.", "img_emb.proj.4."),
    ("condition_embedder.image_embedder.pos_embed", "img_emb.emb_pos"),
    ("attn2.add_k_proj.", "cross_attn.k_img."), ("attn2.add_v_proj.", "cross_attn.v_img."), ("attn2.norm_added_k.", "cross_attn.norm_k_img."),
    ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
    ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
    ("condition_embedder.time_proj.", "time_projection.1."),
    ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
    ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
    ("proj_out.", "head.head."),
    ("attn1.to_q.", "self_attn.q."), ("attn1.to_k.", "self_attn.k."), ("attn1.to_v.", "self_attn.v."), ("attn1.to_out.0.", "self_attn.o."),
    ("attn1.norm_q.", "self_attn.norm_q."), ("attn1.norm_k.", "self_attn.norm_k."),
    ("attn2.to_q.", "cross_attn.q."), ("attn2.to_k.", "cross_attn.k."), ("attn2.to_v.", "cross_attn.v."), ("attn2.to_out.0.", "cross_attn.o."),
    ("attn2.norm_q.", "cross_attn.norm_q."), ("attn2.norm_k.", "cross_attn.norm_k."),
    ("norm2.", "norm3."), ("ffn.net.0.proj.", "ffn.0."), ("ffn.net.2.", "ffn.2."),
]


def to_original(state):
    """diffusers WanTransformer3DModel names → the original Wan names (what city96 GGUF and ComfyUI files use)."""
    out = {}
    for name, value in state.items():
        if name == "scale_shift_table":
            out["head.modulation"] = value
            continue
        if name.endswith(".scale_shift_table"):
            out[name.replace(".scale_shift_table", ".modulation")] = value
            continue
        renamed = name
        for diffusers, original in RENAMES:
            if diffusers in renamed:
                renamed = renamed.replace(diffusers, original)
                break
        out[renamed] = value
    return {k: v.contiguous() for k, v in out.items()}


def write_single(folder, transformer):
    import gguf
    from safetensors.torch import save_file

    os.makedirs(folder, exist_ok=True)
    original = to_original(transformer.state_dict())
    save_file(original, os.path.join(folder, "wan.safetensors"))
    writer = gguf.GGUFWriter(os.path.join(folder, "wan-F16.gguf"), "wan")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    for name, value in original.items():
        writer.add_tensor(name, value.to(torch.float16).numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def schedulers():
    from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

    unipc = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=3.0)
    euler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    return unipc, euler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/video/tiny-random-wan"))
    parser.add_argument("--tokenizer", default=os.path.expanduser("~/models/video/Wan2.1-T2V-1.3B-Diffusers/tokenizer"))
    parser.add_argument("--variant", choices=VARIANTS + ["all"], default="all")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    for variant in (VARIANTS if args.variant == "all" else [args.variant]):
        make(variant, os.path.join(args.out, variant), args.tokenizer)


def make(variant, out, tokenizer_path):
    from diffusers import WanImageToVideoPipeline, WanPipeline
    from transformers import AutoTokenizer

    ti2v = variant == "ti2v"
    clip = variant in ("clip", "flf2v")
    image_only = variant in ("i2v", "clip", "flf2v")
    HEIGHT, WIDTH = SIZES[variant]
    transformer, vae, text = build_models(variant)
    image_encoder, image_processor = build_image_encoder() if clip else (None, None)
    experts = {}
    if variant in ("a14b", "i2v"):
        torch.manual_seed(9)
        from diffusers import WanTransformer3DModel
        second = WanTransformer3DModel(**transformer.config)
        with torch.no_grad():
            for p in second.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        experts = {"transformer_2": second.eval(), "boundary_ratio": 0.875}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if os.path.isdir(tokenizer_path) else "google/umt5-xxl")
    unipc, euler = schedulers()
    pipe = WanPipeline(tokenizer=tokenizer, text_encoder=text, vae=vae, scheduler=unipc, transformer=transformer, expand_timesteps=ti2v, **experts)
    pipe.save_pretrained(os.path.join(out, "diffusers"), safe_serialization=True)
    write_single(os.path.join(out, "single"), transformer)
    ref = Reference(os.path.join(out, "reference"))
    extra = {"height": HEIGHT, "width": WIDTH, "frames": FRAMES, "steps": STEPS, "guidance_scale": GUIDANCE, "prompt": PROMPT,
             "negative_prompt": NEGATIVE, "max_sequence_length": MAX_SEQUENCE}

    # Text encoder: ids and the masked, zero-padded conditioning of the pipeline.
    tokens = {}
    for i, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, padding="max_length", max_length=MAX_SEQUENCE, truncation=True, add_special_tokens=True,
                        return_attention_mask=True)
        tokens[prompt] = [int(t) for t, m in zip(ids["input_ids"], ids["attention_mask"]) if m]
        embeds = pipe._get_t5_prompt_embeds(prompt, max_sequence_length=MAX_SEQUENCE, device="cpu", dtype=torch.float32)
        ref.add(f"text_hidden_{i}", embeds[0])
    extra["tokens"] = tokens

    # Transformer on random inputs (ti2v: per-token timesteps with the first frame's tokens at 0).
    z = transformer.config.out_channels
    scale = vae.config.scale_factor_spatial
    latent_frames = (FRAMES - 1) // 4 + 1
    lh, lw = HEIGHT // scale, WIDTH // scale
    g = torch.Generator().manual_seed(11)
    latent = torch.randn(1, transformer.config.in_channels, latent_frames, lh, lw, generator=g)
    context = torch.randn(1, MAX_SEQUENCE, 32, generator=g)
    tokens_per_frame = (lh // 2) * (lw // 2)
    if ti2v:
        timestep = torch.full((1, latent_frames * tokens_per_frame), 700.0)
        timestep[:, :tokens_per_frame] = 0.0
    else:
        timestep = torch.tensor([700.0])
    ref.add("transformer_latent", latent[0])
    ref.add("transformer_timestep", timestep[0])
    ref.add("transformer_context", context[0])
    image_context = None
    if clip:
        # FLF2V takes the two images' states as a batch of two, which the transformer joins into one row sequence.
        image_context = torch.randn(2 if variant == "flf2v" else 1, (CLIP_SIZE // 14) ** 2 + 1, 16, generator=g)
        ref.add("transformer_image", image_context.reshape(-1, 16))
    ref.add("transformer_out", transformer(latent, timestep, context, encoder_hidden_states_image=image_context, return_dict=False)[0][0])
    if experts:
        ref.add("transformer_2_out", experts["transformer_2"](latent, timestep, context, return_dict=False)[0][0])
        extra["guidance_scale_2"] = GUIDANCE_2
        extra["boundary_ratio"] = experts["boundary_ratio"]

    # VAE: decode random model-space latents (the pipeline's un-normalization included), encode one image (the first chunk).
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1)
    vae_latent = torch.randn(1, z, latent_frames, lh, lw, generator=g)
    ref.add("vae_latent", vae_latent[0])
    ref.add("vae_decoded", vae.decode(vae_latent * std + mean, return_dict=False)[0][0])
    image = torch.rand(1, 3, 1, HEIGHT, WIDTH, generator=g) * 2 - 1
    ref.add("vae_image", image[0, :, 0])
    ref.add("vae_encoded_mean", (vae.encode(image).latent_dist.mean - mean)[0, :, 0] / std[0, :, 0])

    # Whole text-to-video runs from the same initial latents, with UniPC (the published scheduler) and flow-matching Euler.
    noise = torch.randn(1, z, latent_frames, lh, lw, generator=g)
    ref.add("noise", noise[0])
    for name, scheduler in ((("unipc", unipc), ("euler", euler)) if not image_only else ()):
        pipe.scheduler = scheduler
        video = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, height=HEIGHT, width=WIDTH, num_frames=FRAMES, num_inference_steps=STEPS,
                     guidance_scale=GUIDANCE, latents=noise.clone(), output_type="np", max_sequence_length=MAX_SEQUENCE,
                     **({"guidance_scale_2": GUIDANCE_2} if experts else {})).frames[0]
        ref.add(f"video_{name}", torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2))
    pipe.scheduler = unipc

    if ti2v or image_only:
        import diffusers.pipelines.wan.pipeline_wan as t2v_module
        import diffusers.pipelines.wan.pipeline_wan_i2v as i2v_module
        i2v_module.prompt_clean = t2v_module.prompt_clean  # the I2V module calls ftfy without importing it
        # Image-to-video (Wan 2.2 TI2V): the image is encoded as the first latent frame, whose tokens keep timestep 0.
        from PIL import Image
        rng = np.random.default_rng(7)
        picture = Image.fromarray((rng.random((HEIGHT, WIDTH, 3)) * 255).astype(np.uint8))
        picture.save(os.path.join(out, "reference", "input.png"))
        last = {}
        if variant == "flf2v":
            last_picture = Image.fromarray((rng.random((HEIGHT, WIDTH, 3)) * 255).astype(np.uint8))
            last_picture.save(os.path.join(out, "reference", "last.png"))
            last = {"last_image": last_picture}
            extra["last_image"] = "last.png"
        extra["image"] = "input.png"
        images = {"image_encoder": image_encoder, "image_processor": image_processor} if clip else {}
        i2v = WanImageToVideoPipeline(tokenizer=tokenizer, text_encoder=text, vae=vae, scheduler=unipc, transformer=transformer,
                                      expand_timesteps=ti2v, **images, **experts)
        if image_only:
            import shutil
            shutil.rmtree(os.path.join(out, "diffusers"))
            i2v.save_pretrained(os.path.join(out, "diffusers"), safe_serialization=True)
        if clip:
            # The image encoder's penultimate hidden states of the start image (and of the last one), as the pipeline encodes them.
            embeds = i2v.encode_image([picture] + list(last.values()), "cpu")
            ref.add("image_hidden", embeds.reshape(-1, embeds.shape[-1]))
        video = i2v(image=picture, prompt=PROMPT, negative_prompt=NEGATIVE, height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                    num_inference_steps=STEPS, guidance_scale=GUIDANCE, latents=noise.clone(), output_type="np",
                    max_sequence_length=MAX_SEQUENCE, **last, **({"guidance_scale_2": GUIDANCE_2} if experts else {})).frames[0]
        ref.add("video_i2v", torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2))
    ref.save(extra)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
