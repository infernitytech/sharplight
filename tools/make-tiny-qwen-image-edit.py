#!/usr/bin/env python3
"""Creates tiny random Qwen-Image-Edit-Plus pipelines (QwenImageEditPlusPipeline, as Qwen-Image-Edit-2509 and -2511) and diffusers
reference outputs for the image-generation tests.

The layout of tools/make-tiny-qwen-image.py with the editing pipeline: the Qwen2.5-VL text encoder reads the reference images in the prompt
("Picture n: <|vision_start|><|image_pad|><|vision_end|>" before the instruction, the images at 384 · 384 pixels through its windowed
vision tower, M-RoPE positions), the reference images' latents follow the image's tokens in the transformer, true classifier-free
guidance with the norm rescaling. Two variants: "edit-2509" and "edit-2511" (the transformer's zero_cond_t: the reference tokens are
modulated at timestep 0). The reference images are encoded at the output size (VAE_IMAGE_SIZE is patched to the output's area, the
image size Sharplight resizes references to); the output is 64×96.

Written to OUT/VARIANT (default ~/models/qwen-image/tiny-random-qwen-image-edit/{edit-2509,edit-2511}):

  diffusers/                       diffusers folder (model_index.json, transformer/, vae/, text_encoder/, tokenizer/, processor/, scheduler/)
  single/qwen-image-VARIANT-Q8_0.ggufthe transformer as a transformer-only GGUF (qwen_image id; the 2511 file also holds the marker tensor)
  reference/*.f32, ref_*.png       raw little-endian f32 reference tensors and the input images, described by reference/index.json

Requires torch, diffusers, transformers, gguf, safetensors and PIL. The tokenizer and processor come from --pipeline (a Qwen-Image-Edit
diffusers folder with tokenizer/ and processor/).
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

PROMPT = "make the cube red and add a hat"
NEGATIVE = " "
HEIGHT = 64
WIDTH = 96
STEPS = 4
GUIDANCE = 4.0
LENGTH = 1024
DROP = 64


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


def write_gguf(path, state, marker):
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
    if marker:
        writer.add_tensor("__index_timestep_zero__", np.array([0, 1], dtype=np.int32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return dequantized


def picture(seed, kind):
    """A small structured image (shapes on a gradient) so the references differ."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    base = np.stack([x / WIDTH, y / HEIGHT, np.full_like(x, 0.5, dtype=float)], -1) if kind == 0 else np.stack([y / HEIGHT, 1 - x / WIDTH, x / WIDTH * y / HEIGHT], -1)
    base = base + rng.normal(0, 0.03, base.shape)
    base[16:48, 24:60] = [0.8, 0.2, 0.2] if kind == 0 else [0.2, 0.7, 0.3]
    return Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))


def build(variant, args):
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
    from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as module
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

    zero_cond = variant == "edit-2511"
    torch.manual_seed(7)
    transformer = QwenImageTransformer2DModel(patch_size=2, in_channels=64, out_channels=16, num_layers=2, attention_head_dim=32,
                                              num_attention_heads=4, joint_attention_dim=64, axes_dims_rope=(4, 14, 14), zero_cond_t=zero_cond)
    rng = torch.Generator().manual_seed(3)
    latents_mean = (torch.randn(16, generator=rng) * 0.5).tolist()
    latents_std = (torch.rand(16, generator=rng) + 0.75).tolist()
    vae = AutoencoderKLQwenImage(base_dim=32, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1, attn_scales=[],
                                 temperal_downsample=[False, True, True], latents_mean=latents_mean, latents_std=latents_std)
    tokenizer = AutoTokenizer.from_pretrained(args.pipeline, subfolder="tokenizer")
    processor = Qwen2VLProcessor.from_pretrained(args.pipeline, subfolder="processor")
    config = Qwen2_5_VLConfig(
        text_config={"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "vocab_size": 152064, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
                     "rope_parameters": {"rope_type": "default", "mrope_section": [2, 3, 3], "rope_theta": 1000000.0}},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 64,
                       "fullatt_block_indexes": [1], "window_size": 112},
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
                                                base_image_seq_len=256, max_image_seq_len=8192, shift_terminal=0.02, time_shift_type="exponential")
    pipe = QwenImageEditPlusPipeline(scheduler=scheduler, vae=vae, text_encoder=text, tokenizer=tokenizer, processor=processor, transformer=transformer)
    # References are encoded at the output size (Sharplight resizes references to the output's size).
    module.VAE_IMAGE_SIZE = WIDTH * HEIGHT
    root = os.path.join(args.out, variant)
    pipe.save_pretrained(os.path.join(root, "diffusers"), safe_serialization=True)
    os.makedirs(os.path.join(root, "single"), exist_ok=True)
    state = {k: v.contiguous() for k, v in transformer.state_dict().items()}
    dequantized = write_gguf(os.path.join(root, "single", f"qwen-image-{variant}-Q8_0.gguf"), state, zero_cond)

    ref = Reference(os.path.join(root, "reference"))
    images = [picture(1, 0), picture(2, 1)]
    for i, image in enumerate(images):
        image.save(os.path.join(root, "reference", f"ref_{i}.png"))
    index = {}
    for n in (1, 2):
        cw, ch = module.calculate_dimensions(module.CONDITION_IMAGE_SIZE, WIDTH / HEIGHT)
        condition = [pipe.image_processor.resize(image, ch, cw) for image in images[:n]]
        embeds, mask = pipe.encode_prompt(image=condition, prompt=PROMPT)
        ref.add(f"text_hidden_{n}", embeds[0])
        label = "".join(f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>" for i in range(n))
        inputs = processor(text=[pipe.prompt_template_encode.format(label + PROMPT)], images=condition, padding=True, return_tensors="pt")
        index[f"ids_{n}"] = inputs["input_ids"][0].tolist()
        index[f"grid_{n}"] = inputs["image_grid_thw"].tolist()
        negative, _ = pipe.encode_prompt(image=condition, prompt=NEGATIVE)
        ref.add(f"negative_hidden_{n}", negative[0])

    # One transformer evaluation with a reference image.
    g = torch.Generator().manual_seed(1)
    lh, lw = HEIGHT // 8, WIDTH // 8
    latent = torch.randn(1, 16, lh, lw, generator=g)
    refs = torch.randn(2, 16, lh, lw, generator=g)
    context = torch.randn(LENGTH, 64, generator=g)
    valid = 37
    mask = (torch.arange(LENGTH) < valid).to(torch.float32)
    sigma = 0.7317
    packed = pipe._pack_latents(latent, 1, 16, lh, lw)
    packed_refs = [pipe._pack_latents(refs[i:i + 1], 1, 16, lh, lw) for i in range(2)]
    ref.add("transformer_latent", latent)
    ref.add("transformer_refs", refs)
    ref.add("transformer_context", context[None])
    ref.add("transformer_context_mask", mask[None])
    ref.add("transformer_timestep", torch.tensor([sigma * 1000]))
    shape = (1, lh // 2, lw // 2)
    for n in (1, 2):
        out = transformer(hidden_states=torch.cat([packed, *packed_refs[:n]], 1), encoder_hidden_states=context[None, :valid], encoder_hidden_states_mask=None,
                          timestep=torch.tensor([sigma]), img_shapes=[[shape] * (1 + n)], return_dict=False)[0][:, :packed.shape[1]]
        ref.add(f"transformer_out_{n}", pipe._unpack_latents(out, HEIGHT, WIDTH, 8)[:, :, 0])

    noise = torch.randn(1, 16, lh, lw, generator=torch.Generator().manual_seed(42))
    ref.add("noise", noise)
    noise_packed = pipe._pack_latents(noise, 1, 16, lh, lw)
    for n in (1, 2):
        image = pipe(image=images[:n], prompt=PROMPT, negative_prompt=NEGATIVE, true_cfg_scale=GUIDANCE, height=HEIGHT, width=WIDTH,
                     num_inference_steps=STEPS, latents=noise_packed.clone(), output_type="pt").images
        ref.add(f"image_{n}", image[0])
    pipe.transformer.load_state_dict({k: dequantized[k].reshape(v.shape) for k, v in state.items()})
    image = pipe(image=images, prompt=PROMPT, negative_prompt=NEGATIVE, true_cfg_scale=GUIDANCE, height=HEIGHT, width=WIDTH,
                 num_inference_steps=STEPS, latents=noise_packed.clone(), output_type="pt").images
    ref.add("q8_image_2", image[0])
    ref.save({"prompt": PROMPT, "negative": NEGATIVE, "guidance": GUIDANCE, "steps": STEPS, "height": HEIGHT, "width": WIDTH, "length": LENGTH,
              "drop": DROP, "zero_cond_t": zero_cond, **index})
    print("written", root)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/qwen-image/tiny-random-qwen-image-edit"))
    parser.add_argument("--pipeline", required=True, help="a Qwen-Image-Edit diffusers folder with tokenizer/ and processor/")
    parser.add_argument("--variants", nargs="*", default=["edit-2509", "edit-2511"])
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    for variant in args.variants:
        build(variant, args)


if __name__ == "__main__":
    main()
