"""Stable Audio 3 reference: the stable-audio-3 package (github.com/Stability-AI/stable-audio-3) run on CPU in fp32 from a local checkpoint folder of a
stabilityai/stable-audio-3-* repository (model_config.json, model.safetensors, t5gemma-b-b-ul2/). Generates a clip with a fixed seed through StableAudioModel.generate
and records what Sharplight needs to reproduce the run (raw little-endian f32 / i32, time-major, written to <refs>/<tag>_*):

  ids          the prompt's token ids (before the padding)
  text         the conditioner's prompt rows [256, 768] (the padding rows are the learned padding embedding)
  seconds      the seconds_total row [768]
  noise        the initial noise [T, 256] (torch.randn [1, 256, T], transposed)
  step_noise   the pingpong sampler's noise, [steps, T, 256] (pingpong only)
  dit_*        the first transformer evaluation: dit_x [T, 256], dit_t [1], dit_cond [257, 768] (prompt rows and seconds row), dit_out [T, 256] (the conditional
               velocity) and dit_valid [T] (the padding mask)
  steps        the latents after every step [steps, T, 256]
  latent       the final latents [T, 256]
  wav          the audio as generate() returns it (clamped to [-1, 1] and cut to the duration), [samples, 2]
  wav_dither   the same latents decoded with the autoencoder's own random dither (bottleneck noise and mask-token noise), [samples, 2]

The autoencoder adds a little random noise when it decodes (SoftNormBottleneck.noise_regularize and the decoder's mask_noise); the recorded `wav` is decoded without
it (Sharplight's decoder is deterministic), `wav_dither` with it, so the SNR between the two shows how much the dither changes the audio.
The T5Gemma encoder runs in fp32 (the package loads it in its checkpoint dtype, bf16, unless FP32_TEXT=0).

Usage: reference_stable_audio_3.py MODEL_DIR REFS_DIR [tag] [seconds] [steps] [cfg] [sampler] [negative prompt]
       sampler: euler (the base models' default) or pingpong (the post-trained models' default)
Needs the stable-audio-3 package and its dependencies (torch, torchaudio, transformers >= 5.8, einops, safetensors)."""
import json, os, sys
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
model_dir, refs = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else "sa3"
seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 8
cfg = float(sys.argv[6]) if len(sys.argv) > 6 else 7.0
sampler = sys.argv[7] if len(sys.argv) > 7 else "euler"
negative = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8] else None
prompt = os.environ.get("PROMPT", "A warm analog synth arpeggio over a slow hip hop drum loop, 90 BPM")
seed = 42

from stable_audio_3 import model as sa3_model
from stable_audio_3.loading_utils import load_diffusion_cond
from stable_audio_3.models import conditioners, bottleneck

with open(os.path.join(model_dir, "model_config.json")) as f:
    config = json.load(f)
for c in config["model"]["conditioning"]["configs"]:
    if c["type"] == "t5gemma":
        c["config"]["model_path"] = model_dir   # the local folder instead of the (possibly gated) repo_id; the subfolder stays

if os.environ.get("FP32_TEXT", "1") == "1":
    from transformers import T5GemmaEncoderModel
    original_from_pretrained = T5GemmaEncoderModel.from_pretrained.__func__

    def fp32_from_pretrained(cls, *args, **kwargs):
        kwargs["dtype"] = torch.float32
        return original_from_pretrained(cls, *args, **kwargs)

    T5GemmaEncoderModel.from_pretrained = classmethod(fp32_from_pretrained)

model = load_diffusion_cond(config, os.path.join(model_dir, "model.safetensors"), device="cpu", model_half=False)
model.use_lora = False
model.lora_names = []
sam = sa3_model.StableAudioModel(model, config, "cpu", False)
text_conditioner = model.conditioner.conditioners["prompt"]
print("t5gemma dtype", next(text_conditioner.model.parameters()).dtype)

# Deterministic decoder: the dither switched off (recorded separately below).
bottlenecks = [m for m in model.pretransform.modules() if isinstance(m, bottleneck.SoftNormBottleneck)]
resampling = [m for m in model.pretransform.modules() if hasattr(m, "mask_noise")]


def set_dither(on):
    for m in bottlenecks:
        m.noise_regularize = on and m._dither_default
    for m in resampling:
        m.mask_noise = m._mask_noise_default if on else 0.0


for m in bottlenecks:
    m._dither_default = m.noise_regularize
for m in resampling:
    m._mask_noise_default = m.mask_noise
set_dither(False)

# Record the inputs.
encoded = text_conditioner.tokenizer([prompt], truncation=True, max_length=text_conditioner.max_length, padding="max_length", return_tensors="pt")
ids = encoded["input_ids"][0][encoded["attention_mask"][0].bool()]
with torch.no_grad():
    cond = model.conditioner([{"prompt": prompt, "seconds_total": seconds}], "cpu")
text_rows = cond["prompt"][0][0].float()
seconds_row = cond["seconds_total"][0][0, 0].float()

first = {}
dit = model.model.model
original_forward = dit._forward


def recording_forward(x, t, **kwargs):
    out = original_forward(x, t, **kwargs)
    if not first:
        first.update(x=x[0].T.clone(), t=t[0:1].clone(), cond=kwargs["cross_attn_cond"][0].clone(), out=out[0].T.clone(),
                     valid=kwargs["padding_mask"][0].float().clone() if kwargs.get("padding_mask") is not None else None)
    return out


dit._forward = recording_forward

noise_draws = []
original_randn = torch.randn


def recording_randn(*args, **kwargs):
    r = original_randn(*args, **kwargs)
    if not noise_draws:
        noise_draws.append(r[0].T.clone())
    return r


step_noise = []
original_randn_like = torch.randn_like


def recording_randn_like(t, *args, **kwargs):
    r = original_randn_like(t, *args, **kwargs)
    step_noise.append(r[0].T.clone())
    return r


per_step = []
torch.randn = recording_randn
sa3_model.torch.randn = recording_randn
torch.randn_like = recording_randn_like
try:
    latents = sam.generate(prompt=prompt, negative_prompt=negative, duration=seconds, steps=steps, cfg_scale=cfg, seed=seed, sampler_type=sampler,
                           return_latents=True, callback=lambda d: per_step.append(d["x"][0].T.clone()))
finally:
    torch.randn = original_randn
    sa3_model.torch.randn = original_randn
    torch.randn_like = original_randn_like
latent = latents[0].T.clone()
per_step = per_step[1:] + [latent]   # the callback sees x before each step

with torch.no_grad():
    audio = model.pretransform.decode(latents.float(), chunked=None).clamp(-1, 1)[0, :, : int(seconds * 44100)].T
    torch.manual_seed(1234)
    set_dither(True)
    dithered = model.pretransform.decode(latents.float(), chunked=None).clamp(-1, 1)[0, :, : int(seconds * 44100)].T
    set_dither(False)
error = ((dithered - audio).pow(2).mean() / audio.pow(2).mean()).sqrt().item()
print(f"dither: relative RMS {error:.3e}, SNR {-20 * np.log10(error):.1f} dB")
print("latent frames", latent.shape[0], "audio", tuple(audio.shape), "peak", audio.abs().max().item())

os.makedirs(refs, exist_ok=True)
tensors = {
    "ids": ids.int(),
    "text": text_rows,
    "seconds": seconds_row,
    "noise": noise_draws[0],
    "dit_x": first["x"],
    "dit_t": first["t"],
    "dit_cond": first["cond"],
    "dit_out": first["out"],
    "dit_valid": first["valid"],
    "steps": torch.stack(per_step),
    "latent": latent,
    "wav": audio,
    "wav_dither": dithered,
}
if sampler == "pingpong":
    tensors["step_noise"] = torch.stack(step_noise)
for name, t in tensors.items():
    integer = name == "ids"
    np.ascontiguousarray(t.detach().float().numpy().astype("<i4" if integer else "<f4")).tofile(f"{refs}/{tag}_{name}.{'i32' if integer else 'f32'}")
print("done", {k: tuple(v.shape) for k, v in tensors.items()})
