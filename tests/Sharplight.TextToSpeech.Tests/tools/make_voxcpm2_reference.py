"""Reference of VoxCPM2 text to speech (the official voxcpm package 2.0.3, VoxCPM2Model, CPU float32).

Usage: python make_voxcpm2_reference.py <checkpoint dir> <output dir> <text> [--reference wav] [--prompt wav --prompt-text TEXT] [--max-len N] [--seed S]
       [--timesteps N] [--cfg G]
The wav files must be mono 16 kHz (librosa then loads them without resampling). The checkpoint is read in float32 (config dtype overridden in a scratch copy
of config.json next to links to the weights). Each patch's diffusion noise is seeded with seed + patch index.

Writes into <output dir>: info.json (text, tokens, masks, steps, context patches), prompt_feats.bin [rows, 4, 64] (the prompt's audio features, zero for text
rows), noise.bin [steps, 64, 4] (the diffusion start noise of each patch), lm_hidden.bin [steps, hidden] and residual_hidden.bin [steps, hidden] (the
diffusion conditions), stop.bin [steps, 2] (stop logits), patches.bin [steps, 4, 64] (generated latent patches), embeds.bin [steps, hidden] (the local
encoder's image of each generated patch), latents.bin [64, frames] (what the AudioVAE decodes, context patches included) and audio.bin (48 kHz, f32).
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("REF_THREADS", "6")))
ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("out")
ap.add_argument("text")
ap.add_argument("--reference")
ap.add_argument("--prompt")
ap.add_argument("--prompt-text")
ap.add_argument("--max-len", type=int, default=24)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--timesteps", type=int, default=10)
ap.add_argument("--cfg", type=float, default=2.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

from voxcpm.model.voxcpm2 import VoxCPM2Model  # noqa: E402

scratch = tempfile.mkdtemp(prefix="voxcpm2-f32-", dir=a.out)
for name in os.listdir(a.model_dir):
    if name != "config.json":
        os.symlink(os.path.abspath(os.path.join(a.model_dir, name)), os.path.join(scratch, name))
config = json.load(open(os.path.join(a.model_dir, "config.json")))
config["dtype"] = "float32"
config["device"] = "cpu"
json.dump(config, open(os.path.join(scratch, "config.json"), "w"))
model = VoxCPM2Model.from_local(scratch, optimize=False, device="cpu")
model.eval()

record = {"noise": [], "lm": [], "res": [], "stop": [], "patch": [], "embed": [], "prefill": None}
state = {"patch": 0}

orig_forward = model.feat_decoder.forward


def decoder_forward(mu, n_timesteps, patch_size, cond, temperature=1.0, cfg_value=1.0, sway_sampling_coef=1.0, use_cfg_zero_star=True):
    seed = a.seed + state["patch"]
    torch.manual_seed(seed)
    z = torch.randn((mu.shape[0], model.feat_decoder.in_channels, patch_size), dtype=mu.dtype)
    torch.manual_seed(seed)
    out = orig_forward(mu, n_timesteps, patch_size, cond, temperature, cfg_value, sway_sampling_coef, use_cfg_zero_star)
    record["noise"].append(z[0].clone())
    record["patch"].append(out[0].transpose(0, 1).clone())
    state["patch"] += 1
    return out


model.feat_decoder.forward = decoder_forward
model.lm_to_dit_proj.register_forward_hook(lambda m, i, o: record["lm"].append(i[0].reshape(-1)[-i[0].shape[-1]:].clone()))
model.res_to_dit_proj.register_forward_hook(lambda m, i, o: record["res"].append(i[0].reshape(-1)[-i[0].shape[-1]:].clone()))
model.stop_head.register_forward_hook(lambda m, i, o: record["stop"].append(o.reshape(-1, 2)[-1].clone()))


def enc_hook(m, i, o):
    if o.dim() == 3 and o.shape[1] > 1 or record["prefill"] is None:
        record["prefill"] = o[0].clone()
    else:
        record["embed"].append(o.reshape(-1)[-o.shape[-1]:].clone())


model.enc_to_lm_proj.register_forward_hook(enc_hook)

captured = {}
orig_inference = model._inference


def inference(text, text_mask, feat, feat_mask, **kw):
    captured["tokens"] = text[0].tolist()
    captured["text_mask"] = text_mask[0].tolist()
    captured["audio_mask"] = feat_mask[0].tolist()
    captured["feats"] = feat[0].float().clone()
    for item in orig_inference(text, text_mask, feat, feat_mask, **kw):
        captured["latents"] = item[0][0].clone()
        captured["context"] = item[2]
        yield item


model._inference = inference
cache = None
if a.reference or a.prompt:
    cache = model.build_prompt_cache(prompt_text=a.prompt_text if a.prompt else None, prompt_wav_path=a.prompt, reference_wav_path=a.reference)
with torch.no_grad():
    audio, target_tokens, _ = model.generate_with_prompt_cache(target_text=a.text, prompt_cache=cache, max_len=a.max_len, inference_timesteps=a.timesteps,
                                                               cfg_value=a.cfg, retry_badcase=False)

steps = len(record["patch"])
# The prefill's stop logits come first (the last row's), then one per step; the loop computes the stop logits before the language model's step.
stop = torch.stack(record["stop"][:steps])


def save(name, tensor):
    np.ascontiguousarray(tensor.detach().float().numpy()).astype(np.float32).tofile(os.path.join(a.out, name))


save("prompt_feats.bin", captured["feats"])
save("noise.bin", torch.stack(record["noise"]))
save("lm_hidden.bin", torch.stack(record["lm"][:steps]))
save("residual_hidden.bin", torch.stack(record["res"][:steps]))
save("stop.bin", stop)
save("patches.bin", torch.stack(record["patch"]))
save("embeds.bin", torch.stack(record["embed"][:steps]))
save("prefill_embeds.bin", record["prefill"])
save("latents.bin", captured["latents"])
save("audio.bin", audio.reshape(-1))
info = {
    "text": a.text,
    "reference": os.path.basename(a.reference) if a.reference else None,
    "prompt": os.path.basename(a.prompt) if a.prompt else None,
    "prompt_text": a.prompt_text,
    "tokens": captured["tokens"],
    "text_mask": captured["text_mask"],
    "audio_mask": captured["audio_mask"],
    "target_tokens": target_tokens.tolist(),
    "steps": steps,
    "context": int(captured["context"]),
    "samples": int(audio.numel()),
    "seed": a.seed,
    "timesteps": a.timesteps,
    "cfg": a.cfg,
    "max_len": a.max_len,
}
json.dump(info, open(os.path.join(a.out, "info.json"), "w"), indent=1, ensure_ascii=False)
for name in os.listdir(scratch):
    os.unlink(os.path.join(scratch, name))
os.rmdir(scratch)
print("steps", steps, "tokens", len(captured["tokens"]), "samples", audio.numel(), "context", captured["context"])
