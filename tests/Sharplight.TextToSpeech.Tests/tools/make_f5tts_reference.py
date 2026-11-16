"""Writes the reference tensors of the F5-TTS tests (the f5-tts package on torch, float32, CPU).

Usage (in a venv with torch, f5-tts, vocos, x_transformers and torchdiffeq; torchaudio may be the stub next to this script):
    python make_f5tts_reference.py <F5TTS_v1_Base dir> <vocos dir> <output dir> <reference wav> "<reference text>" "<text>"

Written as raw little-endian files next to f5tts.json:
- text.bin (i32): the character ids of reference text + text (the vocabulary of vocab.txt, -1 padding excluded);
- cond.bin (f32 [frames, 100]): the log-mel of the reference audio (RMS-normalized to 0.1 when quieter, resampled to 24 kHz);
- noise.bin (f32 [duration, 100]): the starting noise; times.bin (f32 [33]): the sway-sampled time points;
- mel.bin (f32 [duration, 100]): the sampled mel (32 Euler steps, guidance 2) with the reference frames restored;
- audio.bin (f32): Vocos' waveform of the generated part.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubs"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from f5_tts.infer import utils_infer  # noqa: E402
from f5_tts.model import DiT  # noqa: E402


def save(out: str, name: str, value, dtype) -> None:
    np.ascontiguousarray(np.asarray(value, dtype=dtype)).tofile(os.path.join(out, name + ".bin"))


def main(model_dir: str, vocos_dir: str, out: str, wav: str, ref_text: str, text: str) -> None:
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    arch = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, text_mask_padding=True, conv_layers=4, pe_attn_head=None)
    model = utils_infer.load_model(DiT, arch, os.path.join(model_dir, "model_1250000.safetensors"), mel_spec_type="vocos",
                                   vocab_file=os.path.join(model_dir, "vocab.txt"), device="cpu")
    vocoder = utils_infer.load_vocoder("vocos", is_local=True, local_path=vocos_dir, device="cpu")
    captured = {}
    original_sample = model.sample

    def recording_sample(cond, text, duration, **kwargs):
        original_randn = torch.randn

        def recording_randn(*args, **kw):
            value = original_randn(*args, **kw)
            captured["noise"] = value.clone()
            return value

        torch.randn = recording_randn
        import torchdiffeq
        original_odeint = torchdiffeq.odeint

        def recording_odeint(fn, y0, t, **kw):
            captured["times"] = t.clone()
            return original_odeint(fn, y0, t, **kw)

        utils_infer_cfm = sys.modules["f5_tts.model.cfm"]
        utils_infer_cfm.odeint = recording_odeint
        try:
            captured["cond"] = model.mel_spec(cond).permute(0, 2, 1)[0].clone()
            ids = utils_infer_cfm.list_str_to_idx(text, model.vocab_char_map)[0]
            captured["text"] = ids[ids >= 0].clone()
            captured["duration"] = int(duration)
            result, trajectory = original_sample(cond, text, duration, **kwargs)
            captured["mel"] = result[0].clone()
            return result, trajectory
        finally:
            torch.randn = original_randn
            utils_infer_cfm.odeint = original_odeint

    model.sample = recording_sample
    wave, _, _ = utils_infer.infer_process(wav, ref_text, text, model, vocoder, mel_spec_type="vocos", device="cpu",
                                           nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0, speed=1.0, cross_fade_duration=0.15)
    save(out, "text", captured["text"].numpy(), np.int32)
    save(out, "cond", captured["cond"].numpy(), np.float32)
    save(out, "noise", captured["noise"].reshape(-1, 100).numpy(), np.float32)
    save(out, "times", captured["times"].numpy(), np.float32)
    save(out, "mel", captured["mel"].numpy(), np.float32)
    save(out, "audio", np.asarray(wave), np.float32)
    with open(os.path.join(out, "f5tts.json"), "w") as f:
        json.dump({"reference_text": ref_text, "text": text, "duration": captured["duration"], "reference_frames": captured["cond"].shape[0],
                   "samples": int(np.asarray(wave).shape[-1]), "sample_rate": 24000}, f)
    print("duration", captured["duration"], "cond", tuple(captured["cond"].shape), "samples", np.asarray(wave).shape)


if __name__ == "__main__":
    main(*sys.argv[1:7])
