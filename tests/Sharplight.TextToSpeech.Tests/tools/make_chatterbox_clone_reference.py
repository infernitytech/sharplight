"""Writes the reference tensors of the Chatterbox voice-cloning tests (the chatterbox-tts package on torch, float32, CPU).

Usage (in a venv with torch, chatterbox-tts and librosa; torchaudio may be the stub next to this script with SL_TORCHAUDIO_KALDI set):
    python make_chatterbox_clone_reference.py <chatterbox dir> <output dir> <reference wav>

Follows ChatterboxTTS.prepare_conditionals on the recording and writes raw little-endian files, each stage's input included so that every encoder
can be compared on the reference's own input:
- ref24.f32, ref16.f32: the recording at 24 kHz (librosa.load) and resampled to 16 kHz (librosa.resample);
- voice encoder: ve_trimmed.f32 (librosa.effects.trim, 20 dB), ve_mel.f32 [frames, 40], ve_partials.f32 [partials, 256], ve_embed.f32 [256];
- T3 prompt tokens: t3_mel.f32 [frames, 128] (the S3 tokenizer's log-mel of the first 6 s at 16 kHz), t3_tokens.i32 (at most 150);
- S3Gen reference: dec24.f32 (the first 10 s at 24 kHz), dec16.f32 (dec24 resampled by S3Gen), prompt_feat.f32 [frames, 80], cam_fbank.f32
  [frames, 80] (the mean-normalized Kaldi filter bank), xvector.f32 [192], prompt_tokens.i32 (S3 tokens of dec16, cut to half the mel frames);
- clone.json: the lengths.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubs"))

import librosa  # noqa: E402
import perth  # noqa: E402

perth.PerthImplicitWatermarker = lambda: None  # no audio is generated; the watermarker's own dependencies are not needed
import numpy as np  # noqa: E402
import torch  # noqa: E402

from chatterbox.models.s3gen.xvector import extract_feature  # noqa: E402
from chatterbox.models.voice_encoder.melspec import melspectrogram  # noqa: E402
from chatterbox.tts import ChatterboxTTS  # noqa: E402


def save(out: str, name: str, value) -> None:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    dtype = np.int32 if name.endswith(".i32") else np.float32
    np.ascontiguousarray(array.astype(dtype)).tofile(os.path.join(out, name))


def main(model_dir: str, out: str, wav_path: str) -> None:
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    tts = ChatterboxTTS.from_local(model_dir, "cpu")
    info = {}

    ref24, _ = librosa.load(wav_path, sr=24000)
    ref16 = librosa.resample(ref24, orig_sr=24000, target_sr=16000)
    save(out, "ref24.f32", ref24)
    save(out, "ref16.f32", ref16)

    # Voice encoder (embeds_from_wavs: trim, mel, partials at rate 1.3, mean of the normalized partial embeddings, normalized).
    ve = tts.ve
    trimmed = librosa.effects.trim(ref16, top_db=20)[0]
    mel = melspectrogram(trimmed, ve.hp).T
    save(out, "ve_trimmed.f32", trimmed)
    save(out, "ve_mel.f32", mel)
    from chatterbox.models.voice_encoder.voice_encoder import get_frame_step, get_num_wins
    step = get_frame_step(0.5, 1.3, ve.hp)
    n_partials, target = get_num_wins(len(mel), step, 0.8, ve.hp)
    mels = torch.from_numpy(np.pad(mel, ((0, max(0, target - len(mel))), (0, 0)))).float()
    partials = torch.stack([mels[i * step:i * step + ve.hp.ve_partial_frames] for i in range(n_partials)])
    with torch.inference_mode():
        partial_embeds = ve(partials)
    save(out, "ve_partials.f32", partial_embeds)
    ve_embed = torch.from_numpy(ve.embeds_from_wavs([ref16], sample_rate=16000)).mean(axis=0)
    save(out, "ve_embed.f32", ve_embed)
    info.update(ve_frames=len(mel), ve_partials=n_partials, ve_step=step)

    # T3 prompt tokens: the S3 tokenizer on the first 6 s at 16 kHz, at most 150 tokens.
    tokenizer = tts.s3gen.tokenizer
    head = torch.from_numpy(ref16[:tts.ENC_COND_LEN]).unsqueeze(0)
    t3_mel = tokenizer.log_mel_spectrogram(head)[0][..., :150 * 4]
    save(out, "t3_mel.f32", t3_mel.T)
    t3_tokens, _ = tokenizer.forward([ref16[:tts.ENC_COND_LEN]], max_len=150)
    save(out, "t3_tokens.i32", t3_tokens[0])
    info.update(t3_frames=int(t3_mel.shape[-1]), t3_tokens=int(t3_tokens.shape[-1]))

    # S3Gen reference (embed_ref): mel of the first 10 s at 24 kHz, x-vector and S3 tokens of its 16 kHz resampling.
    from chatterbox.models.s3gen.s3gen import get_resampler
    dec24 = torch.from_numpy(ref24[:tts.DEC_COND_LEN]).unsqueeze(0)
    dec16 = get_resampler(24000, 16000, "cpu")(dec24)
    save(out, "dec24.f32", dec24[0])
    save(out, "dec16.f32", dec16[0])
    with torch.inference_mode():
        ref = tts.s3gen.embed_ref(dec24[0].numpy(), 24000, device="cpu")
        features, _, _ = extract_feature(dec16)
    save(out, "prompt_feat.f32", ref["prompt_feat"][0])
    save(out, "cam_fbank.f32", features[0])
    save(out, "xvector.f32", ref["embedding"][0])
    save(out, "prompt_tokens.i32", ref["prompt_token"][0])
    info.update(prompt_frames=int(ref["prompt_feat"].shape[1]), prompt_tokens=int(ref["prompt_token"].shape[1]), fbank_frames=int(features.shape[1]))

    with open(os.path.join(out, "clone.json"), "w") as f:
        json.dump(info, f)
    print(info, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
