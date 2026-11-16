"""Writes the reference tensors of the XTTS v2 voice-cloning tests (coqui-tts on torch, float32, CPU).

Usage (in a venv with torch and coqui-tts):
    SL_TORCHAUDIO_DIR=<an unpacked torchaudio wheel> python make_xtts_clone_reference.py <XTTS-v2 dir> <output dir> <reference wav (16-bit PCM)>

XTTS's cloning front ends need the real torchaudio (MelSpectrogram with the Slaney norm and a Hamming window, functional.resample); its Python
part runs from an unpacked wheel named by SL_TORCHAUDIO_DIR even when its compiled part does not match the installed torch. torchaudio.load is
replaced by a reader of 16-bit PCM WAV files (torchaudio's own load needs torchcodec). Follows Xtts.get_conditioning_latents with the model's
settings (gpt_cond_len 30, gpt_cond_chunk_len 4, max_ref_len 30) and writes raw little-endian float32 files:
- audio22.f32 (the recording at 22.05 kHz), audio16.f32 (resampled to 16 kHz for the speaker encoder);
- cond_mels.f32 [chunks, 80, frames] (the normalized log-mel of every 4 s chunk, the last one zero padded to the longest), cond_frames.i32;
- style.f32 [chunks, 32, 1024] (each chunk's Perceiver latents), latents.f32 [32, 1024] (their mean);
- speaker_spec.f32 [64, frames] (the speaker encoder's mel power spectrum), speaker.f32 [512];
- clone.json: the lengths.
"""
import json
import os
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubs"))
if os.environ.get("SL_TORCHAUDIO_DIR"):
    sys.path.insert(0, os.environ["SL_TORCHAUDIO_DIR"])

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402


def read_wav(path, *args, **kwargs):
    with wave.open(str(path), "rb") as f:
        assert f.getsampwidth() == 2, "16-bit PCM expected"
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        channels = f.getnchannels()
        return torch.from_numpy(data.reshape(-1, channels).T.copy()), f.getframerate()


torchaudio.load = read_wav

from TTS.tts.configs.xtts_config import XttsConfig  # noqa: E402
from TTS.tts.models import xtts as xtts_module  # noqa: E402
from TTS.tts.models.xtts import Xtts  # noqa: E402


def save(out: str, name: str, value) -> None:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    dtype = np.int32 if name.endswith(".i32") else np.float32
    np.ascontiguousarray(array.astype(dtype)).tofile(os.path.join(out, name))


def main(model_dir: str, out: str, wav_path: str) -> None:
    os.makedirs(out, exist_ok=True)
    config = XttsConfig()
    config.load_json(os.path.join(model_dir, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)

    audio = xtts_module.load_audio(wav_path, 22050)[:, : 22050 * config.max_ref_len]
    save(out, "audio22.f32", audio[0])
    audio16 = torchaudio.functional.resample(audio, 22050, 16000)
    save(out, "audio16.f32", audio16[0])

    with torch.inference_mode():
        encoder = model.hifigan_decoder.speaker_encoder
        spec = encoder.torch_spec(audio16.clone())
        save(out, "speaker_spec.f32", spec[0])
        speaker = model.get_speaker_embedding(audio, 22050)
        save(out, "speaker.f32", speaker.reshape(-1))

        clip = audio[:, : 22050 * config.gpt_cond_len]
        mels, styles = [], []
        step = 22050 * config.gpt_cond_chunk_len
        for i in range(0, clip.shape[1], step):
            chunk = clip[:, i:i + step]
            if chunk.size(-1) < 22050 * 0.33:
                continue
            mel = xtts_module.wav_to_mel_cloning(chunk, mel_norms=model.mel_stats.cpu(), n_fft=2048, hop_length=256, win_length=1024, power=2,
                                                 normalized=False, sample_rate=22050, f_min=0, f_max=8000, n_mels=80)
            mels.append(mel[0])
            styles.append(model.gpt.get_style_emb(mel, None)[0].transpose(0, 1))
        latents, _ = model.get_conditioning_latents(wav_path, gpt_cond_len=config.gpt_cond_len, gpt_cond_chunk_len=config.gpt_cond_chunk_len,
                                                    max_ref_length=config.max_ref_len, sound_norm_refs=config.sound_norm_refs)

    frames = [m.shape[-1] for m in mels]
    padded = torch.zeros(len(mels), 80, max(frames))
    for i, m in enumerate(mels):
        padded[i, :, : m.shape[-1]] = m
    save(out, "cond_mels.f32", padded)
    save(out, "cond_frames.i32", frames)
    save(out, "style.f32", torch.stack(styles))
    save(out, "latents.f32", latents[0])
    info = {"samples22": int(audio.shape[1]), "samples16": int(audio16.shape[1]), "chunks": len(mels), "cond_frames": frames,
            "speaker_frames": int(spec.shape[-1])}
    with open(os.path.join(out, "clone.json"), "w") as f:
        json.dump(info, f)
    print(info, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
