"""Writes the reference tensors of the XTTS v2 tests (the coqui-tts package on torch, float32, CPU).

Usage (in a venv with torch, coqui-tts and its dependencies; torchaudio may be the stub next to this script):
    python make_xtts_reference.py <XTTS-v2 dir> <output dir> "<speaker>" "<text>"

A built-in speaker (speakers_xtts.pth) speaks the English text. Written as raw little-endian files next to xtts.json:
- text_tokens.bin (i32): the tokenizer's ids with the start and stop tokens;
- codes.bin (i32): the GPT's greedy codes (repetition penalty 10, as in the reference's defaults) without the stop code; margins.bin (f32): the gap
  between the best and the second-best processed score per step;
- latents.bin (f32 [rows, 1024]): the GPT latents of those codes; audio.bin (f32): the HiFi-GAN waveform of the latents.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubs"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from TTS.tts.configs.xtts_config import XttsConfig  # noqa: E402
from TTS.tts.models.xtts import Xtts  # noqa: E402


def save(out: str, name: str, value, dtype) -> None:
    np.ascontiguousarray(np.asarray(value, dtype=dtype)).tofile(os.path.join(out, name + ".bin"))


def main(model_dir: str, out: str, speaker: str, text: str) -> None:
    os.makedirs(out, exist_ok=True)
    config = XttsConfig()
    config.load_json(os.path.join(model_dir, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
    voice = model.speaker_manager.speakers[speaker]
    cond_latents, speaker_embedding = voice["gpt_cond_latent"], voice["speaker_embedding"]
    gpt = model.gpt
    tokens = torch.IntTensor(model.tokenizer.encode(text.strip().lower(), lang="en")).unsqueeze(0)
    save(out, "text_tokens", F.pad(F.pad(tokens, (1, 0), value=gpt.start_text_token), (0, 1), value=gpt.stop_text_token)[0].numpy(), np.int32)
    with torch.no_grad():
        generated = gpt.generate(cond_latents=cond_latents, text_inputs=tokens, do_sample=False, num_beams=1, repetition_penalty=10.0,
                                 length_penalty=1.0, output_scores=True, return_dict_in_generate=True)
        codes, details = generated
        margins = [float((lambda t: t[0] - t[1])(torch.topk(score[0], 2).values)) for score in details.scores]
        expected = torch.tensor([codes.shape[-1] * gpt.code_stride_len])
        latents = gpt(tokens, torch.tensor([tokens.shape[-1]]), codes, expected, cond_latents=cond_latents, return_attentions=False, return_latent=True)
        wav = model.hifigan_decoder(latents, g=speaker_embedding)
    code_list = codes[0].tolist()
    real = [c for c in code_list if c != gpt.stop_audio_token]
    save(out, "codes", real, np.int32)
    save(out, "margins", margins, np.float32)
    save(out, "latents", latents[0].numpy(), np.float32)
    save(out, "audio", wav.reshape(-1).numpy(), np.float32)
    with open(os.path.join(out, "xtts.json"), "w") as f:
        json.dump({"speaker": speaker, "text": text, "codes": len(real), "latents": latents.shape[1], "samples": int(wav.numel()), "sample_rate": 24000}, f)
    print("codes", len(real), "latents", tuple(latents.shape), "samples", int(wav.numel()), "min margin", min(margins), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
