"""Writes the Hugging Face transformers reference fixtures of the Canary tests.

Usage (in a venv with torch, transformers and soundfile):
    python make_canary_fixtures.py <checkpoint dir> <fixture.json> <wav> [<wav> ...]

Per recording (keyed by the file name without extension) the fixture holds every 97th value of the encoder output [rows, hidden], the
decoder prompt of `apply_transcription_request` (English, punctuation) and the greedy ids of `generate` (at most 256 new tokens) with
their text. The whole recording is one input, as the model card does.
"""
import json
import os
import sys

import soundfile as sf
import torch
from transformers import AutoProcessor, CanaryForConditionalGeneration

STRIDE = 97


def sample(tensor: torch.Tensor) -> dict:
    flat = tensor.flatten().double()
    return {"count": flat.numel(), "rows": tensor.shape[0], "stride": STRIDE, "values": [round(v, 6) for v in flat[::STRIDE].tolist()]}


def main(model_dir: str, out: str, wavs: list[str]) -> None:
    processor = AutoProcessor.from_pretrained(model_dir)
    model = CanaryForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    fixture = {}
    for wav in wavs:
        audio, rate = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert rate == 16000, rate
        inputs = processor.apply_transcription_request(audio, source_language="en")
        frames = int(inputs["attention_mask"].sum())
        with torch.no_grad():
            encoded = model.model.encoder(inputs["input_features"][:, :frames]).last_hidden_state[0]
            ids = model.generate(**inputs, max_new_tokens=256)[0].tolist()
        prompt = inputs["decoder_input_ids"][0].tolist()
        fixture[os.path.splitext(os.path.basename(wav))[0]] = {
            "encoder": sample(encoded),
            "prompt": prompt,
            "ids": ids,
            "text": processor.decode(ids[len(prompt):], skip_special_tokens=True),
        }
    with open(out, "w") as f:
        json.dump(fixture, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
