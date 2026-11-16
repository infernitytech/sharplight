"""Writes the Hugging Face transformers reference fixtures of the Moonshine tests.

Usage (in a venv with torch, transformers and soundfile):
    python make_moonshine_fixtures.py <checkpoint dir> <fixture.json> <wav> [<wav> ...]

Per recording (keyed by the file name without extension) the fixture holds the greedy token ids of `generate` (the whole recording as one
input, as the model card does) with their text, and every 97th value of the encoder output [rows, hidden] with its full size.
"""
import json
import os
import sys

import soundfile as sf
import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration

STRIDE = 97


def sample(tensor: torch.Tensor) -> dict:
    flat = tensor.flatten().double()
    return {"count": flat.numel(), "rows": tensor.shape[0], "stride": STRIDE, "values": [round(v, 6) for v in flat[::STRIDE].tolist()]}


def main(model_dir: str, out: str, wavs: list[str]) -> None:
    processor = AutoProcessor.from_pretrained(model_dir)
    model = MoonshineForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    fixture = {}
    for wav in wavs:
        audio, rate = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert rate == 16000, rate
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            encoded = model.model.encoder(inputs.input_values).last_hidden_state[0]
            ids = model.generate(inputs.input_values)[0].tolist()
        name = os.path.splitext(os.path.basename(wav))[0]
        fixture[name] = {"encoder": sample(encoded), "ids": ids, "text": processor.decode(ids, skip_special_tokens=True)}
    with open(out, "w") as f:
        json.dump(fixture, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
