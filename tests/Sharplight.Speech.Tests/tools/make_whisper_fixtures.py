"""Writes the Hugging Face transformers reference fixtures of the Whisper tests.

Usage (in a venv with torch, transformers and soundfile):
    python make_whisper_fixtures.py <checkpoint dir> <wav> <fixture.json>

The fixture holds the greedy token ids (without timestamps, with timestamps, and with language detection for multilingual models), and
every 997th value of the log-mel features [mels, 3000] and of the encoder output [1500, d_model] with their full sizes, so the tests
compare against transformers without storing megabytes of reference tensors.
"""
import json
import sys

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

STRIDE = 997


def sample(tensor: torch.Tensor) -> dict:
    flat = tensor.flatten().double()
    return {"count": flat.numel(), "stride": STRIDE, "values": [round(v, 6) for v in flat[::STRIDE].tolist()]}


def main(model_dir: str, wav: str, out: str) -> None:
    audio, rate = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert rate == 16000, rate
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    multilingual = model.generation_config.is_multilingual
    fixture = {"features": sample(features[0])}
    with torch.no_grad():
        fixture["encoder"] = sample(model.model.encoder(features).last_hidden_state[0])
        language = {"language": "en", "task": "transcribe"} if multilingual else {}
        for key, timestamps in (("plain", False), ("timestamps", True)):
            ids = model.generate(features, return_timestamps=timestamps, **language)[0].tolist()
            fixture[key] = {"ids": ids, "text": processor.decode(ids, skip_special_tokens=True)}
        if multilingual:
            ids = model.generate(features)[0].tolist()
            fixture["detected"] = {"ids": ids, "text": processor.decode(ids, skip_special_tokens=True)}
    with open(out, "w") as f:
        json.dump(fixture, f)


if __name__ == "__main__":
    main(*sys.argv[1:4])
