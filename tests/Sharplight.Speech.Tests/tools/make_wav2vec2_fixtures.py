"""Writes the Hugging Face transformers reference fixtures of the wav2vec2 CTC tests.

Usage (in a venv with torch, transformers and soundfile):
    python make_wav2vec2_fixtures.py <checkpoint dir> <fixture.json> <wav> [<wav> ...]

Per recording (keyed by the file name without extension) the fixture holds every 97th value of the encoder's hidden states
[rows, hidden] and the greedy CTC result: the per-row argmax ids of `Wav2Vec2ForCTC` and the decoded text. The whole recording is one input.
"""
import json
import os
import sys

import soundfile as sf
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

STRIDE = 97


def sample(tensor: torch.Tensor) -> dict:
    flat = tensor.flatten().double()
    return {"count": flat.numel(), "rows": tensor.shape[0], "stride": STRIDE, "values": [round(v, 6) for v in flat[::STRIDE].tolist()]}


def main(model_dir: str, out: str, wavs: list[str]) -> None:
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    fixture = {}
    for wav in wavs:
        audio, rate = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert rate == 16000, rate
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            output = model(inputs.input_values, output_hidden_states=True)
        ids = output.logits.argmax(-1)[0]
        fixture[os.path.splitext(os.path.basename(wav))[0]] = {
            "encoder": sample(output.hidden_states[-1][0]),
            "ids": ids.tolist(),
            "text": processor.decode(ids),
        }
    with open(out, "w") as f:
        json.dump(fixture, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
