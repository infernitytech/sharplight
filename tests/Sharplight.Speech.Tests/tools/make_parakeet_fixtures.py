"""Writes the Hugging Face transformers reference fixtures of the Parakeet (FastConformer CTC / TDT) tests.

Usage (in a venv with torch, transformers and soundfile):
    python make_parakeet_fixtures.py <checkpoint dir> <fixture.json> <wav> [<wav> ...]

Per recording (keyed by the file name without extension) the fixture holds every 97th value of the log-mel features [frames, mels]
(without the masked last frame) and of the encoder output [rows, hidden], and the greedy result of `generate` with its text: CTC models
the per-row argmax ids, TDT models the generated ids and durations. The whole recording is one input, as the model card does.
"""
import json
import os
import sys

import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

STRIDE = 97


def sample(tensor: torch.Tensor) -> dict:
    flat = tensor.flatten().double()
    return {"count": flat.numel(), "rows": tensor.shape[0], "stride": STRIDE, "values": [round(v, 6) for v in flat[::STRIDE].tolist()]}


def main(model_dir: str, out: str, wavs: list[str]) -> None:
    from transformers import ParakeetForCTC, ParakeetForTDT

    processor = AutoProcessor.from_pretrained(model_dir)
    config = json.load(open(os.path.join(model_dir, "config.json")))
    tdt = config["model_type"] == "parakeet_tdt"
    model = (ParakeetForTDT if tdt else ParakeetForCTC).from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    fixture = {}
    for wav in wavs:
        audio, rate = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert rate == 16000, rate
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        frames = int(inputs.attention_mask.sum())
        features = inputs.input_features[:, :frames]
        entry = {"features": sample(features[0])}
        with torch.no_grad():
            entry["encoder"] = sample(model.encoder(features).last_hidden_state[0])
            if tdt:
                result = model.generate(**inputs, return_dict_in_generate=True)
                entry["ids"] = result.sequences[0].tolist()
                entry["durations"] = result.durations[0].tolist()
                entry["text"] = processor.decode(result.sequences[0], skip_special_tokens=True)
            else:
                ids = model.generate(**inputs)[0]
                entry["ids"] = ids.tolist()
                entry["text"] = processor.decode(ids, skip_special_tokens=True)
        fixture[os.path.splitext(os.path.basename(wav))[0]] = entry
    with open(out, "w") as f:
        json.dump(fixture, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
