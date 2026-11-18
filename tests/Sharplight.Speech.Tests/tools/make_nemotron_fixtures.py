"""Writes the Hugging Face transformers reference fixtures of the Nemotron 3.5 ASR streaming tests.

Usage (in a venv with torch, transformers >= 5.23 and soundfile):
    python make_nemotron_fixtures.py <checkpoint dir> <fixture.json> <wav> [<wav> ...]

Per recording (keyed by the file name without extension): every 97th value of the log-mel features [frames, mels] (without the masked
frames), of the encoder output [rows, hidden] (before the language prompt projector) and of the projected joint-space encoder output, the
greedy RNN-T ids of `generate` (offline, default lookahead of the checkpoint, language "auto") and their text. The whole recording is one input.
"""
import json
import os
import sys

import soundfile as sf
import torch
from transformers import AutoProcessor, Nemotron3_5AsrForRNNT

STRIDE = 97


def sample(tensor: torch.Tensor) -> dict:
    flat = tensor.flatten().double()
    return {"count": flat.numel(), "rows": tensor.shape[0], "stride": STRIDE, "values": [round(v, 6) for v in flat[::STRIDE].tolist()]}


def main(model_dir: str, out: str, wavs: list[str]) -> None:
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Nemotron3_5AsrForRNNT.from_pretrained(model_dir, dtype=torch.float32).eval()
    fixture = {}
    for wav in wavs:
        audio, rate = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert rate == 16000, rate
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="auto")
        frames = int(inputs.attention_mask.sum())
        entry = {"features": sample(inputs.input_features[0, :frames]), "lookahead": int(inputs["num_lookahead_tokens"])}
        with torch.no_grad():
            features = model.get_audio_features(
                input_features=inputs.input_features, attention_mask=inputs.attention_mask, prompt_ids=inputs["prompt_ids"],
                num_lookahead_tokens=inputs["num_lookahead_tokens"])
            entry["encoder"] = sample(features.last_hidden_state[0])
            entry["joint"] = sample(features.pooler_output[0])
            ids = model.generate(**inputs)
            ids = ids.sequences if hasattr(ids, "sequences") else ids
            entry["ids"] = ids[0].tolist()
            entry["text"] = processor.decode(ids[0], skip_special_tokens=True)
        print(os.path.basename(wav), entry["lookahead"], entry["text"], flush=True)
        fixture[os.path.splitext(os.path.basename(wav))[0]] = entry
    with open(out, "w") as f:
        json.dump(fixture, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
