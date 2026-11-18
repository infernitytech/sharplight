"""Writes the transformers reference fixtures of the Voxtral Realtime tests.

Usage (venv with torch, transformers >= 5.23, soundfile, mistral_common[audio]):
    python make_voxtral_realtime_fixtures.py <checkpoint dir> <fixture.json> <wav> [<wav> ...]

Per recording: the prompt ids, the number of mel frames, a stride sample of the projected audio embeddings and the greedy ids of `generate`
(offline mode, float32) with the decoded text.
"""
import json
import os
import sys

import soundfile as sf
import torch
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

STRIDE = 97


def sample(tensor: torch.Tensor) -> dict:
    flat = tensor.flatten().double()
    return {"count": flat.numel(), "rows": tensor.shape[0], "stride": STRIDE, "values": [round(v, 6) for v in flat[::STRIDE].tolist()]}


def main(model_dir: str, out: str, wavs: list[str]) -> None:
    processor = AutoProcessor.from_pretrained(model_dir)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
    fixture = {}
    for wav in wavs:
        audio, rate = sf.read(wav, dtype="float32")
        assert rate == 16000
        inputs = processor(audio, return_tensors="pt")
        print({k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in inputs.items()}, flush=True)
        with torch.no_grad():
            audio_out = model.get_audio_features(input_features=inputs["input_features"], return_dict=True)
            ids = model.generate(**inputs, do_sample=False)[0].tolist()
        prompt = inputs["input_ids"][0].tolist()
        entry = {
            "prompt": prompt,
            "frames": inputs["input_features"].shape[-1],
            "numDelayTokens": int(inputs["num_delay_tokens"]),
            "embeddings": sample(audio_out.pooler_output.reshape(-1, audio_out.pooler_output.shape[-1])),
            "ids": ids[len(prompt):],
            "text": processor.batch_decode([ids[len(prompt):]], skip_special_tokens=True)[0],
        }
        print(os.path.basename(wav), entry["text"], len(prompt), len(ids), flush=True)
        fixture[os.path.splitext(os.path.basename(wav))[0]] = entry
    with open(out, "w") as f:
        json.dump(fixture, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
