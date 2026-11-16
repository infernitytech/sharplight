"""Writes the Hugging Face transformers reference fixtures of the Voxtral tests.

Usage (in a venv with torch, transformers >= 5.23, soundfile and mistral_common[audio]):
    python make_voxtral_fixtures.py <checkpoint dir> <fixture.json> <wav> [<wav> ...]

Per recording (keyed by the file name without extension) the fixture holds every 97th value of the projected audio embeddings [rows, hidden]
and, for two requests built by `apply_transcription_request` (mistral-common's transcription template), the prompt ids and the greedy ids of
`generate` (at most 256 new tokens) with the decoded text: "auto" (no language) and "en" (English given).
"""
import json
import os
import sys

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

STRIDE = 97


def sample(tensor: torch.Tensor) -> dict:
    flat = tensor.flatten().double()
    return {"count": flat.numel(), "rows": tensor.shape[0], "stride": STRIDE, "values": [round(v, 6) for v in flat[::STRIDE].tolist()]}


def main(model_dir: str, out: str, wavs: list[str]) -> None:
    processor = AutoProcessor.from_pretrained(model_dir)
    model = VoxtralForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
    fixture = {}
    for wav in wavs:
        entry = {}
        for name, language in (("auto", None), ("en", "en")):
            inputs = processor.apply_transcription_request(language=language, audio=wav, model_id=model_dir, return_tensors="pt")
            with torch.no_grad():
                if name == "auto":
                    embeddings = model.get_audio_features(inputs["input_features"], return_dict=True).pooler_output
                    entry["embeddings"] = sample(embeddings.reshape(-1, embeddings.shape[-1]))
                ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)[0].tolist()
            prompt = inputs["input_ids"][0].tolist()
            entry[name] = {
                "prompt": prompt,
                "ids": ids[len(prompt):],
                "text": processor.batch_decode([ids[len(prompt):]], skip_special_tokens=True)[0],
            }
            print(os.path.basename(wav), name, entry[name]["text"], flush=True)
        fixture[os.path.splitext(os.path.basename(wav))[0]] = entry
    with open(out, "w") as f:
        json.dump(fixture, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
