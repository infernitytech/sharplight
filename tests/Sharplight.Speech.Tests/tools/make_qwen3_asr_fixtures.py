"""Writes the Hugging Face transformers reference fixtures of the Qwen3-ASR tests.

Usage (in a venv with torch, transformers >= 5.23 and soundfile):
    python make_qwen3_asr_fixtures.py <checkpoint dir (the -hf layout)> <fixture.json> <wav> [<wav> ...]

Per recording (keyed by the file name without extension) the fixture holds the number of valid log-mel frames, every 97th value of the
log-mel features [mel bins, frames] and of the projected audio embeddings [rows, hidden], and for two requests the prompt ids and the
greedy ids of `generate` (at most 256 new tokens) with the decoded text: "auto" (no language given, the model names it) and "en" (the
language forced by the assistant prefix "language English<asr_text>").
"""
import json
import os
import sys

import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

STRIDE = 97


def sample(tensor: torch.Tensor) -> dict:
    flat = tensor.flatten().double()
    return {"count": flat.numel(), "rows": tensor.shape[0], "stride": STRIDE, "values": [round(v, 6) for v in flat[::STRIDE].tolist()]}


def main(model_dir: str, out: str, wavs: list[str]) -> None:
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
    fixture = {}
    for wav in wavs:
        audio, rate = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert rate == 16000, rate
        entry = {}
        for name, language in (("auto", None), ("en", "en")):
            inputs = processor.apply_transcription_request(audio, language=language)
            mask = inputs["input_features_mask"]
            frames = int(mask.sum())
            with torch.no_grad():
                if name == "auto":
                    embeddings = model.get_audio_features(inputs["input_features"], mask, return_dict=True).pooler_output
                    entry["frames"] = frames
                    entry["features"] = sample(inputs["input_features"][0, :, :frames])
                    entry["embeddings"] = sample(embeddings)
                ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)[0].tolist()
            prompt = inputs["input_ids"][0].tolist()
            entry[name] = {
                "prompt": prompt,
                "ids": ids[len(prompt):],
                "text": processor.decode(ids[len(prompt):], skip_special_tokens=True),
            }
            print(os.path.basename(wav), name, entry[name]["text"], flush=True)
        fixture[os.path.splitext(os.path.basename(wav))[0]] = entry
    with open(out, "w") as f:
        json.dump(fixture, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
