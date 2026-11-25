"""Writes the Hugging Face transformers reference fixtures of the Cohere Transcribe tests.

Usage (in a venv with torch, transformers >= 5.4, librosa and soundfile; transformers' native `cohere_asr`, no remote code):
    python make_cohere_transcribe_fixtures.py <checkpoint dir> <fixture.json> <wav>:<language> [<wav>:<language> ...]

Per recording (keyed by the file name without extension) the fixture holds the language, the waveform chunks of the feature extractor's
energy-based split (35 s at most, cut at the quietest 0.1 s of the last 5 s), and per chunk the decoder prompt (punctuation), the greedy ids
of `generate` with the chunk alone (batch 1, at most 512 new tokens) and their text; for the first chunk every 97th value of the
log-mel features [frames, mels] and of the projected encoder output [rows, decoder hidden]. Dither is off so the features are
deterministic (Sharplight adds none); `dithered_text` is the text of the model card's call (dither on, all chunks in one padded batch,
chunks joined by the processor) for comparison.
"""
import json
import os
import sys

import soundfile as sf
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

STRIDE = 97
MAX_NEW_TOKENS = 512


def sample(tensor: torch.Tensor) -> dict:
    flat = tensor.flatten().double()
    return {"count": flat.numel(), "rows": tensor.shape[0], "stride": STRIDE, "values": [round(v, 6) for v in flat[::STRIDE].tolist()]}


def main(model_dir: str, out: str, clips: list[str]) -> None:
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "6")))
    processor = AutoProcessor.from_pretrained(model_dir)
    model = CohereAsrForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
    extractor = processor.feature_extractor
    fixture = json.load(open(out)) if os.path.exists(out) else {}
    for clip in clips:
        wav, language = clip.rsplit(":", 1)
        audio, rate = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert rate == 16000, rate
        extractor.dither = 1e-5
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language=language)
        chunk_index = inputs.pop("audio_chunk_index", None)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=256)
        dithered = processor.decode(ids, skip_special_tokens=True, audio_chunk_index=chunk_index, language=language)
        dithered = dithered[0] if isinstance(dithered, list) else dithered

        extractor.dither = 0.0
        waveform = torch.from_numpy(audio)
        pieces = extractor._split_audio_chunks_energy(waveform) if len(audio) > extractor.max_audio_clip_s * 16000 else [waveform]
        chunks = []
        start = 0
        entry = {"language": language}
        for index, piece in enumerate(pieces):
            inputs = processor(piece.numpy(), sampling_rate=16000, return_tensors="pt", language=language)
            inputs.pop("audio_chunk_index", None)
            with torch.no_grad():
                if index == 0:
                    frames = int(inputs["attention_mask"].sum())
                    encoded = model.model.encoder(inputs["input_features"][:, :frames]).last_hidden_state
                    projected = model.model.decoder.proj(encoded)[0]
                    entry["features"] = sample(inputs["input_features"][0, :frames])
                    entry["encoder"] = sample(projected)
                generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()
            prompt = inputs["decoder_input_ids"][0].tolist()
            chunks.append({
                "start": start,
                "end": start + len(piece),
                "prompt": prompt,
                "ids": generated,
                "text": processor.decode(generated[len(prompt):], skip_special_tokens=True),
            })
            start += len(piece)
            print(os.path.basename(wav), index, chunks[-1]["text"], flush=True)
        separator = "" if language in ("ja", "zh") else " "
        entry["chunks"] = chunks
        entry["text"] = separator.join(c["text"].strip() for c in chunks if c["text"].strip())
        entry["dithered_text"] = dithered
        fixture[os.path.splitext(os.path.basename(wav))[0]] = entry
        with open(out, "w") as f:
            json.dump(fixture, f, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
