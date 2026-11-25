#!/usr/bin/env python3
"""Creates a tiny random Cohere Transcribe checkpoint (transformers' native cohere_asr) and its reference outputs for the speech tests.

Writes OUT (default ~/models/hf/tiny-random-cohere-asr): config.json in transformers' own layout (encoder_config, decoder keys at the top
level), model.safetensors as save_pretrained writes it (NeMo tensor names under "model."; a 2-block FastConformer of width 64 with 4 heads and
16 subsampling channels, projected to a 2-layer decoder of width 32 with 2 heads; random weights incl. batch-norm statistics), the real
checkpoint's tokenizer files (16384 tokens) and the processor configuration.

Writes FIXTURE (default tests/Sharplight.Speech.Tests/Fixtures/tiny-random-cohere-asr.json) for AUDIO (default ~/models/audio/jfk.wav): every
97th value of the log-mel features [frames, mels] (dither off) and of the projected encoder output [rows, 32], the English prompt with
punctuation, the forced tokens after it, and every 97th logit of the decoder after the prompt and after each forced token.

Usage (in a venv with torch, transformers >= 5.4, librosa, soundfile and sentencepiece):
    python tools/make-tiny-cohere-asr.py --tokenizer <cohere-transcribe-03-2026 checkpoint dir> [--out OUT] [--fixture FIXTURE] [--audio AUDIO]
"""

import argparse
import json
import os
import shutil

import soundfile as sf
import torch
from transformers import AutoProcessor, CohereAsrConfig, CohereAsrForConditionalGeneration

STRIDE = 97
FORCED = [1000, 2345, 7, 5000, 13764, 3]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "processor_config.json"]


def sample(tensor: torch.Tensor) -> dict:
    flat = tensor.flatten().double()
    return {"count": flat.numel(), "rows": tensor.shape[0], "stride": STRIDE, "values": [round(v, 6) for v in flat[::STRIDE].tolist()]}


def main() -> None:
    home = os.path.expanduser("~")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="a cohere-transcribe-03-2026 checkpoint directory (tokenizer and processor files)")
    parser.add_argument("--out", default=os.path.join(home, "models/hf/tiny-random-cohere-asr"))
    parser.add_argument("--fixture", default=os.path.join(os.path.dirname(__file__), "../tests/Sharplight.Speech.Tests/Fixtures/tiny-random-cohere-asr.json"))
    parser.add_argument("--audio", default=os.path.join(home, "models/audio/jfk.wav"))
    args = parser.parse_args()

    config = CohereAsrConfig(
        encoder_config=dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256, subsampling_conv_channels=16, num_mel_bins=128),
        vocab_size=16384, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, max_position_embeddings=128,
        decoder_start_token_id=13764)
    torch.manual_seed(0)
    model = CohereAsrForConditionalGeneration(config).eval()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if not tensor.is_floating_point():
                continue
            if name.endswith("running_var"):
                tensor.uniform_(0.5, 1.5)
            elif name.endswith("norm.weight") or "layer_norm" in name or name.endswith("layernorm.weight"):
                tensor.uniform_(0.8, 1.2)
            else:
                tensor.normal_(0.0, 0.1)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    for file in TOKENIZER_FILES:
        if os.path.exists(os.path.join(args.tokenizer, file)):
            shutil.copy(os.path.join(args.tokenizer, file), args.out)
    processor = AutoProcessor.from_pretrained(args.tokenizer)
    processor.save_pretrained(args.out)

    model = CohereAsrForConditionalGeneration.from_pretrained(args.out, dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(args.out)
    processor.feature_extractor.dither = 0.0
    audio, rate = sf.read(args.audio, dtype="float32")
    assert rate == 16000, rate
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="en")
    frames = int(inputs["attention_mask"].sum())
    prompt = inputs["decoder_input_ids"][0].tolist()
    with torch.no_grad():
        encoded = model.model.encoder(inputs["input_features"][:, :frames]).last_hidden_state
        projected = model.model.decoder.proj(encoded)[0]
        ids = torch.tensor([prompt + FORCED])
        logits = model(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"], decoder_input_ids=ids).logits[0]
    logits = logits[len(prompt) - 1:]
    fixture = {
        "features": sample(inputs["input_features"][0, :frames]),
        "encoder": sample(projected),
        "prompt": prompt,
        "forced": FORCED,
        "logits": [sample(row) for row in logits],
    }
    with open(args.fixture, "w") as f:
        json.dump(fixture, f)


if __name__ == "__main__":
    main()
