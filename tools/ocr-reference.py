#!/usr/bin/env python3
"""Reference outputs of TrOCR (transformers, CPU, greedy) for the Sharplight OCR tests.

Renders a line of printed text (or reads IMAGE), writes `sharplight-sample.png`, `tokenizer.json` (when the checkpoint has only vocab.json and merges.txt)
and `sharplight-reference.json` (token ids, text, the encoder output sample and the logits of the first step) into the model directory.

Usage: ocr-reference.py MODEL_DIR [IMAGE]
"""
import json
import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

TEXT = "Sharplight reads printed text from images."


def main() -> None:
    model_dir = sys.argv[1]
    sample = os.path.join(model_dir, "sharplight-sample.png")
    if len(sys.argv) > 2:
        Image.open(sys.argv[2]).convert("RGB").save(sample)
    else:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 34)
        image = Image.new("RGB", (760, 64), "white")
        ImageDraw.Draw(image).text((12, 10), TEXT, fill="black", font=font)
        image.save(sample)
    image = Image.open(sample).convert("RGB")
    processor = TrOCRProcessor.from_pretrained(model_dir)
    if not os.path.exists(os.path.join(model_dir, "tokenizer.json")):
        processor.tokenizer.backend_tokenizer.save(os.path.join(model_dir, "tokenizer.json")) if hasattr(processor.tokenizer, "backend_tokenizer") else None
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).eval()
    pixels = processor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        encoded = model.encoder(pixel_values=pixels).last_hidden_state
        start = torch.tensor([[model.config.decoder.decoder_start_token_id]])
        first = model(pixel_values=pixels, decoder_input_ids=start).logits[0, 0]
        ids = model.generate(pixels, max_new_tokens=48, num_beams=1, do_sample=False)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    result = {
        "pixel_sample": pixels.flatten()[:16].tolist(),
        "encoder_shape": list(encoded.shape),
        "encoder_output": encoded[0].flatten().tolist(),
        "first_logits": first.tolist(),
        "token_ids": ids[0].tolist(),
        "text": text,
    }
    print(repr(text), ids[0].tolist())
    with open(os.path.join(model_dir, "sharplight-reference.json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
