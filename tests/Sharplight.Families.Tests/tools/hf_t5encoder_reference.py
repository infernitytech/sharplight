"""Reference embeddings of a T5 encoder GGUF (llama.cpp t5encoder) from transformers' T5EncoderModel on its source checkpoint.

Usage: python hf_t5encoder_reference.py <hf_model_dir> <model.gguf>

Mean-pooled final hidden states (after the encoder's final norm) of the token sequences llamacpp_embeddings.py uses, written next to the
GGUF as <model.gguf>.reference_embeddings.json. GGUF files carry no relative_attention_max_distance, llama.cpp always uses 128, so the
reference does too. llama.cpp b11177's llama-server /embedding is not used: for t5encoder it disagrees with transformers (and with its own
t5 encoder, which Sharplight matches through the encoder-decoder cases).
"""
import json
import sys

import torch
from transformers import T5EncoderModel

INPUTS = [[5, 77, 301, 12, 999, 4], [17 + 31 * i % 1000 for i in range(23)], [3 + 57 * i % 1000 for i in range(40)]]

model_dir, gguf = sys.argv[1], sys.argv[2]
model = T5EncoderModel.from_pretrained(model_dir, dtype=torch.float32, relative_attention_max_distance=128).eval()
embeddings = []
with torch.no_grad():
    for ids in INPUTS:
        embeddings.append(model(input_ids=torch.tensor([ids])).last_hidden_state[0].mean(0).tolist())
with open(gguf + ".reference_embeddings.json", "w") as f:
    json.dump({"pooling": "mean", "inputs": INPUTS, "embeddings": embeddings}, f)
print("wrote", gguf + ".reference_embeddings.json")
