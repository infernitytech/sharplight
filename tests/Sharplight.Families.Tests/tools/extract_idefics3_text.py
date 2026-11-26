"""Writes the Llama text model of an Idefics3 / SmolVLM checkpoint as a plain LlamaForCausalLM checkpoint (with the tokenizer files), which
llama.cpp's convert_hf_to_gguf.py converts to the text GGUF (it only converts the mmproj of Idefics3ForConditionalGeneration itself).

Usage: python extract_idefics3_text.py <idefics3_dir> <out_dir>
"""
import json
import os
import shutil
import sys

from safetensors.torch import load_file, save_file

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
config = json.load(open(os.path.join(src, "config.json")))
text = dict(config["text_config"])
text.update(architectures=["LlamaForCausalLM"], model_type="llama", tie_word_embeddings=config.get("tie_word_embeddings", False))
json.dump(text, open(os.path.join(out, "config.json"), "w"), indent=2)
tensors = load_file(os.path.join(src, "model.safetensors"))
renamed = {}
for k, v in tensors.items():
    if k.startswith("model.text_model."):
        renamed["model." + k[len("model.text_model."):]] = v
    elif k == "lm_head.weight":
        renamed[k] = v
save_file(renamed, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
for f in os.listdir(src):
    if f.startswith(("tokenizer", "special_tokens", "added_tokens", "vocab", "merges", "chat_template")):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
print("wrote", out)
