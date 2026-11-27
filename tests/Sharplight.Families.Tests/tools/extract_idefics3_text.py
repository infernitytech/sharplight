"""Writes the Llama text model of an Idefics3 / SmolVLM checkpoint as a plain LlamaForCausalLM checkpoint (with the tokenizer files), which
llama.cpp's convert_hf_to_gguf.py converts to the text GGUF (it only converts the mmproj of Idefics3ForConditionalGeneration itself). An
LFM2-VL checkpoint (model_type lfm2_vl) gives its LFM2 text model as Lfm2ForCausalLM the same way.

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
lfm2 = config["model_type"] == "lfm2_vl"
if lfm2:
    text.update(architectures=["Lfm2ForCausalLM"], model_type="lfm2", tie_word_embeddings=True)
else:
    text.update(architectures=["LlamaForCausalLM"], model_type="llama", tie_word_embeddings=config.get("tie_word_embeddings", False))
prefix = "model.language_model." if lfm2 else "model.text_model."
json.dump(text, open(os.path.join(out, "config.json"), "w"), indent=2)
tensors = load_file(os.path.join(src, "model.safetensors"))
renamed = {}
for k, v in tensors.items():
    if k.startswith(prefix):
        renamed["model." + k[len(prefix):]] = v
    elif k == "lm_head.weight":
        renamed[k] = v
save_file(renamed, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
for f in os.listdir(src):
    if f.startswith(("tokenizer", "special_tokens", "added_tokens", "vocab", "merges", "chat_template")):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
print("wrote", out)
