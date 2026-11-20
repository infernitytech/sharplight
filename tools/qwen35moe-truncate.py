"""Builds a layer- and expert-reduced text-only copy of a Qwen3.5-MoE checkpoint by fetching only the needed tensor byte ranges.

  python -I tools/qwen35moe-truncate.py <repo> <revision> <out-dir> [--layers 4] [--experts 32]

Keeps decoder layers 0..L-1 (the config's layer types), experts 0..E-1 (router rows sliced to match), the embedding, final norm and
LM head, renames model.language_model.* to model.* and writes a qwen3_5_moe_text config. Quantized tensors are copied as stored.
"""
import json
import os
import re
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

args = sys.argv[1:]
repo, rev, out = args[0], args[1], Path(args[2])
layers = int(args[args.index("--layers") + 1]) if "--layers" in args else 4
experts = int(args[args.index("--experts") + 1]) if "--experts" in args else 32
token = os.environ.get("HF_TOKEN")
base = f"https://huggingface.co/{repo}/resolve/{rev}/"
out.mkdir(parents=True, exist_ok=True)


def get(name, start=None, end=None):
    request = urllib.request.Request(base + name)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end}")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == 4:
                raise


for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "vocab.json", "merges.txt"]:
    try:
        (out / name).write_bytes(get(name))
    except Exception:
        pass

config = json.loads(get("config.json"))
text = dict(config["text_config"])
text["model_type"] = "qwen3_5_moe_text"
text["architectures"] = ["Qwen3_5MoeForCausalLM"]
text["num_hidden_layers"] = layers
text["layer_types"] = text["layer_types"][:layers]
text["num_experts"] = experts
text["num_experts_per_tok"] = min(text["num_experts_per_tok"], experts)
text["mtp_num_hidden_layers"] = 0
text["tie_word_embeddings"] = config.get("tie_word_embeddings", False)
quant = config.get("quantization_config") or text.get("quantization_config")
if quant:
    for key in ["modules_to_not_convert", "ignore"]:
        if isinstance(quant.get(key), list):
            quant[key] = [m.replace("model.language_model.", "model.") for m in quant[key]
                          if not re.search(r"layers\.(\d+)\.", m) or int(re.search(r"layers\.(\d+)\.", m).group(1)) < layers]
    text["quantization_config"] = quant
(out / "config.json").write_text(json.dumps(text, indent=2))
for extra in ["hf_quant_config.json"]:
    try:
        (out / extra).write_bytes(get(extra))
    except Exception:
        pass

weight_map = json.loads(get("model.safetensors.index.json"))["weight_map"]


def wanted(name):
    if not name.startswith("model.language_model.") and not name.startswith("lm_head."):
        return False
    m = re.search(r"\.layers\.(\d+)\.", name)
    if m and int(m.group(1)) >= layers:
        return False
    e = re.search(r"\.experts\.(\d+)\.", name)
    return not (e and int(e.group(1)) >= experts)


names = [n for n in weight_map if wanted(n)]
headers = {}
for shard in sorted({weight_map[n] for n in names}):
    size = struct.unpack("<Q", get(shard, 0, 7))[0]
    headers[shard] = (8 + size, json.loads(get(shard, 8, 7 + size)))
DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "U8": 1, "I32": 4, "I8": 1}


def fetch(name):
    shard = weight_map[name]
    offset, header = headers[shard]
    info = header[name]
    begin, end = info["data_offsets"]
    data = get(shard, offset + begin, offset + end - 1) if end > begin else b""
    shape = list(info["shape"])
    if name.endswith("mlp.gate.weight") and shape[0] > experts:
        row = len(data) // shape[0]
        data = data[: row * experts]
        shape[0] = experts
    return name.replace("model.language_model.", "model."), info["dtype"], shape, data


with ThreadPoolExecutor(8) as pool:
    tensors = list(pool.map(fetch, names))
tensors.sort(key=lambda t: t[0])
header, position = {}, 0
for name, dtype, shape, data in tensors:
    header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [position, position + len(data)]}
    position += len(data)
blob = json.dumps(header).encode()
blob += b" " * ((8 - len(blob) % 8) % 8)
with open(out / "model.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    for _, _, _, data in tensors:
        f.write(data)
print(f"{len(tensors)} tensors, {position / 1e9:.2f} GB -> {out}")
