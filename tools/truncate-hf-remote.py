"""Cut a Hugging Face safetensors checkpoint on the Hub to its first layers and experts, fetching only the kept tensors' byte ranges.

    python -I tools/truncate-hf-remote.py <repo> <revision> <out dir> --layers N [--experts E] [--drop REGEX ...]
        [--rows REGEX=COUNT ...] [--set path.to.key=JSON ...] [--layer-key KEY]

Keeps decoder layers 0..N-1 (any tensor name with `layers.<i>.`), experts 0..E-1 (`experts.<j>.`; router rows and selection biases whose
first dimension is the expert count are cut to their first E rows), and every other tensor except the ones matching a --drop pattern
(default: vision towers, projectors and multi-token-prediction layers). --rows keeps the first COUNT rows of matching tensors (a prefix
of the stored bytes, so quantized tensors stay as stored; row-wise scales must be listed too). In config.json (and its text_config) the
layer count becomes N, per-layer lists at least as long as the original layer count keep their first N entries, and the expert counts
become E; --set overrides any value afterwards (dotted path, JSON value). The small files of the repository (tokenizer, templates,
remote code, hf_quant_config.json) are copied. Tensors are written to one model.safetensors with their stored dtype and bytes.
HF_TOKEN is used when set.
"""
import json
import os
import re
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def option(name, many=False, default=None):
    values = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == name]
    return values if many else (values[-1] if values else default)


repo, rev, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
layers = int(option("--layers"))
experts = int(option("--experts")) if option("--experts") else None
drops = option("--drop", many=True) or [r"(^|\.)(vision|visual|vision_tower|vision_model|multi_modal_projector|patch_merge_mlp|mm_projector|aligner)\.",
                                        r"^(model\.)?mtp\.", r"^image_(start|end|newline)$"]
rows = {pattern: int(count) for pattern, count in (r.rsplit("=", 1) for r in option("--rows", many=True))}
sets = [s.split("=", 1) for s in option("--set", many=True)]
token = os.environ.get("HF_TOKEN")
base = f"https://huggingface.co/{repo}/resolve/{rev}/"
out.mkdir(parents=True, exist_ok=True)


def get(name, start=None, end=None):
    request = urllib.request.Request(base + name)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end}")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except Exception:
            if attempt == 5:
                raise


api = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/revision/{rev}")
if token:
    api.add_header("Authorization", f"Bearer {token}")
with urllib.request.urlopen(api, timeout=120) as response:
    files = [s["rfilename"] for s in json.loads(response.read())["siblings"]]
for name in files:
    if "/" not in name and name.endswith((".json", ".jinja", ".txt", ".py", ".model", ".tiktoken")) and name != "model.safetensors.index.json":
        (out / name).write_bytes(get(name))

config = json.loads((out / "config.json").read_text())
sections = [config] + [config[k] for k in ("text_config", "language_config", "llm_config") if isinstance(config.get(k), dict)]
# Without num_hidden_layers (Nemotron-H) the per-layer type lists or Puzzle block configs give the layer count.
original_layers = max(max([s.get("num_hidden_layers", 0)] + [len(s.get(k) or []) for k in ("block_configs", "layers_block_type", "layer_types")])
                      for s in sections)
original_experts = None
for section in sections:
    if "num_hidden_layers" in section:
        section["num_hidden_layers"] = layers
    for key, value in list(section.items()):
        if isinstance(value, list) and original_layers and len(value) >= original_layers and key not in ("architectures",):
            section[key] = value[:layers]
        if isinstance(value, dict) and key.endswith(("attention_config", "rope_parameters")):
            for inner, v in list(value.items()):
                if isinstance(v, list) and len(v) >= original_layers:
                    value[inner] = v[:layers]
    for key in ("n_routed_experts", "num_local_experts", "num_experts", "moe_num_experts"):
        if experts is not None and isinstance(section.get(key), int) and section[key] > experts:
            original_experts = section[key]
            section[key] = experts
    for key in ("num_experts_per_tok", "moe_topk"):
        if experts is not None and isinstance(section.get(key), int):
            section[key] = min(section[key], experts)
for path, value in sets:
    node = config
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = json.loads(value)
(out / "config.json").write_text(json.dumps(config, indent=2))

if "model.safetensors.index.json" in files:
    weight_map = json.loads(get("model.safetensors.index.json"))["weight_map"]
else:
    weight_map = None
shards = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
headers = {}


def read_header(shard):
    size = struct.unpack("<Q", get(shard, 0, 7))[0]
    return shard, 8 + size, json.loads(get(shard, 8, 7 + size))


with ThreadPoolExecutor(16) as pool:
    for shard, offset, header in pool.map(read_header, shards):
        header.pop("__metadata__", None)
        for name, info in header.items():
            headers[name] = (shard, offset, info)


def keep(name):
    if any(re.search(d, name) for d in drops):
        return False
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if m and int(m.group(1)) >= layers:
        return False
    e = re.search(r"\.experts\.(\d+)\.", name)
    return not (e and experts is not None and int(e.group(1)) >= experts)


plan = []
for name in sorted(n for n in headers if keep(n)):
    shard, offset, info = headers[name]
    begin, end = info["data_offsets"]
    shape = list(info["shape"])
    length = end - begin
    count = next((c for p, c in rows.items() if re.search(p, name)), None)
    if count is None and experts is not None and original_experts and shape and shape[0] == original_experts and re.search(r"(gate|router|e_score_correction|expert_bias)", name):
        count = experts
    if count is not None and shape and shape[0] > count:
        length = length // shape[0] * count
        shape[0] = count
    plan.append((name, info["dtype"], shape, shard, offset + begin, length))

header, position = {}, 0
for name, dtype, shape, _, _, length in plan:
    header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [position, position + length]}
    position += length
blob = json.dumps(header).encode()
blob += b" " * ((8 - len(blob) % 8) % 8)


def fetch(item):
    _, _, _, shard, start, length = item
    if length == 0:
        return b""
    chunks = []
    for chunk in range(start, start + length, 1 << 28):
        chunks.append(get(shard, chunk, min(start + length, chunk + (1 << 28)) - 1))
    return b"".join(chunks)


with open(out / "model.safetensors", "wb") as f, ThreadPoolExecutor(8) as pool:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    for batch in range(0, len(plan), 32):
        for data in pool.map(fetch, plan[batch:batch + 32]):
            f.write(data)
print(f"{len(plan)} tensors, {position / 1e9:.2f} GB -> {out}")
