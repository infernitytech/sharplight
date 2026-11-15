"""
Transformers reference for a published-layout Kimi K3 checkpoint as it is (e.g. tiny-random/kimi-k3: bfloat16 weights, routed experts in
compressed-tensors MXFP4, Kimi's tiktoken tokenizer): the prompt is encoded with the checkpoint's own remote tokenizer
(tokenization_kimi.py), the MXFP4 experts are decompressed with compressed_tensors' own helpers, and the text model runs in float32 through
kimi_k3_reference.py. Prints the prompt ids, the greedy ids and per-step margins; with --write it stores the prompt logits next to the
checkpoint (reference_logits.json).

    python kimi_k3_checkpoint_reference.py <checkpoint directory> [--write]
"""
import json, os, sys
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from compressed_tensors.compressors.mx_utils import decompress_mx_scale
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimi_k3_reference import load_text_model

PROMPT = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"
NEW_TOKENS = 24

d = os.path.expanduser(sys.argv[1])
cfg = json.load(open(os.path.join(d, "config.json")))["text_config"]
cfg = {k: v for k, v in cfg.items() if k != "quantization_config"}
raw = {k[len("language_model."):]: v for k, v in load_file(os.path.join(d, "model.safetensors")).items() if k.startswith("language_model.")}
state = {}
for name, value in raw.items():
    if name.endswith(".weight_scale"):
        continue
    if name.endswith(".weight_packed"):
        stem = name[: -len("_packed")]
        rows, columns = value.shape[0], value.shape[1] * 2
        codes = unpack_fp4_from_uint8(value, rows, columns, dtype=torch.float32)
        scale = decompress_mx_scale(raw[stem + "_scale"]).float()                    # [rows, columns / 32]
        state[stem] = codes * scale.repeat_interleave(32, dim=1)
        continue
    state[name] = value.float()

tokenizer = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
prompt_ids = tokenizer(PROMPT).input_ids
model = load_text_model(d, cfg, state)
ids = torch.tensor([prompt_ids])
generated, margins, prompt_logits = [], [], None
with torch.no_grad():
    for _ in range(NEW_TOKENS):
        logits = model(input_ids=ids, use_cache=False).logits[0, -1]
        prompt_logits = logits.tolist() if prompt_logits is None else prompt_logits
        top = torch.topk(logits, 2)
        margins.append(float(top.values[0] - top.values[1]))
        generated.append(int(top.indices[0]))
        ids = torch.cat([ids, top.indices[:1].view(1, 1)], dim=1)

print("prompt ids:", ",".join(str(i) for i in prompt_ids))
print("ids:", ",".join(str(i) for i in generated))
print("text:", repr(tokenizer.decode(generated)))
print("min margin: %.5f" % min(margins))
print("margins:", " ".join("%.4f" % m for m in margins))
if "--write" in sys.argv[2:]:
    json.dump({"prompt": PROMPT, "prompt_ids": prompt_ids, "logits": prompt_logits}, open(os.path.join(d, "reference_logits.json"), "w"))
