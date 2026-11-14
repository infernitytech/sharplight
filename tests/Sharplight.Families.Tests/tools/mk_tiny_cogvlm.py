"""Tiny random CogVLM (THUDM/cogvlm-chat-hf layout) and its transformers references.

Usage: python mk_tiny_cogvlm.py <remote_code_dir> <tokenizer_dir> <out_dir>

<remote_code_dir> holds the checkpoint's remote code (configuration_cogvlm.py, modeling_cogvlm.py, visual.py, util.py of
THUDM/cogvlm-chat-hf); it is imported as a package, with xformers' memory_efficient_attention replaced by PyTorch's scaled dot-product
attention (the tower attends without a mask, so the two agree). <tokenizer_dir> holds the Llama-2 / Vicuna tokenizer the checkpoint uses
(tokenizer.model, tokenizer_config.json, special_tokens_map.json; lmsys/vicuna-7b-v1.5).

Writes <out_dir> (config.json, model.safetensors, the tokenizer) and two references computed by the remote code in fp32 without a KV cache:
reference_logits.json (prompt logits of a text-only prompt; the greedy ids are printed for the C# case list) and reference_image.json (the
format of hf_image_reference.py: the tower output of the synthetic normalized image of the C# tests fed without resizing, the top logits of an
image prompt and its greedy continuation). The image prompt follows build_conversation_input_ids: bos, the image's rows (begin, patches, end;
"prompt_ids" holds one <unk> placeholder for them), "Question: ... Answer:"; token types and position ids are the remote code's.
"""
import importlib
import json
import math
import os
import shutil
import sys
import types

import torch

remote_dir, tokenizer_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]

# xformers is only used for the tower's unmasked attention ([B, L, heads, D] layout).
xformers = types.ModuleType("xformers")
xformers.ops = types.ModuleType("xformers.ops")


def memory_efficient_attention(q, k, v, scale=None):
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale).transpose(1, 2)


xformers.ops.memory_efficient_attention = memory_efficient_attention
sys.modules["xformers"] = xformers
sys.modules["xformers.ops"] = xformers.ops

package = types.ModuleType("cogvlm_remote")
package.__path__ = [os.path.abspath(remote_dir)]
sys.modules["cogvlm_remote"] = package
configuration = importlib.import_module("cogvlm_remote.configuration_cogvlm")
modeling = importlib.import_module("cogvlm_remote.modeling_cogvlm")

torch.manual_seed(66)
config = configuration.CogVLMConfig(
    vocab_size=32000, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=2, rms_norm_eps=1e-5,
    template_version="chat", max_position_embeddings=2048,
    vision_config={"dropout_prob": 0.0, "hidden_act": "gelu", "hidden_size": 32, "image_size": 56, "in_channels": 3, "intermediate_size": 48,
                   "layer_norm_eps": 1e-6, "num_heads": 2, "num_hidden_layers": 2, "num_positions": 17, "patch_size": 14})
config.architectures = ["CogVLMForCausalLM"]
model = modeling.CogVLMForCausalLM(config).float().eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.dim() == 1 and ("norm" in name or "layernorm" in name):
            # Norm weights around 1, biases around 0.
            p.copy_(1 + 0.2 * torch.randn_like(p) if name.endswith("weight") else 0.2 * torch.randn_like(p))
        elif name.endswith(("boi", "eoi", "cls_embedding")) or "position_embedding" in name:
            p.normal_(0, 1)
        elif p.dim() == 1:
            p.normal_(0, 0.2)
        else:
            fan_in = p.shape[1] if p.dim() == 2 else p[0].numel()
            # Large language-model weights give clear greedy margins; the attention projections stay moderate, so that the softmax is not so
            # peaked that an f16 KV cache flips a step.
            scale = 1.5 if name.startswith("model.vision.") else 3.0 if "query_key_value" in name else 7.0
            p.normal_(0, scale / math.sqrt(fan_in))

os.makedirs(out, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
for f in os.listdir(tokenizer_dir):
    if f.startswith("tokenizer") or f in ("special_tokens_map.json", "generation_config.json"):
        shutil.copy(os.path.join(tokenizer_dir, f), out)

from transformers import AutoTokenizer  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained(out)
tokenizer.save_pretrained(out)   # tokenizer.json (the fast tokenizer converted from tokenizer.model)


def greedy(input_ids, token_types, images, steps):
    """Full forward per step (no cache): the prompt logits and the greedy ids."""
    ids, types_, generated, margins, first = input_ids, token_types, [], [], None
    with torch.no_grad():
        for _ in range(steps):
            position_ids = modeling.build_position_ids(types_)
            logits = model(input_ids=ids, token_type_ids=types_, images=images, position_ids=position_ids, use_cache=False).logits[0, -1]
            first = logits if first is None else first
            top = torch.topk(logits, 2)
            margins.append(float(top.values[0] - top.values[1]))
            nxt = int(top.indices[0])
            generated.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], 1)
            types_ = torch.cat([types_, torch.tensor([[modeling.LANGUAGE_TOKEN_TYPE]])], 1)
    return first, generated, margins


# Text only: what FamilyGenerationTests feeds (the tokenizer's encoding of the prompt, bos included).
prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"
text_ids = tokenizer(prompt).input_ids
first, ids, margins = greedy(torch.tensor([text_ids]), torch.zeros(1, len(text_ids), dtype=torch.long), None, 24)
with open(os.path.join(out, "reference_logits.json"), "w") as f:
    json.dump({"prompt": prompt, "prompt_ids": text_ids, "logits": first.tolist()}, f)
print("text prompt ids:", text_ids)
print("text greedy:", ",".join(map(str, ids)))
print("text margins:", " ".join("%.4f" % m for m in margins))

# Image: the synthetic normalized image of the C# tests, at the tower's fixed size.
side = config.vision_config["image_size"]
y, x, c = torch.meshgrid(torch.arange(side), torch.arange(side), torch.arange(3), indexing="ij")
pixels = torch.sin(0.031 * x + 0.017 * (c + 1) * y + 0.5 * c).float().permute(2, 0, 1).unsqueeze(0)   # [1, 3, H, W]
with torch.no_grad():
    embeddings = model.model.vision(pixels)[0]                                                           # [boi + patches + eoi, hidden]
rows = embeddings.shape[0]
question = tokenizer.encode(modeling._history_to_prompt("chat", [], "Describe this image."), add_special_tokens=False)
input_ids = [tokenizer.bos_token_id] + [tokenizer.pad_token_id] * rows + question
token_types = [modeling.LANGUAGE_TOKEN_TYPE] + [modeling.VISION_TOKEN_TYPE] * rows + [modeling.LANGUAGE_TOKEN_TYPE] * len(question)
first, ids, margins = greedy(torch.tensor([input_ids]), torch.tensor([token_types]), [[pixels[0]]], 16)
top = torch.topk(first, 10)
result = {
    "width": side, "height": side, "prompt_ids": [tokenizer.bos_token_id, tokenizer.unk_token_id] + question, "image_tokens": rows,
    "embeddings": [round(v, 7) for v in embeddings.reshape(-1).tolist()],
    "top_ids": top.indices.tolist(), "top_logits": [round(v, 5) for v in top.values.tolist()], "greedy": ids,
}
with open(os.path.join(out, "reference_image.json"), "w") as f:
    json.dump(result, f)
print("image rows:", rows, "top:", top.indices[:5].tolist(), "greedy:", ids)
print("image margins:", " ".join("%.4f" % m for m in margins))
