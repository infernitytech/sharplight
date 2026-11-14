"""Writes tiny-random-vllm-decision2: a vllm-sr Decision 2.0 package with a tiny Qwen3 backbone, plus its reference scores.

Usage: python mk_tiny_vllm_decision2.py <models/hf directory> <model repo directory>

The model repo directory is a copy of the Python sources of vllm-sr/Decision-2.0-Kai-0.6B (decision2/_vendor/dev2model/*.py); its
CandidateHead and encode() give the reference. The package has the repo's layout (config.json with decision_format "vllm-sr-decision",
backbone/ with the Qwen3Model checkpoint, decision_head.safetensors, decision_config.json, score_bias.json) and the tokenizer of
Qwen3-0.6B (<models/hf>/Qwen3-0.6B). Three prompts (a choice, a 5-level score and a yes/no question) are scored; reference_scores.json
holds, per prompt, its segments (text and role, each tokenized on its own as the reference does), the token ids, the llama.cpp decision
order of every token (the first token of each option is left out of its span, so that adjacent spans stay separate: the head reads only
the last token) and the scores.
"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

root, repo = os.path.expanduser(sys.argv[1]), sys.argv[2]
sys.path.insert(0, os.path.join(repo, "decision2", "_vendor"))
from dev2model.decision_model import CandidateHead, encode, segments  # noqa: E402

out = os.path.join(root, "tiny-random-vllm-decision2")
HEAD_DIM = 24
ORDER = {"noul": 1, "choice": 2, "score": 3}
KIND = {"noul": 0, "choice": 1, "score": 2}

os.makedirs(os.path.join(out, "backbone"), exist_ok=True)
tokenizer_dir = os.path.join(root, "Qwen3-0.6B")
for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
    if os.path.exists(os.path.join(tokenizer_dir, f)):
        shutil.copy(os.path.join(tokenizer_dir, f), out)
tokenizer = AutoTokenizer.from_pretrained(out)

torch.manual_seed(0)
config = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=1000000.0,
                     tie_word_embeddings=True, bos_token_id=151643, eos_token_id=151643, dtype="float32")
backbone = Qwen3Model(config).eval()
head = CandidateHead(config.hidden_size, HEAD_DIM).eval()
with torch.no_grad():
    for module in (backbone, head):
        for n, p in module.named_parameters():
            p.copy_(1.0 + 0.2 * torch.randn_like(p) if p.dim() == 1 and "norm" in n else torch.randn_like(p) * 0.2)
backbone.save_pretrained(os.path.join(out, "backbone"), safe_serialization=True)
save_file({k: v.contiguous() for k, v in head.state_dict().items()}, os.path.join(out, "decision_head.safetensors"))
json.dump({"architecture": "qwen3-text-endpoints-global-query-shared-bilinear-mlp", "head_variant": "shared", "backbone_model_type": "qwen3",
           "prompt_version": "decision2-segmented-options-global-query-v1", "head_dim": HEAD_DIM, "max_options": 255},
          open(os.path.join(out, "decision_config.json"), "w"), indent=2)
json.dump({"format": "tiny", "offsets": {"5": [0.04, 0.2, 0.08, -0.15, -0.17]}}, open(os.path.join(out, "score_bias.json"), "w"), indent=2)
json.dump({"architectures": ["Decision2Model"], "backbone": {"config": "backbone/config.json", "weights": ["backbone/model.safetensors"]},
           "decision_format": "vllm-sr-decision", "decision_weights": {"decision_head": "decision_head.safetensors"}, "format_version": 2,
           "max_input_tokens": 8192, "model_config": "decision_config.json", "model_name": "tiny-random-vllm-decision2",
           "model_type": "decision2"}, open(os.path.join(out, "config.json"), "w"), indent=2)

rows = [
    {"id": "a", "task_type": "choice", "state": {"ticket": "The app crashes when I open the camera.", "plan": "pro"},
     "instructions": "Which team should handle the ticket?",
     "options": [{"key": "mobile", "description": "Mobile apps"}, {"key": "billing", "description": "Payments and invoices"},
                 {"key": "infra", "description": "Servers and networking"}]},
    {"id": "b", "task_type": "score", "state": "The review says: decent food, slow service, nice view.",
     "instructions": "Rate the overall sentiment from 1 (bad) to 5 (great).",
     "options": [{"key": str(i), "description": f"level {i}"} for i in range(1, 6)]},
    {"id": "c", "task_type": "noul", "state": {"temperature_c": 38.4}, "instructions": "Does the patient have a fever?",
     "options": [{"key": "true", "description": "Yes"}, {"key": "false", "description": "No"}]},
]
prompts = []
for row in rows:
    row = dict(row, label=0, family="tiny")
    item = encode(row, tokenizer, 8192)
    ids = torch.tensor([item["ids"]])
    with torch.no_grad():
        hidden = backbone(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        scores = head(hidden[:, item["candidate_positions"]], hidden[:, item["query_position"]])[0]
    prefix, options, suffix = segments(row)
    order = [0] * len(tokenizer.encode(prefix, add_special_tokens=False))
    for option in options:
        n = len(tokenizer.encode(option, add_special_tokens=False))
        order += [0] + [4] * (n - 1)
    order += [ORDER[row["task_type"]]] * len(tokenizer.encode(suffix, add_special_tokens=False))
    assert len(order) == len(item["ids"])
    prompts.append({
        "segments": [{"text": prefix, "role": "text"}] + [{"text": o, "role": "option"} for o in options]
        + [{"text": suffix, "role": "question", "kind": KIND[row["task_type"]]}],
        "token_ids": item["ids"], "orders": order, "scores": [float(s) for s in scores]})
    print(row["id"], len(item["ids"]), "tokens, scores:", ",".join("%.5f" % s for s in scores))
json.dump({"prompts": prompts}, open(os.path.join(out, "reference_scores.json"), "w"), indent=1)
print("wrote", out)
