"""Writes tiny-random-clef: a Cloudflare Clef checkpoint with a tiny Qwen3.5 text trunk and joint schema head, plus its reference scores.

Usage: python mk_tiny_clef.py <models/hf directory> <model repo directory>

The model repo directory holds joint_schema_model.py of cloudflare/clef-flash; its JointSchemaHead, ClefModel and encode_record give the
reference. The checkpoint has the release layout (config.json and model.safetensors of the trunk, joint_head_config.json,
joint_head.safetensors) with a 4-layer Qwen3.5 trunk (3 Gated DeltaNet layers, 1 gated attention layer, hidden 64, untied LM head) and the
tokenizer of Qwen3.5-0.8B (<models/hf>/Qwen3.5-0.8B). Two records (a choice and a yes/no question decided jointly; a 5-level score) are
scored; reference_scores.json holds, per record, its segments (each tokenized on its own as encode_record does), the token ids, the llama.cpp
decision order of every token and the scores of the options, question by question.
"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

root, repo = os.path.expanduser(sys.argv[1]), sys.argv[2]
sys.path.insert(0, repo)
from joint_schema_model import (SYSTEM_PROMPT, QUESTION_TYPES, ClefModel, JointSchemaHead, collate_records,  # noqa: E402
                                encode_record, question_options, render)

out = os.path.join(root, "tiny-random-clef")
HEAD = {"hidden_size": 64, "width": 32, "routing_layers": 2, "layers": 2, "heads": 4, "feedforward": 48}

os.makedirs(out, exist_ok=True)
tokenizer_dir = os.path.join(root, "Qwen3.5-0.8B")
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
    if os.path.exists(os.path.join(tokenizer_dir, f)):
        shutil.copy(os.path.join(tokenizer_dir, f), out)
tokenizer = AutoTokenizer.from_pretrained(out)

torch.manual_seed(0)
config = Qwen3_5TextConfig(
    vocab_size=248320, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"], full_attention_interval=4,
    linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_value_head_dim=16,
    max_position_embeddings=4096, rms_norm_eps=1e-6, tie_word_embeddings=False, attn_output_gate=True, eos_token_id=248044,
    rope_parameters={"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default", "rope_theta": 10000000,
                     "partial_rotary_factor": 0.25}, dtype="float32")
trunk = Qwen3_5ForCausalLM(config).eval()
head = JointSchemaHead(**HEAD).eval()
with torch.no_grad():
    for n, p in trunk.named_parameters():
        if "norm" in n:
            p.copy_(0.1 * torch.randn_like(p))              # Qwen3.5 RMS norms scale by 1 + w
        elif n.endswith("A_log"):
            p.copy_(torch.log(torch.rand_like(p) * 4 + 1))
        else:
            p.copy_(torch.randn_like(p) * 0.08)
    for n, p in head.named_parameters():
        if p.dim() == 0:
            continue
        p.copy_(1.0 + 0.2 * torch.randn_like(p) if "norm" in n and n.endswith("weight") else torch.randn_like(p) * 0.15)
    head.prior_logit_scale.fill_(0.4)
    head.joint_logit_scale.fill_(1.2)
    head.residual_gate.fill_(-0.3)
trunk.save_pretrained(out, safe_serialization=True)
save_file({k: v.contiguous() for k, v in head.state_dict().items()}, os.path.join(out, "joint_head.safetensors"))
json.dump(HEAD, open(os.path.join(out, "joint_head_config.json"), "w"), indent=2)
model = ClefModel(trunk, head).eval()

records = [
    {"id": "a", "state": {"ticket": "The app crashes when I open the camera.", "plan": "pro"},
     "questions": {"team": {"type": "choice", "instructions": "Which team should handle the ticket?",
                            "criteria": {"mobile": "Mobile apps", "billing": "Payments and invoices", "infra": "Servers and networking"}},
                   "urgent": {"type": "noul", "instructions": "Does the ticket need an answer today?"}}},
    {"id": "b", "state": "The review says: decent food, slow service, nice view.",
     "questions": {"sentiment": {"type": "score", "instructions": "Rate the overall sentiment.",
                                 "criteria": ["bad", "poor", "mixed", "good", "great"]}}},
]


def segments(record):
    """The pieces encode_record tokenizes one by one, with their roles."""
    seg = [{"text": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n", "role": "text"},
           {"text": render(record["state"]), "role": "text"}, {"text": "\n\nSCHEMA FIELDS:\n", "role": "text"}]
    for index, (qid, q) in enumerate(record["questions"].items()):
        seg.append({"text": f"\nFIELD {index + 1}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: ", "role": "text"})
        seg.append({"text": render(q.get("instructions") or str(qid)), "role": "question", "kind": QUESTION_TYPES[q["type"]]})
        seg.append({"text": "\nALLOWED OPTIONS:\n", "role": "text"})
        for oi, (oid, description) in enumerate(question_options(q)):
            semantics = {"option_id": oid}
            if description is not None:
                semantics["description"] = description
            seg += [{"text": f"OPTION {oi + 1}: ", "role": "text"}, {"text": render(semantics), "role": "option"}, {"text": "\n", "role": "text"}]
        seg.append({"text": "END FIELD\n", "role": "text"})
    seg.append({"text": "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:", "role": "text"})
    return seg


prompts = []
for record in records:
    encoded = encode_record(tokenizer, record)
    with torch.no_grad():
        logits = model(collate_records([encoded], 0, torch.device("cpu")))[0]
    seg = segments(record)
    ids, order = [], []
    for s in seg:
        t = tokenizer(s["text"], add_special_tokens=False).input_ids
        ids += t
        order += [0 if s["role"] == "text" else 4 if s["role"] == "option" else 1 + s["kind"]] * len(t)
    assert ids == list(encoded.input_ids)
    scores = [float(v) for q in logits for v in q]
    prompts.append({"segments": seg, "token_ids": ids, "orders": order, "scores": scores,
                    "options_per_question": [len(q.option_spans) for q in encoded.questions]})
    print(record["id"], len(ids), "tokens, scores:", ",".join("%.5f" % s for s in scores))
json.dump({"prompts": prompts}, open(os.path.join(out, "reference_scores.json"), "w"), indent=1)
print("wrote", out)
