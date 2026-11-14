#!/usr/bin/env python3
"""Writes a tiny random Gemma 4 target and a tiny random Gemma 4 assistant (MTP drafter) for it (Hugging Face layouts, f32 safetensors) for the
speculative-decoding tests, plus the assistant's reference drafts computed by transformers.

Target (gemma4_text, Gemma4ForCausalLM): hidden 64, 6 layers [sliding, sliding, full, sliding, sliding, full] with the last 2 sharing the KV of
layers 3 (sliding) and 2 (full), head size 16 (sliding) / 32 (full, proportional rope over a quarter of it), 4 heads / 2 KV heads, sliding window
8, per-layer inputs of 8, final logit soft cap, the Gemma 4 tokenizer (default ~/models/hf/gemma-4-E2B-it, 262144 tokens).
Assistant (gemma4_assistant, Gemma4AssistantForCausalLM): hidden 32, 4 layers [sliding, sliding, sliding, full] that compute only queries and
attend the target's KV of its last sliding and last full layer (layers 3 and 2), pre_projection [32, 2*64], post_projection [64, 32], tied
embedding / LM head, and with ASSISTANT_ORDERED=1 (default) the centroid head (use_ordered_embeddings: 2048 centroids, top 32, a random token
ordering).

The reference (<assistant>/assistant_reference.json) is transformers' SinglePositionMultiTokenCandidateGenerator after the prompt: the target
runs the prompt (shared KV states, last hidden state), its argmax is the pending token, and the assistant drafts DRAFTS tokens at the pending
token's position from (target embedding of the last token, last hidden state), every further draft from (previous draft, the assistant's
projected output); "drafts_full_head" are the drafts of the same assistant without its centroid head. The minimum top-2 logit gap is printed.

Usage: mk_tiny_gemma4_assistant.py [TARGET_DIR] [ASSISTANT_DIR] [TOKENIZER_DIR]
       (defaults ~/models/hf/tiny-random-gemma4-assistant-target, ~/models/hf/tiny-random-gemma4-assistant)
GGUF versions: convert_hf_to_gguf.py TARGET_DIR and convert_hf_to_gguf.py ASSISTANT_DIR (the converter drops the centroid head).
ASSISTANT_SEED (default 3) picks the weights. Needs torch and transformers.
"""
import json
import os
import shutil
import sys

import torch
from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4ForCausalLM, Gemma4TextConfig

HOME = os.path.expanduser("~")
TARGET = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, "models/hf/tiny-random-gemma4-assistant-target")
ASSISTANT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HOME, "models/hf/tiny-random-gemma4-assistant")
TOK = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HOME, "models/hf/gemma-4-E2B-it")
ORDERED = os.environ.get("ASSISTANT_ORDERED", "1") == "1"

VOCAB = 262144
DRAFTS = 3
ROPE = {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}
PROMPTS = [[2, 818, 5279, 529, 7001, 563, 9079, 236761, 669, 5279, 529, 9405, 563],
           [2, 3689, 1003, 496, 1694, 3375, 236764, 506, 4209, 7384, 1053, 3069, 532, 2369, 529, 1027, 1730]]

torch.manual_seed(int(os.environ.get("ASSISTANT_SEED", "3")))

target_config = Gemma4TextConfig(
    vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    global_head_dim=32, hidden_activation="gelu_pytorch_tanh", max_position_embeddings=4096, rms_norm_eps=1e-6, rope_parameters=ROPE,
    sliding_window=8, layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"],
    final_logit_softcapping=30.0, vocab_size_per_layer_input=VOCAB, hidden_size_per_layer_input=8, num_kv_shared_layers=2,
    use_double_wide_mlp=False, tie_word_embeddings=True, bos_token_id=2, eos_token_id=1, pad_token_id=0)
target_config.architectures = ["Gemma4ForCausalLM"]
assistant_text = Gemma4TextConfig(
    vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    global_head_dim=32, hidden_activation="gelu_pytorch_tanh", max_position_embeddings=4096, rms_norm_eps=1e-6, rope_parameters=ROPE,
    sliding_window=8, layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    vocab_size_per_layer_input=0, hidden_size_per_layer_input=0, num_kv_shared_layers=4, use_double_wide_mlp=False, tie_word_embeddings=True,
    bos_token_id=2, eos_token_id=1, pad_token_id=0)
assistant_config = Gemma4AssistantConfig(text_config=assistant_text, backbone_hidden_size=64, use_ordered_embeddings=ORDERED,
                                         num_centroids=2048, centroid_intermediate_top_k=32)
assistant_config.architectures = ["Gemma4AssistantForCausalLM"]

target = Gemma4ForCausalLM(target_config).eval()
assistant = Gemma4AssistantForCausalLM(assistant_config).eval()


def randomize(model):
    """Weights with enough spread that the argmax of every step is clear (transformers' init gives nearly uniform logits)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layer_scalar"):
                p.fill_(0.9)
            elif "norm" in name:
                p.copy_(1.0 + torch.randn_like(p) * 0.1)
            elif "embed" in name or "centroids" in name:
                p.copy_(torch.randn_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) * (1.0 / p.shape[-1] ** 0.5))


randomize(target)
randomize(assistant)
if ORDERED:
    assistant.masked_embedding.token_ordering.copy_(torch.randperm(VOCAB))

for directory, model in ((TARGET, target), (ASSISTANT, assistant)):
    os.makedirs(directory, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        if os.path.exists(os.path.join(TOK, name)):
            shutil.copy(os.path.join(TOK, name), os.path.join(directory, name))
# generation_config of the real model asks for sampling; the tiny models decode greedily.
for directory in (TARGET, ASSISTANT):
    path = os.path.join(directory, "generation_config.json")
    if os.path.exists(path):
        os.remove(path)


def gap(logits):
    top = torch.topk(logits.double(), 2).values
    return float(top[0] - top[1])


def draft(prompt, out, pending):
    """The assistant's drafts after the prompt (and the largest logit of each)."""
    global min_gap
    shared = {k: (v[0][:, :, : len(prompt)], v[1][:, :, : len(prompt)]) for k, v in out.shared_kv_states.items()}
    hidden = out.hidden_states[-1][:, -1:]
    token = torch.tensor([[pending]])
    position = torch.tensor([[len(prompt)]])
    drafts, logits = [], []
    for _ in range(DRAFTS):
        with torch.no_grad():
            step = assistant(inputs_embeds=torch.cat([embed(token), hidden], dim=-1), position_ids=position, shared_kv_states=shared,
                             attention_mask=torch.ones(1, len(prompt) + 1, dtype=torch.long))
        row = step.logits[0, -1]
        min_gap = min(min_gap, gap(row))
        token = row.argmax().view(1, 1)
        hidden = step.last_hidden_state
        drafts.append(int(token))
        logits.append(round(float(row.max()), 5))
    return drafts, logits


reference = []
min_gap = float("inf")
embed = target.get_input_embeddings()
for prompt in PROMPTS:
    ids = torch.tensor([prompt])
    with torch.no_grad():
        out = target(ids, output_hidden_states=True, return_shared_kv_states=True, use_cache=True)
    pending = int(out.logits[0, -1].argmax())
    min_gap = min(min_gap, gap(out.logits[0, -1]))
    entry = {"prompt": prompt, "pending": pending}
    entry["drafts"], entry["draft_max_logits"] = draft(prompt, out, pending)
    if ORDERED:
        # The same assistant with its full LM head (what the GGUF converter keeps).
        assistant.config.use_ordered_embeddings = False
        entry["drafts_full_head"], _ = draft(prompt, out, pending)
        assistant.config.use_ordered_embeddings = True
    reference.append(entry)

with open(os.path.join(ASSISTANT, "assistant_reference.json"), "w") as f:
    json.dump(reference, f, indent=1)
print(json.dumps(reference))
print(f"wrote {TARGET} and {ASSISTANT}; minimum logit gap {min_gap:.4f}")
