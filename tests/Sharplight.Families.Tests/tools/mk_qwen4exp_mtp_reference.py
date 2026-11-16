#!/usr/bin/env python3
"""Writes the reference drafts of the hyper-connected MTP head of tiny-random-qwen3.8-flash-next (HF qwen4_exp) for the speculative-decoding
tests: <model>/qwen4exp_mtp_reference.json, a list of {prompt, pending, drafts}.

transformers ignores the checkpoint's mtp.* tensors, so the head is assembled here from transformers' own modules, following llama.cpp's
qwen4exp graph_mtp (src/models/qwen4exp.cpp): the target's hidden state is its residual [T, hc * hidden] before the final
hyper_connection_mixer (captured with a hook); row i reads e = pre_fc_norm_embedding(embed(token after i)) and the per-stream
h = pre_fc_norm_hidden(hidden_i), every stream s starts as fc_embedding(e) + fc_hidden(h_s), one Qwen4ExpTextDecoderLayer (full attention
with the QSA indexer, hyper-connected) runs over the rows, mtp.hyper_connection_mixer mixes the streams and the target's lm_head gives the
draft logits. The last prompt row (token = the target's greedy next token, "pending") gives the first draft; every further draft appends a row
that reads the previous draft and the head's own residual of the previous row. Three drafts per prompt; the minimum top-2 logit gap is printed.

Usage: mk_qwen4exp_mtp_reference.py [MODEL_DIR]   (default ~/models/hf/tiny-random-qwen3.8-flash-next)
Needs torch, transformers (with qwen4_exp) and safetensors.
"""
import copy
import json
import os
import sys

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText
from transformers.models.qwen4_exp import modeling_qwen4_exp as q4

HOME = os.path.expanduser("~")
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, "models/hf/tiny-random-qwen3.8-flash-next")
DRAFTS = 3
PROMPTS = [[11, 22, 33, 44, 55, 66, 77, 88, 99], [1001, 2002, 3003, 4004, 5005], [7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8]]

torch.manual_seed(0)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").eval()
text = model.model.language_model
cfg = text.config
hidden_size, streams = cfg.hidden_size, cfg.hc_count

with safe_open(os.path.join(MODEL, "model.safetensors"), "pt") as f:
    mtp = {k[len("mtp."):]: f.get_tensor(k).float() for k in f.keys() if k.startswith("mtp.")}

layer_cfg = copy.deepcopy(cfg)
layer_cfg.layer_types = list(cfg.layer_types) + list(cfg.mtp["layer_types"])
layer_cfg.num_hidden_layers = len(layer_cfg.layer_types)
layer = q4.Qwen4ExpTextDecoderLayer(layer_cfg, cfg.num_hidden_layers).eval()
layer.load_state_dict({k[len("layers.0."):]: v for k, v in mtp.items() if k.startswith("layers.0.")}, strict=True)
mixer = q4.Qwen4ExpTextGatedResidual(cfg, use_combine=False).eval()
mixer.load_state_dict({k[len("hyper_connection_mixer."):]: v for k, v in mtp.items() if k.startswith("hyper_connection_mixer.")}, strict=True)
norm_e = q4.Qwen4ExpTextRMSNorm(hidden_size, eps=cfg.rms_norm_eps)
norm_e.load_state_dict({"weight": mtp["pre_fc_norm_embedding.weight"]})
norm_h = q4.Qwen4ExpTextRMSNorm(hidden_size * streams, group_size=hidden_size, eps=cfg.rms_norm_eps)
norm_h.load_state_dict({"weight": mtp["pre_fc_norm_hidden.weight"]})
fc_e, fc_h = mtp["fc_embedding.weight"], mtp["fc_hidden.weight"]
embed, lm_head = text.embed_tokens.weight, model.lm_head.weight

captured = {}
text.hyper_connection_mixer.register_forward_pre_hook(lambda _m, args: captured.__setitem__("streams", args[0].detach()))


def head(rows_hidden, rows_tokens):
    """Runs the head over all rows (causal); returns the logits and the residual of every row."""
    n = rows_tokens.shape[0]
    e = norm_e(embed[rows_tokens])                                                   # [n, H]
    h = norm_h(rows_hidden).view(n, streams, hidden_size)                            # [n, hc, H]
    x = (h @ fc_h.T + (e @ fc_e.T)[:, None, :]).reshape(1, n, streams * hidden_size)
    positions = torch.arange(n).view(1, 1, n).expand(3, 1, n)
    position_embeddings = text.rotary_emb(x, positions)
    mask = torch.ones(n, n, dtype=torch.bool).tril().view(1, 1, n, n)
    out = layer(x, position_embeddings=position_embeddings, attention_mask=mask)
    return mixer(out)[0] @ lm_head.T, out[0]


cases, gaps = [], []
with torch.no_grad():
    for prompt in PROMPTS:
        ids = torch.tensor([prompt])
        logits = model(input_ids=ids).logits[0]
        pending = int(logits[-1].argmax())
        hidden = captured["streams"][0]                                              # [T, hc * H]
        tokens = torch.tensor(prompt[1:] + [pending])
        drafts = []
        for _ in range(DRAFTS):
            draft_logits, residual = head(hidden, tokens)
            top = draft_logits[-1].topk(2).values
            gaps.append(float(top[0] - top[1]))
            draft = int(draft_logits[-1].argmax())
            drafts.append(draft)
            hidden = torch.cat([hidden, residual[-1:]])
            tokens = torch.cat([tokens, torch.tensor([draft])])
        cases.append({"prompt": prompt, "pending": pending, "drafts": drafts})

with open(os.path.join(MODEL, "qwen4exp_mtp_reference.json"), "w") as out:
    json.dump(cases, out, indent=1)
print(json.dumps(cases))
print(f"minimum top-2 logit gap {min(gaps):.3g}")
