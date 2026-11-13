"""Greedy reference ids from Hugging Face transformers (fp32 compute on the checkpoint weights).

Usage: python hf_greedy_reference.py <model_dir> <new_tokens> <prompt>

Prints the generated ids and, per step, the margin between the best and second-best logit (small margins mean the case is
sensitive to rounding and makes a poor golden reference).
"""
import os
import re
import sys
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

model_dir, new_tokens, prompt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
config = AutoConfig.from_pretrained(model_dir)
model_class = getattr(transformers, config.architectures[0])
model, info = model_class.from_pretrained(model_dir, dtype=torch.float32, output_loading_info=True)
if info["missing_keys"]:
    # Some tiny checkpoints nest the text model several times (model.language_model.language_model.language_model.*), which
    # transformers does not resolve: load the state dict again with the repeated prefix collapsed.
    from safetensors.torch import load_file
    state = load_file(os.path.join(model_dir, "model.safetensors"))
    state = {re.sub(r"(language_model\.)+", "language_model.", k): v.float() for k, v in state.items()}
    # Per-expert tensors → the fused 3-D parameters of the transformers module (from_pretrained converts them itself).
    experts = {}
    for k in [k for k in state if re.search(r"\.experts\.\d+\.", k)]:
        prefix, index, proj = re.match(r"(.*\.experts)\.(\d+)\.(\w+)_proj\.weight$", k).groups()
        experts.setdefault(prefix, {}).setdefault(int(index), {})[proj] = state.pop(k)
    for prefix, by_index in experts.items():
        ordered = [by_index[i] for i in range(len(by_index))]
        state[prefix + ".gate_up_proj"] = torch.stack([torch.cat([e["gate"], e["up"]]) for e in ordered])
        state[prefix + ".down_proj"] = torch.stack([e["down"] for e in ordered])
    result = model.load_state_dict(state, strict=False)
    still_missing = [k for k in result.missing_keys if "rotary" not in k and "visual" not in k]
    if still_missing:
        sys.exit(f"weights missing from the checkpoint: {still_missing[:5]}")
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir)
ids = tokenizer(prompt, return_tensors="pt").input_ids
generated, margins = [], []
with torch.no_grad():
    for _ in range(new_tokens):
        logits = model(input_ids=ids).logits[0, -1]
        top = torch.topk(logits, 2)
        margins.append(float(top.values[0] - top.values[1]))
        next_id = int(top.indices[0])
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
print("prompt ids:", tokenizer(prompt).input_ids)
print("ids:", ",".join(str(i) for i in generated))
print("text:", repr(tokenizer.decode(generated)))
print("min margin: %.5f" % min(margins))
print("margins:", " ".join("%.4f" % m for m in margins))
