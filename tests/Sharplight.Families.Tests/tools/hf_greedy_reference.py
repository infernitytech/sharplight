"""Greedy reference ids from Hugging Face transformers (fp32 compute on the checkpoint weights).

Usage: python hf_greedy_reference.py <model_dir> <new_tokens> <prompt> [--logits]

Prints the generated ids and, per step, the margin between the best and second-best logit (small margins mean the case is
sensitive to rounding and makes a poor golden reference). With --logits the logits of the last prompt position are written to
<model_dir>/reference_logits.json ({"prompt", "prompt_ids", "logits"}); FamilyGenerationTests compares them when the file is present.
"""
import json
import os
import re
import sys
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

args = [a for a in sys.argv[1:] if a != "--logits"]
write_logits = "--logits" in sys.argv[1:]
model_dir, new_tokens, prompt = args[0], int(args[1]), args[2]
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
prompt_logits = None
with torch.no_grad():
    for _ in range(new_tokens):
        logits = model(input_ids=ids).logits[0, -1]
        if prompt_logits is None:
            prompt_logits = logits.tolist()
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
if write_logits:
    path = os.path.join(model_dir, "reference_logits.json")
    with open(path, "w") as f:
        json.dump({"prompt": prompt, "prompt_ids": tokenizer(prompt).input_ids, "logits": prompt_logits}, f)
    print("wrote", path)
