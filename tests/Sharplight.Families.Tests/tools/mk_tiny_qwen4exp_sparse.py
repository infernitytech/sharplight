"""
Writes a tiny Qwen3.8-Flash-Next (qwen4_exp) checkpoint whose QSA indexer runs in its sparse regime on the test prompt: the config of
tiny-random-qwen3.8-flash-next with a token budget of 8 (two blocks of 4 plus the incomplete tail), attention head size 64 (so the indexer
rotates 16 of its 128 channels; the mrope sections cover 8 pairs), hidden size 16 (llama.cpp needs the n-gram embedding as wide as the
hidden state), one n-gram head per n-gram size (8-wide table rows, which the Vulkan gather takes) and fresh random weights. Several seeds are tried; the one whose smallest
relative gap between the last picked and the first dropped block score, over every row of the prompt plus 24 greedy tokens, is largest is
kept (the selection should not depend on rounding).

    python mk_tiny_qwen4exp_sparse.py <source dir> <model dir> <prompt> <seeds>

Reference ids and logits: hf_greedy_reference.py --logits; GGUF: convert_tiny_gguf.py.
"""
import json, os, shutil, sys
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

src, path, prompt, seeds = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.makedirs(path, exist_ok=True)
for n in os.listdir(src):
    if n not in ("reference_logits.json", "model.safetensors") and os.path.isfile(os.path.join(src, n)):
        shutil.copy(os.path.join(src, n), os.path.join(path, n))
cfg_path = os.path.join(path, "config.json")
cfg = json.load(open(cfg_path))
text = cfg["text_config"]
text.update({"indexer_budget": 8, "indexer_compress_ratio": 4, "head_dim": 64, "hidden_size": text["ple_embed_dim"], "heads_per_ngram": 1,
             "initializer_range": 0.2})
text["rope_parameters"]["mrope_section"] = [3, 3, 2]
text["dtype"] = "float32"
cfg["vision_config"]["out_hidden_size"] = text["hidden_size"]
# The published config names the indexed layers "full_attention" (transformers reads them as "indexed_attention"; llama.cpp's converter
# only knows the published name).
text["layer_types"] = ["full_attention" if t == "indexed_attention" else t for t in text["layer_types"]]
json.dump(cfg, open(cfg_path, "w"), indent=2)
ids = AutoTokenizer.from_pretrained(path)(prompt)["input_ids"]
config = AutoConfig.from_pretrained(path)
model_class = getattr(transformers, config.architectures[0])


def build(seed):
    torch.manual_seed(seed)
    model = model_class(config).float().eval()
    # Norm weights are stored as offsets from 1 (zero-centred): give them some spread.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or "layernorm" in name:
                p.normal_(0.0, 0.1)
    return model


def smallest_gap(model):
    with torch.no_grad():
        full = model.generate(torch.tensor([ids]), max_new_tokens=24, do_sample=False)[0].tolist()
    gaps = []
    original = torch.Tensor.topk

    def recording_topk(self, k, *args, **kwargs):
        if self.dim() == 1 and self.numel() > k:
            top = self.sort(descending=True).values
            gaps.append((top[k - 1] - top[k]).item() / max(top.abs().max().item(), 1e-30))
        return original(self, k, *args, **kwargs)

    torch.Tensor.topk = recording_topk
    try:
        with torch.no_grad():
            model(torch.tensor([full]))
    finally:
        torch.Tensor.topk = original
    return min(gaps) if gaps else -1.0, len(gaps)


best = (-2.0, None)
for seed in range(1, seeds + 1):
    gap, rows = smallest_gap(build(seed))
    print(f"seed {seed}: smallest relative boundary gap {gap:.3g} over {rows} sparse rows")
    if gap > best[0]:
        best = (gap, seed)
print("best", best)
build(best[1]).save_pretrained(path)
gen = os.path.join(path, "generation_config.json")
if os.path.exists(gen):
    os.remove(gen)
