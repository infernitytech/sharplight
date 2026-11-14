"""Copies a tiny checkpoint whose indexer selects groups of keys (mk_tiny_minimax_m3.py, mk_tiny_glm5_next.py) with a small selection budget,
so that the test prompt runs in the indexer's sparse regime, and re-initializes the indexer weights. Tries several seeds and keeps the one whose
smallest relative gap between the last picked and the first dropped group score, over every row of the prompt plus 24 greedy tokens, is largest
(torch.topk breaks ties arbitrarily, and the selection should survive f16 key caches).

    python mk_tiny_group_sparse.py <source dir> <model dir> <prompt> <seeds> key=value ...

The key=value pairs override the text config (e.g. index_block_size=4 index_topk_blocks=3 for MiniMax-M3, index_topk=8 index_kpool=4
index_n_heads=16 for GLM-5.3-Flash). The sparse test models are
  tiny-random-minimax-m3-sparse   index_block_size=4 index_topk_blocks=3 index_local_blocks=1
  tiny-random-glm5-next-sparse    index_topk=8 index_kpool=4 index_n_heads=16
Reference ids and logits: hf_greedy_reference.py --logits; GGUF: convert_tiny_gguf.py.
"""
import json, os, shutil, sys, torch, transformers
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

src, path, prompt, seeds = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
overrides = {k: json.loads(v) for k, v in (a.split("=", 1) for a in sys.argv[5:])}
os.makedirs(path, exist_ok=True)
for n in os.listdir(src):
    if n != "reference_logits.json":
        shutil.copy(os.path.join(src, n), os.path.join(path, n))
cfg_path = os.path.join(path, "config.json")
cfg = json.load(open(cfg_path))
text = cfg.get("text_config", cfg)
text.update(overrides)
json.dump(cfg, open(cfg_path, "w"), indent=2)
heads, dims, hidden = text["index_n_heads"], text["index_head_dim"], text["hidden_size"]
weights = os.path.join(path, "model.safetensors")
state = load_file(weights)
ids = AutoTokenizer.from_pretrained(path)(prompt)["input_ids"]


def is_indexer(name):
    return ".indexer." in name or ".self_attn.index_" in name


def initialize(seed):
    gen = torch.Generator().manual_seed(seed)
    out = dict(state)
    for name in list(out):
        if not is_indexer(name) or "norm" in name:
            continue
        if name.endswith(("wq_b.weight", "index_q_proj.weight")):
            out[name] = torch.randn(heads * dims, out[name].shape[1], generator=gen) * 0.3
        elif name.endswith("weights_proj.weight"):
            out[name] = torch.randn(heads, hidden, generator=gen) * 0.3
        elif name.endswith("index_kpool_compress_ape"):
            out[name] = torch.randn(text["index_kpool"], dims, generator=gen) * 0.5
        else:
            out[name] = torch.randn(out[name].shape, generator=gen) * 0.3
    return out


def smallest_gap(model):
    with torch.no_grad():
        full = model.generate(torch.tensor([ids]), max_new_tokens=24, do_sample=False)[0].tolist()
    recorded = []
    original = torch.Tensor.topk

    def recording_topk(self, k, *args, **kwargs):
        if self.dim() >= 3 and self.shape[-2] == len(full):
            recorded.append((self.detach().clone(), k))
        return original(self, k, *args, **kwargs)

    torch.Tensor.topk = recording_topk
    try:
        with torch.no_grad():
            model(torch.tensor([full]))
    finally:
        torch.Tensor.topk = original
    worst = float("inf")
    for scores, k in recorded:
        for row in scores.reshape(-1, scores.shape[-1]):
            forced = int(torch.isposinf(row).sum())
            finite = row[torch.isfinite(row) & (row > -1e30)].sort(descending=True).values
            need = k - forced
            if need <= 0 or finite.numel() <= need:
                continue
            worst = min(worst, (finite[need - 1] - finite[need]).item() / max(finite.abs().max().item(), 1e-30))
    return worst


config = AutoConfig.from_pretrained(path)
best = (-1.0, None)
for seed in range(1, seeds + 1):
    # Written and loaded again, so that transformers maps the checkpoint layout onto its modules.
    save_file(initialize(seed), weights, metadata={"format": "pt"})
    model = getattr(transformers, config.architectures[0]).from_pretrained(path, dtype=torch.float32, config=config)
    gap = smallest_gap(model)
    print(f"seed {seed}: smallest relative boundary gap {gap:.3g}")
    if gap > best[0]:
        best = (gap, seed)
print("best", best)
save_file(initialize(best[1]), weights, metadata={"format": "pt"})
