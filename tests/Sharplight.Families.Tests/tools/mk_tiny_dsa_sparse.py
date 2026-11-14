"""Copies a tiny DSA checkpoint (mk_tiny_decoders.py) with another index_topk and re-initializes its indexer
with more scoring heads, so that index scores are rarely tied (torch.topk breaks ties at the top-k boundary arbitrarily). Tries several seeds and keeps the one whose smallest relative gap between the k-th and (k+1)-th score, over
every row of the prompt plus 24 greedy tokens, is largest (selections then survive f16 key caches).

    python mk_tiny_dsa_sparse.py <source dir> <model dir> <index_topk> <heads> <prompt ids comma separated> <seeds>

The sparse test models use index_topk 8 and 16 heads; the reference ids and prompt logits are then written as for the other tiny models.
"""
import json, os, shutil, sys, torch, transformers
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

src, path, topk, heads = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ids, seeds = [int(x) for x in sys.argv[5].split(",")], int(sys.argv[6])
os.makedirs(path, exist_ok=True)
for n in os.listdir(src):
    if n != "reference_logits.json":
        shutil.copy(os.path.join(src, n), os.path.join(path, n))
cfg_path = os.path.join(path, "config.json")
cfg = json.load(open(cfg_path))
cfg["index_topk"] = topk
cfg["index_n_heads"] = heads
json.dump(cfg, open(cfg_path, "w"), indent=2)
dims = cfg["index_head_dim"]
weights = os.path.join(path, "model.safetensors")
state = load_file(weights)


def initialize(seed):
    gen = torch.Generator().manual_seed(seed)
    out = dict(state)
    for name in list(out):
        if name.endswith("indexer.wq_b.weight"):
            out[name] = torch.randn(heads * dims, out[name].shape[1], generator=gen) * 0.2
        elif name.endswith("indexer.weights_proj.weight"):
            out[name] = torch.randn(heads, out[name].shape[1], generator=gen) * 0.2
    return out


def smallest_gap(model):
    with torch.no_grad():
        full = model.generate(torch.tensor([ids]), max_new_tokens=24, do_sample=False)[0].tolist()
    recorded = []
    original = torch.Tensor.topk

    def recording_topk(self, k, *args, **kwargs):
        if self.dim() == 3 and self.shape[1] == len(full) and self.shape[-1] == len(full):
            recorded.append((self.clone(), k))
        return original(self, k, *args, **kwargs)

    torch.Tensor.topk = recording_topk
    try:
        with torch.no_grad():
            model(torch.tensor([full]))
    finally:
        torch.Tensor.topk = original
    worst = float("inf")
    for scores, k in recorded:
        for t in range(k, len(full)):
            row = scores[0, t, : t + 1].sort(descending=True).values
            worst = min(worst, (row[k - 1] - row[k]).item() / max(row.abs().max().item(), 1e-30))
    return worst


config = AutoConfig.from_pretrained(path)
model = getattr(transformers, config.architectures[0]).from_pretrained(path, dtype=torch.float32, config=config, ignore_mismatched_sizes=True)
best = (-1.0, None)
for seed in range(1, seeds + 1):
    candidate = initialize(seed)
    model.load_state_dict({k: v for k, v in candidate.items() if "indexer" in k}, strict=False)
    gap = smallest_gap(model)
    print(f"seed {seed}: smallest relative boundary gap {gap:.3g}")
    if gap > best[0]:
        best = (gap, seed)
print("best", best)
save_file(initialize(best[1]), weights, metadata={"format": "pt"})
