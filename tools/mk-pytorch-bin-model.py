#!/usr/bin/env python3
"""Re-saves a safetensors Hugging Face model directory as PyTorch checkpoints for the HF importer's pytorch_model*.bin tests.

  python tools/mk-pytorch-bin-model.py <source-dir> <single-bin-dir> <sharded-bin-dir>

Both outputs hold every file of the source except *.safetensors; weights go to pytorch_model.bin (zip format) or, sharded
(two shards plus pytorch_model.bin.index.json, the second shard in the legacy stream format), as torch.save writes them.
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    src, single, sharded = (Path(a) for a in sys.argv[1:4])
    tensors = load_file(str(src / "model.safetensors"))
    for out in (single, sharded):
        out.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.suffix != ".safetensors":
                shutil.copy(f, out / f.name)
    torch.save(tensors, single / "pytorch_model.bin")
    names = sorted(tensors)
    half = len(names) // 2
    parts = [names[:half], names[half:]]
    weight_map = {}
    for i, part in enumerate(parts, 1):
        shard = f"pytorch_model-{i:05d}-of-00002.bin"
        torch.save({n: tensors[n] for n in part}, sharded / shard, _use_new_zipfile_serialization=(i == 1))
        weight_map.update({n: shard for n in part})
    (sharded / "pytorch_model.bin.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


if __name__ == "__main__":
    main()
