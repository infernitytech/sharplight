#!/usr/bin/env python3
"""Writes the PyTorch checkpoint fixtures of tests/Sharplight.Import.PyTorch.Tests/Data with the reference implementation (torch.save).

Run with a venv that has torch (CPU is enough):  python tools/gen-torch-fixtures.py <output-dir>

Produces zip-format and legacy-format checkpoints holding every supported dtype, views, shared storages, scalars, an nn.Module
state dict (with its _metadata attribute) and a training-style checkpoint, plus expected.json: the raw little-endian bytes of
every tensor as hex, taken from torch itself.
"""
import collections
import json
import sys
from pathlib import Path

import torch


def tensors():
    g = torch.Generator().manual_seed(7)
    base = torch.randn(4, 6, generator=g)
    flat = torch.arange(10, dtype=torch.float32)
    return collections.OrderedDict(
        [
            ("f32", base),
            ("f16", torch.randn(3, 5, generator=g).half()),
            ("bf16", torch.randn(2, 7, generator=g).bfloat16()),
            ("f64", torch.randn(3, generator=g).double()),
            ("i64", torch.arange(-3, 5, dtype=torch.int64)),
            ("i32", torch.arange(6, dtype=torch.int32).reshape(2, 3)),
            ("i16", torch.arange(-4, 4, dtype=torch.int16)),
            ("i8", torch.arange(-8, 8, dtype=torch.int8).reshape(4, 4)),
            ("u8", torch.arange(0, 250, 25, dtype=torch.uint8)),
            ("bool", torch.tensor([True, False, True, True])),
            ("scalar", torch.tensor(3.5)),
            ("empty", torch.zeros(0, 3)),
            ("view_tail", flat[4:]),
            ("view_rows", base[1:3]),
            ("tied_a", base),
            ("param", torch.nn.Parameter(torch.randn(2, 2, generator=g))),
            ("nested.deep.w", torch.randn(2, 3, 4, generator=g)),
        ]
    )


def expected(state):
    out = {}
    for name, t in state.items():
        t = t.detach().contiguous()
        out[name] = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "hex": t.reshape(-1).view(torch.uint8).numpy().tobytes().hex() if t.numel() else "",
        }
    return out


def flatten(state):
    flat = collections.OrderedDict()
    for k, v in state.items():
        flat[k] = v
    return flat


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    state = tensors()
    nested = collections.OrderedDict((k, v) for k, v in state.items() if not k.startswith("nested"))
    nested_value = state["nested.deep.w"]
    saved = collections.OrderedDict(nested)
    saved["nested"] = {"deep": {"w": nested_value}}
    torch.save(saved, out / "zip.pt")
    torch.save(saved, out / "legacy.pt", _use_new_zipfile_serialization=False)

    module = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.BatchNorm1d(3))
    torch.manual_seed(1)
    for p in module.parameters():
        torch.nn.init.normal_(p)
    torch.save(module.state_dict(), out / "module.bin")
    torch.save({"epoch": 3, "global_step": 120, "state_dict": module.state_dict(), "lr": [0.1]}, out / "training.ckpt")

    exp = {}
    flat = collections.OrderedDict(nested)
    flat["nested.deep.w"] = nested_value
    exp["zip.pt"] = expected(flat)
    exp["legacy.pt"] = expected(flat)
    exp["module.bin"] = expected(module.state_dict())
    exp["training.ckpt"] = expected(module.state_dict())
    (out / "expected.json").write_text(json.dumps(exp))
    print("torch", torch.__version__, "->", out)


if __name__ == "__main__":
    main()
