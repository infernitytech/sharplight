#!/usr/bin/env python3
"""summarize.py <results folder>: one markdown table per case from run-bench.sh's output (medians of the timed runs, the per-stage split,
peak device memory from nvidia-smi and the PSNR of the parity run against PyTorch's output)."""
import glob
import json
import os
import re
import statistics
import sys

folder = sys.argv[1]


def peak(log):
    values = [int(m) for m in re.findall(r"peak_mib (\d+)", open(log).read())] if os.path.exists(log) else []
    return values[-1] / 1024 if values else float("nan")


def worst(log):
    found = re.findall(r"worst ([0-9.]+) dB", open(log).read()) if os.path.exists(log) else []
    return found[-1] if found else "-"


cases = sorted({os.path.basename(p)[:-len("-pytorch.log")] for p in glob.glob(os.path.join(folder, "*-pytorch.log"))})
for case in cases:
    print(f"\n{case}\n")
    print("| implementation | total | read | encode (resize + VAE encoder) | DiT | decode (VAE decoder + colour fix) | write | peak memory |")
    print("|---|---|---|---|---|---|---|---|")
    torch_json = os.path.join(folder, case + "-pytorch.json")
    if os.path.exists(torch_json):
        runs = json.load(open(torch_json))["runs"][1:]
        m = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
        encode = m["preprocess"] + m["vae_encode"]
        decode = m["vae_decode"] + m["color_fix"]
        print(f"| PyTorch (official, CUDA) | {m['total']:.2f} s | {m['read']:.2f} s | {encode:.2f} s | {m['dit']:.2f} s | {decode:.2f} s "
              f"| {m['write']:.2f} s | {peak(os.path.join(folder, case + '-pytorch.log')):.1f} GiB |")
    for provider in ("cuda", "vulkan"):
        path = os.path.join(folder, f"{case}-sharplight-{provider}.json")
        if not os.path.exists(path):
            continue
        m = json.load(open(path))["median"]
        psnr = worst(os.path.join(folder, f"{case}-sharplight-{provider}-parity.log"))
        print(f"| Sharplight {provider} (worst frame vs PyTorch {psnr} dB) | {m['total']:.2f} s | {m['read']:.2f} s | {m['encoding']:.2f} s "
              f"| {m['restoring']:.2f} s | {m['decoding']:.2f} s | {m['write']:.2f} s "
              f"| {peak(os.path.join(folder, f'{case}-sharplight-{provider}.log')):.1f} GiB |")
