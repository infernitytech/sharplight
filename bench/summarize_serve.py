#!/usr/bin/env python3
"""Markdown tables for the batched serving benchmark from bench/results/serve_<engine>_<model>_c<N>.json."""
import glob, json, os, re
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rows = {}
for p in glob.glob(R + "/serve_*_c*.json"):
    m = re.match(r"serve_(sharplight|llama-cuda13|llama-vulkan|vllm)_(.+)_c(\d+)\.json", os.path.basename(p))
    if m and not m[2].startswith("test"): rows.setdefault(m[2], {})[(m[1], int(m[3]))] = json.load(open(p))
ENG = ["sharplight", "llama-cuda13", "llama-vulkan", "vllm"]
for model, d in sorted(rows.items()):
    print(f"\n#### {model}\n")
    print("| conc | engine | agg tok/s | per-stream tok/s | TTFT p50 ms | TTFT p95 ms | ITL mean ms | ITL p95 ms | avg out tok | errors |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in (1, 4, 16, 64):
        for e in ENG:
            j = d.get((e, c))
            if j: print(f"| {c} | {e} | {j['agg_tok_s']:.0f} | {j['per_stream_tok_s']:.1f} | {j['ttft_p50_ms']:.0f} | {j['ttft_p95_ms']:.0f} | {j['itl_mean_ms']:.1f} | {j['itl_p95_ms']:.1f} | {j['avg_out_tokens']:.0f} | {j['errors']} |")
