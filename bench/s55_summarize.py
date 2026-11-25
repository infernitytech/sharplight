#!/usr/bin/env python3
"""Markdown table of bench/s55_step_snapshots.sh results: per provider, workload and round the three settings (snapshots kept inside a
step, copied after a step, no prefix caching) with TTFT p50 / p99, output tok/s and prefix-cache hit rate.

usage: s55_summarize.py <result dir>
"""
import json, os, sys

d = sys.argv[1]
settings = [("inside", "in-step"), ("copy", "copied, chunk cut"), ("off", "no prefix cache")]


def load(name):
    path = os.path.join(d, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


print("| provider | workload | round | " + " | ".join(label for _, label in settings) + " | in-step / copied TTFT p50 |")
print("|---|---|---:|" + "---|" * len(settings) + "---:|")
for p in ["vulkan", "cuda"]:
    for wl, c in [("multiturn", 16), ("shared", 16)]:
        for r in [1, 2, 3]:
            res = [load(f"{p}-{s}_{wl}_c{c}_r{r}.json") for s, _ in settings]
            if not any(res):
                continue
            cells = [f"{x['ttft_p50_ms']:.0f} / {x['ttft_p99_ms']:.0f}, {x['output_tok_s']:.0f}, {100 * x['hit_rate']:.1f} %" if x else "-" for x in res]
            ratio = f"{res[0]['ttft_p50_ms'] / res[1]['ttft_p50_ms']:.2f}x" if res[0] and res[1] else "-"
            print(f"| {p} | {wl} {c} | {r} | " + " | ".join(cells) + f" | {ratio} |")
