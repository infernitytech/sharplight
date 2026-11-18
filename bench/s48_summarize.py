#!/usr/bin/env python3
"""Markdown table of bench/s48_prefix.sh results: per workload the engines with and without prefix caching (one round, default
the last), TTFT p50 / p99, output tok/s, prefix-cache hit rate, and Sharplight / vLLM ratios.

usage: s48_summarize.py <result dir> [round]
"""
import json, os, sys

d = sys.argv[1]
rnd = sys.argv[2] if len(sys.argv) > 2 else None
rows = [("shared", 16), ("shared", 64), ("multiturn", 16), ("rag", 16)]
engines = ["sharplight", "vllm", "sharplight-nocache", "vllm-nocache"]


def load(engine, wl, c):
    rounds = sorted(f for f in os.listdir(d) if f.startswith(f"{engine}_{wl}_c{c}_r") and f.endswith(".json"))
    if rnd:
        rounds = [f for f in rounds if f.endswith(f"_r{rnd}.json")]
    if not rounds:
        return None
    with open(os.path.join(d, rounds[-1])) as f:
        return json.load(f)


print("| workload | chats | engine | prefix cache | TTFT p50 / p99 ms | output tok/s | hit rate | errors |")
print("|---|---:|---|---|---|---:|---:|---:|")
for wl, c in rows:
    for e in engines:
        r = load(e, wl, c)
        if r is None:
            continue
        name, cache = (e.split("-")[0], "off" if e.endswith("nocache") else "on")
        print(f"| {wl} | {c} | {name} | {cache} | {r['ttft_p50_ms']:.0f} / {r['ttft_p99_ms']:.0f} | {r['output_tok_s']:.0f} | "
              f"{100 * r['hit_rate']:.1f} % | {r['errors']} |")

print()
print("| workload | chats | TTFT p50 SL / vLLM (cache on) | tok/s SL / vLLM (cache on) | TTFT p50 cache on / off, SL | vLLM |")
print("|---|---:|---:|---:|---:|---:|")
for wl, c in rows:
    s, v, sn, vn = (load(e, wl, c) for e in engines)
    if not (s and v):
        continue
    ratio = lambda a, b, k: f"{a[k] / b[k]:.2f}x" if a and b and b[k] else "-"
    print(f"| {wl} | {c} | {ratio(s, v, 'ttft_p50_ms')} | {ratio(s, v, 'output_tok_s')} | {ratio(s, sn, 'ttft_p50_ms')} | {ratio(v, vn, 'ttft_p50_ms')} |")
