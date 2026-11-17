#!/usr/bin/env python3
# Table of docs/benchmarks.md section 40.9 (identical procedure: median of 5 prefills on every engine).
# usage: s40_identical_summarize.py <out dir of s40_identical.sh>
import json, os, re, statistics, sys
R = sys.argv[1]
def sl(n):
    out = {}
    p = f"{R}/sl_{n}.txt"
    if os.path.exists(p):
        for l in open(p):
            x = re.match(r"\[N=(\d+)\] .*prefill median of 5: [\d.]+ ms = ([\d.]+) tok/s", l)
            if x: out[int(x.group(1))] = float(x.group(2))
    return out
def vllm(n):
    p = f"{R}/vllm_{n}.json"
    if not os.path.exists(p): return {}
    return {int(k[2:]): v["tok_s_median"] for k, v in json.load(open(p))["results"].items() if k.startswith("pp")}
def llama(n):
    p = f"{R}/llama_{n}.json"
    if not os.path.exists(p): return {}
    out = {}
    for r in json.load(open(p)):
        if r.get("n_gen", 0) == 0 and r.get("n_prompt", 0) > 0:
            out[r["n_prompt"]] = r["n_prompt"] / (statistics.median(r["samples_ns"]) * 1e-9)
    return out
print("| model | weights | prompt | Sharplight | other engine | other | Sharplight / other |")
print("|---|---|---:|---:|---|---:|---:|")
for n, w, f, eng in [("qwen3-0.6b", "HF bf16", vllm, "vLLM"), ("llama3.2-1b", "HF bf16", vllm, "vLLM"), ("qwen3-8b", "HF bf16", vllm, "vLLM"),
                     ("qwen3-8b-fp8", "HF FP8", vllm, "vLLM"), ("qwen3-0.6b-q8_0", "GGUF Q8_0", llama, "llama.cpp"),
                     ("llama3.2-1b-q8_0", "GGUF Q8_0", llama, "llama.cpp"), ("qwen3-8b-q4_k_m", "GGUF Q4_K_M", llama, "llama.cpp")]:
    s, o = sl(n), f(n)
    for N in (512, 2048, 8192):
        a, b = s.get(N), o.get(N)
        print(f"| {n.split('-q')[0] if 'gguf' in w.lower() else n} | {w} | {N} | {a:.0f} | {eng} | {b:.0f} | {a / b:.2f}x |" if a and b
              else f"| {n} | {w} | {N} | {a or '-'} | {eng} | {b or '-'} | - |")
