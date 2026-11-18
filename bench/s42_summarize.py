#!/usr/bin/env python3
# Gap table of docs/benchmarks.md section 42 from the output of bench/s42_gap.sh (median of 5 prefills on every engine; decode
# medians). usage: s42_summarize.py <out dir> <name:engine:weights> ...   (engine: vllm or llama)
import json, os, re, statistics, sys
R = sys.argv[1]

def sl(n):
    pp, tg = {}, {}
    p = f"{R}/sl_{n}.txt"
    if os.path.exists(p):
        for l in open(p):
            x = re.match(r"\[N=(\d+)\] .*decode: best [\d.]+, median ([\d.]+).*prefill median of 5: [\d.]+ ms = ([\d.]+) tok/s", l)
            if x:
                pp[int(x.group(1))] = float(x.group(3))
                tg[int(x.group(1))] = float(x.group(2))
    return pp, tg

def vllm(n):
    p = f"{R}/vllm_{n}.json"
    if not os.path.exists(p):
        return {}, {}
    res = json.load(open(p))["results"]
    pp = {int(k[2:]): v["tok_s_median"] for k, v in res.items() if k.startswith("pp")}
    tg = {int(k.split("@d")[1]): v["tok_s_median"] for k, v in res.items() if k.startswith("tg")}
    return pp, tg

def llama(n):
    p = f"{R}/llama_{n}.json"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return {}, {}
    pp, tg = {}, {}
    for r in json.load(open(p)):
        med = statistics.median(r["samples_ns"]) * 1e-9
        if r.get("n_gen", 0) == 0:
            pp[r["n_prompt"]] = r["n_prompt"] / med
        else:
            tg[0] = r["n_gen"] / med
    return pp, tg

def fmt(v):
    return f"{v:.0f}" if v is not None else "-"

def ratio(a, b):
    return f"{a / b:.2f}x" if a and b else "-"

print("| model | weights | other engine | metric | Sharplight | other | Sharplight / other |")
print("|---|---|---|---|---:|---:|---:|")
for spec in sys.argv[2:]:
    n, eng, w = spec.split(":")
    spp, stg = sl(n)
    opp, otg = (vllm if eng == "vllm" else llama)(n)
    label = "vLLM" if eng == "vllm" else "llama.cpp"
    for N in sorted(set(spp) | set(opp)):
        print(f"| {n} | {w} | {label} | pp{N} | {fmt(spp.get(N))} | {fmt(opp.get(N))} | {ratio(spp.get(N), opp.get(N))} |")
    short = min(stg) if stg else None
    if eng == "vllm":
        pairs = [(f"tg@d16 (SL: after {short})", stg.get(short), otg.get(16))] + [(f"tg@d{d}", stg.get(d), otg.get(d)) for d in sorted(otg) if d != 16 and d in stg]
    else:
        pairs = [(f"tg128 (SL: after {short})", stg.get(short), otg.get(0))]
    for m, a, b in pairs:
        print(f"| {n} | {w} | {label} | {m} | {fmt(a)} | {fmt(b)} | {ratio(a, b)} |")
