#!/usr/bin/env python3
# Tables of docs/benchmarks.md section 38. usage: s38_summarize.py <results dir> <tags, e.g. before,after>
import json, re, sys, glob, os, statistics
R = sys.argv[1]; tags = sys.argv[2].split(",")  # e.g. before,after
models = ["qwen3-0.6b", "llama3.2-1b", "qwen3-8b", "qwen3-8b-fp8"]
def sl_single(tag, m):
    f = f"{R}/single_sl-{tag}_{m}.txt"
    if not os.path.exists(f): return {}
    pp, tg = {}, {}
    for l in open(f):
        x = re.match(r"\[N=(\d+) run=\d+\] prefill (\d+) tok: best \d+ ms = ([\d.]+) tok/s \| decode: best [\d.]+, median ([\d.]+)", l)
        if not x: continue
        n = int(x.group(1)); pp.setdefault(n, []).append(float(x.group(3))); tg.setdefault(n, []).append(float(x.group(4)))
    out = {f"pp{n}": max(v) for n, v in pp.items() if n >= 512}
    if 16 in tg: out["tg@d16"] = statistics.median(tg[16])
    if 8192 in tg: out["tg@d8192"] = statistics.median(tg[8192])
    return out
def vl_single(m):
    f = f"{R}/single_vllm_{m}.json"
    if not os.path.exists(f): return {}
    r = json.load(open(f))["results"]; out = {k: v["tok_s_median"] for k, v in r.items() if k.startswith("pp")}
    for k, v in r.items():
        if k.startswith("tg128@d"): out["tg@d" + k.split("@d")[1]] = v["tok_s_median"]
    return out
def serve(eng, m, c):
    fs = sorted(glob.glob(f"{R}/serve_{eng}_{m}_c{c}_r*.json"))
    if not fs: return None
    v = [json.load(open(f)) for f in fs]
    return statistics.median(x["ttft_p50_ms"] for x in v), statistics.median(x["agg_tok_s"] for x in v)
metrics = ["pp512", "pp2048", "pp8192", "tg@d16", "tg@d8192"]
hdr = "| model | metric | " + " | ".join(f"Sharplight {t}" for t in tags) + " | vLLM | " + " | ".join(f"SL/vLLM {t}" for t in tags) + " |"
print(hdr); print("|---|---|" + "---:|" * (2 * len(tags) + 1))
for m in models:
    v = vl_single(m); s = [sl_single(t, m) for t in tags]
    for k in metrics:
        if k not in v and all(k not in x for x in s): continue
        cells = [f"{x[k]:.0f}" if k in x else "-" for x in s]
        rat = [f"{x[k]/v[k]:.2f}x" if k in x and k in v else "-" for x in s]
        print(f"| {m} | {k} | " + " | ".join(cells) + f" | {v.get(k, float('nan')):.0f} | " + " | ".join(rat) + " |")
print()
print("| model | streams | " + " | ".join(f"TTFT p50 SL {t}" for t in tags) + " | TTFT p50 vLLM | " + " | ".join(f"tok/s SL {t}" for t in tags) + " | tok/s vLLM |")
print("|---|---:|" + "---:|" * (2 * len(tags) + 2))
for m in models:
    for c in (1, 16):
        v = serve("vllm", m, c); s = [serve(f"sl-{t}", m, c) for t in tags]
        if v is None and all(x is None for x in s): continue
        f = lambda x, i, fmt: (fmt % x[i]) if x else "-"
        print(f"| {m} | {c} | " + " | ".join(f(x, 0, "%.0f") for x in s) + f" | {f(v, 0, '%.0f')} | " + " | ".join(f(x, 1, "%.0f") for x in s) + f" | {f(v, 1, '%.0f')} |")
