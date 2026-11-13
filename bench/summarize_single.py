#!/usr/bin/env python3
"""Print single-stream tables (markdown) from bench/results: llama-bench jsonl (samples_ts) and sharplight_single.sh output."""
import glob, json, os, re, statistics, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
def llama(path):
    d = {}
    for l in open(path):
        j = json.loads(l); ts = j["samples_ts"]
        key = ("pp%d" % j["n_prompt"] if j["n_prompt"] else "tg%d" % j["n_gen"], j["n_depth"])
        d[key] = (min(ts), statistics.median(ts), max(ts))
    return d
def sharp(path):
    d = {}
    for l in open(path):
        m = re.match(r"\[N=(\d+) run=(\d+)\] prefill (\d+) tok: best (\d+) ms = ([\d.]+) tok/s \| decode: best ([\d.]+), median ([\d.]+), mean ([\d.]+) tok/s \| GPU ms/step: best ([\d.]+), median ([\d.]+), wall best ([\d.]+)", l)
        if m:
            n = int(m[1]); d.setdefault(n, []).append(dict(pp=float(m[5]), dbest=float(m[6]), dmed=float(m[7]), dmean=float(m[8]), gpu=float(m[10])))
    return d
VLLM = {"qwen27b-q8": "qwen27b-fp8", "qwen3-0.6b-q8": "qwen3-0.6b", "llama3.2-1b-q8": "llama3.2-1b-bf16", "gemma3-1b-q8": "gemma3-1b-bf16",
        "mistral7b-q4km": "mistral7b-bf16", "olmoe-q4km": "olmoe-bf16", "nemotron30b-q4km": "nemotron30b-fp8"}
def vllm(n):
    p = f"{R}/single_vllm_{VLLM.get(n)}.json"
    if not os.path.exists(p): return {}
    r = json.load(open(p))["results"]
    deep = [k for k in r if k.startswith("tg128@d") and k != "tg128@d16"]
    pps = [k for k in r if k.startswith("pp") and k not in ("pp512", "pp2048")]
    return {"pp512": r["pp512"]["tok_s_median"], "pp2048": r["pp2048"]["tok_s_median"], "tg128 @d0": r["tg128@d16"]["tok_s_median"],
            "tg128 @d4096": r[deep[0]]["tok_s_median"] if deep else None}
names = sorted({re.sub(r"single_(llama-\w+|sharplight)_", "", os.path.basename(p))[:-6 if p.endswith(".jsonl") else -4] for p in glob.glob(R + "/single_*.jsonl") + glob.glob(R + "/single_sharplight_*.txt")})
def f(x): return "%.0f" % x if x >= 100 else "%.1f" % x
print("| model | metric | Sharplight med (min..max) | llama.cpp CUDA med (min..max) | llama.cpp Vulkan med (min..max) | vLLM med | SL/CUDA | SL/Vulkan | SL/vLLM |")
print("|---|---|---|---|---|---|---|---|---|")
for n in names:
    cu = llama(f"{R}/single_llama-cuda13_{n}.jsonl") if os.path.exists(f"{R}/single_llama-cuda13_{n}.jsonl") else {}
    vk = llama(f"{R}/single_llama-vulkan_{n}.jsonl") if os.path.exists(f"{R}/single_llama-vulkan_{n}.jsonl") else {}
    sl = sharp(f"{R}/single_sharplight_{n}.txt") if os.path.exists(f"{R}/single_sharplight_{n}.txt") else {}
    vl = vllm(n)
    rows = [("pp512", ("pp512", 0), lambda: sl.get(512), "pp"), ("pp2048", ("pp2048", 0), lambda: sl.get(2048), "pp"),
            ("tg128 @d0", ("tg128", 0), lambda: sl.get(16), "dmed"), ("tg128 @d4096", ("tg128", 4096), lambda: sl.get(4096), "dmed"),
            ("pp512 @d4096 (llama only)", ("pp512", 4096), lambda: None, "pp")]
    for label, k, getter, field in rows:
        s = getter(); sv = [x[field] for x in s] if s else None
        cell = lambda v: "n/a" if not v else f"{f(statistics.median(v))} ({f(min(v))}..{f(max(v))})"
        c = cu.get(k); v = vk.get(k)
        sm = statistics.median(sv) if sv else None
        r = lambda a, b: "%.2fx" % (a / b) if a and b else "-"
        print(f"| {n} | {label} | {cell(sv)} | {cell(c and list(c))} | {cell(v and list(v))} | {f(vl[label]) if vl.get(label) else 'n/a'} | {r(sm, c and c[1])} | {r(sm, v and v[1])} | {r(sm, vl.get(label))} |")
