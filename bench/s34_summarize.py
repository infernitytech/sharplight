#!/usr/bin/env python3
"""Single-stream table of docs/benchmarks.md section 34 from the files of s34_single_all.sh / s16_single.sh.
usage: s34_summarize.py [results dir, default bench/results/s34/single]
Cells: median (min..max) over Sharplight process runs / llama-bench repetitions; vLLM: median of its repetitions.
Prints the table and the gap list (Sharplight / best other engine, ascending)."""
import glob, json, os, re, statistics, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "s34", "single")
METRICS = ["pp512", "pp2048", "tg128 @d0", "tg128 @d4096"]
VLLM_NAMES = {"qwen3-0.6b-q8": "qwen3-0.6b-bf16", "llama3.2-1b-q8": "llama3.2-1b-bf16", "gemma3-1b-q8": "gemma3-1b-bf16",
              "olmoe-q4km": "olmoe-bf16"}


def cell(xs):
    if not xs:
        return None
    return (statistics.median(xs), min(xs), max(xs))


def fmt(c):
    if c is None:
        return "-"
    m, lo, hi = c
    f = "{:.0f}" if m >= 100 else "{:.1f}"
    return f"{f.format(m)} ({f.format(lo)}..{f.format(hi)})"


def sharplight(path):
    pp, tg = {}, {}
    for line in open(path):
        m = re.match(r"\[N=(\d+) run=\d+\] prefill \d+ tok: best [\d.]+ ms = ([\d.]+) tok/s \| decode: best [\d.]+, median ([\d.]+)", line)
        if m:
            n = int(m.group(1))
            pp.setdefault(n, []).append(float(m.group(2)))
            tg.setdefault(n, []).append(float(m.group(3)))
    return {"pp512": cell(pp.get(512)), "pp2048": cell(pp.get(2048)), "tg128 @d0": cell(tg.get(16)), "tg128 @d4096": cell(tg.get(4096))}


def llama(path):
    r = {}
    for line in open(path):
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j["n_prompt"] and j["n_depth"] == 0:
            r[f"pp{j['n_prompt']}"] = cell(j["samples_ts"])
        elif j["n_gen"]:
            r[f"tg128 @d{j['n_depth']}"] = cell(j["samples_ts"])
    return r


def vllm(path):
    j = json.load(open(path))
    r = {}
    for k, v in j["results"].items():
        key = {"pp512": "pp512", "pp2048": "pp2048", "tg128@d16": "tg128 @d0"}.get(k)
        if key is None and k.startswith("tg128@d"):
            key = "tg128 @d4096"
        if key:
            r[key] = (v["tok_s_median"], v["tok_s_min"], v["tok_s_max"])
    return r


rows = {}
for f in sorted(glob.glob(os.path.join(D, "single_sharplight_*.txt"))):
    name = os.path.basename(f)[len("single_sharplight_"):-4]
    rows[name] = {"sl": sharplight(f)}
    for b in ("cuda13", "vulkan"):
        p = os.path.join(D, f"single_llama-{b}_{name}.jsonl")
        rows[name][b] = llama(p) if os.path.exists(p) else {}
    p = os.path.join(D, f"single_vllm_{VLLM_NAMES.get(name, name)}.json")
    rows[name]["vllm"] = vllm(p) if os.path.exists(p) else {}

print("| model | metric | Sharplight | llama.cpp CUDA | llama.cpp Vulkan | vLLM (bf16) | SL/CUDA | SL/Vulkan | SL/vLLM | SL/best |")
print("|---|---|---|---|---|---|---:|---:|---:|---:|")
gaps = []
for name, r in rows.items():
    for mtr in METRICS:
        s = r["sl"].get(mtr)
        others = {e: r[e].get(mtr) for e in ("cuda13", "vulkan", "vllm")}
        ratios = {e: (s[0] / o[0] if s and o else None) for e, o in others.items()}
        best = [o[0] for o in others.values() if o]
        rb = s[0] / max(best) if s and best else None
        rs = lambda x: f"{x:.2f}x" if x else "-"
        print(f"| {name} | {mtr} | {fmt(s)} | {fmt(others['cuda13'])} | {fmt(others['vulkan'])} | {fmt(others['vllm'])} | "
              f"{rs(ratios['cuda13'])} | {rs(ratios['vulkan'])} | {rs(ratios['vllm'])} | {rs(rb)} |")
        if rb:
            be = max((e for e in others if others[e]), key=lambda e: others[e][0])
            gaps.append((rb, name, mtr, be))
print("\nGaps (Sharplight / best other engine, ascending):\n")
for rb, name, mtr, be in sorted(gaps):
    print(f"* {rb:.2f}x {name} {mtr} (best: {be})")
