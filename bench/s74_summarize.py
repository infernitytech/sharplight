#!/usr/bin/env python3
"""Tables of bench/s74_feed.sh results (docs/benchmarks.md section 74): Sharplight against llama.cpp per model and provider.
usage: s74_summarize.py <result dir> [<model name> ...]
Prefill: best tok/s of 5 (Sharplight) against llama-bench's mean of 3. Decode, one stream: the `--bench 512` decode (best tok/s)
against tg128. Decode at depth: 1000 / BatchBench's minimum GPU ms per step against llama-bench tg64 at -d D. Batched: sequences x
1000 / minimum GPU ms per step at depth 256 against llama-batched-bench's S_TG (the second, warm pass when there are two)."""
import os
import re
import sys


def sharplight(path):
    out = {}
    if not os.path.exists(path):
        return out
    batched = False
    for line in open(path):
        m = re.match(r"\[N=(\d+)\] prefill \d+ tok: best \d+ ms = ([\d.]+) tok/s \| decode: best ([\d.]+)", line)
        if m:
            out[f"pp{m[1]}"] = float(m[2])
            if m[1] == "512":
                out["tg"] = float(m[3])
            continue
        m = re.match(r"\[decode depth (\d+)\] \| 1 \| [\d.]+ \| [\d/ ]+ \| [\d.]+ / [\d.]+, ([\d.]+)", line)
        if m:
            out[f"d{m[1]}"] = 1000 / float(m[2])
            continue
        if line.startswith("[batched"):
            batched = True
            continue
        m = re.match(r"\| (\d+) \| [\d.]+ \| [\d/ ]+ \| [\d.]+ / [\d.]+, ([\d.]+)", line)
        if batched and m:
            out[f"b{m[1]}"] = int(m[1]) * 1000 / float(m[2])
    return out


def llama(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 8 and cells[6].startswith("pp"):
            out[cells[6]] = float(cells[7].split()[0])
        elif len(cells) >= 8 and cells[6].startswith("tg"):
            m = re.match(r"tg\d+ @ d(\d+)", cells[6])
            out[f"d{m[1]}" if m else "tg"] = float(cells[7].split()[0])
        elif len(cells) == 10 and cells[0].isdigit():
            out[f"b{cells[2]}"] = float(cells[7])   # later (warm) passes overwrite the first
    return out


def main():
    root = sys.argv[1]
    names = sys.argv[2:] or sorted({f.split("_", 1)[1].rsplit("_", 1)[0] for f in os.listdir(root) if f.startswith("sl_")})
    keys = ["pp512", "pp2048", "pp8192", "pp32768", "tg", "d32768", "d131072", "b1", "b16", "b64"]
    for name in names:
        for prov in ("cuda", "vulkan"):
            s = sharplight(os.path.join(root, f"sl_{name}_{prov}.txt"))
            l = llama(os.path.join(root, f"llama_{name}_{prov}.txt"))
            if not s and not l:
                continue
            print(f"\n{name} {prov}")
            print("| point | Sharplight | llama.cpp | ratio |")
            print("|---|---:|---:|---:|")
            for k in keys:
                if k in s or k in l:
                    a, b = s.get(k), l.get(k)
                    ratio = f"{a / b:.2f}" if a and b else ""
                    print(f"| {k} | {a:.1f} | " if a else f"| {k} | | ", end="")
                    print(f"{b:.1f} | {ratio} |" if b else f" | {ratio} |")


if __name__ == "__main__":
    main()
