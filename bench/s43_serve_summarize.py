#!/usr/bin/env python3
"""Serving tables of docs/benchmarks.md section 43 from the files bench/s43_serve.sh writes.
Per model, engine and concurrency: aggregate output tok/s, TTFT p50 and per-stream decode tok/s (1000 / mean inter-token
latency), each the median over the rounds (min..max for the throughput), and Sharplight / other engine.
usage: s43_serve_summarize.py <results dir> [<results dir after>]   (with two dirs: Sharplight before / after plus the others)"""
import glob, json, os, re, statistics, sys


def load(d):
    runs = {}
    for f in glob.glob(os.path.join(d, "serve_*_c*_r*.json")):
        m = re.match(r"serve_(sharplight|llama-cuda13|llama-vulkan|vllm)_(.+)_c(\d+)_r\d+\.json$", os.path.basename(f))
        if not m:
            continue
        j = json.load(open(f))
        if j.get("errors"):
            continue
        runs.setdefault((m.group(2), m.group(1), int(m.group(3))), []).append(j)
    return runs


def stat(js):
    tok = [j["agg_tok_s"] for j in js]
    return (statistics.median(tok), min(tok), max(tok), statistics.median(j["ttft_p50_ms"] for j in js),
            statistics.median(1000.0 / j["itl_mean_ms"] for j in js), len(js))


before = load(sys.argv[1])
after = load(sys.argv[2]) if len(sys.argv) > 2 else {}
models = sorted({k[0] for k in before} | {k[0] for k in after})
for model in models:
    others = sorted({k[1] for k in before if k[0] == model and k[1] != "sharplight"})
    concs = sorted({k[2] for k in list(before) + list(after) if k[0] == model})
    for other in others:
        head = "| model | streams | SL tok/s |" + (" SL after tok/s |" if after else "") + f" {other} tok/s | SL / {other} |" \
            + " TTFT p50 SL / other (ms) | decode tok/s per stream SL / other |"
        print(head)
        print("|" + "---|" * (head.count("|") - 1))
        for c in concs:
            sl = before.get((model, "sharplight", c))
            sa = after.get((model, "sharplight", c))
            ot = before.get((model, other, c))
            if not ot or not (sl or sa):
                continue
            o = stat(ot)
            best = stat(sa) if sa else stat(sl)
            cells = [model, str(c)]
            cells.append(f"{stat(sl)[0]:.0f} ({stat(sl)[1]:.0f}..{stat(sl)[2]:.0f})" if sl else "-")
            if after:
                cells.append(f"{stat(sa)[0]:.0f} ({stat(sa)[1]:.0f}..{stat(sa)[2]:.0f})" if sa else "-")
            cells += [f"{o[0]:.0f} ({o[1]:.0f}..{o[2]:.0f})", f"{best[0] / o[0]:.2f}x", f"{best[3]:.0f} / {o[3]:.0f}", f"{best[4]:.0f} / {o[4]:.0f}"]
            print("| " + " | ".join(cells) + " |")
        print()
