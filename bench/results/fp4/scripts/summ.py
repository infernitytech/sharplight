#!/usr/bin/env python3
"""Serving table of the FP4 section: per engine, model and concurrency the rounds' tok/s, TTFT p50 and ITL mean / p95 / p99.
usage: summ.py <serve results dir>"""
import glob, json, os, re, sys

D = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(D, "serve_*_c*_r*.json"))):
    m = re.match(r"serve_(\w+?)_(\w+)_c(\d+)_r(\d+)\.json", os.path.basename(f))
    if not m:
        continue
    d = json.load(open(f))
    rows.setdefault((m.group(2), int(m.group(3)), m.group(1)), []).append(d)
for (name, c, eng), ds in sorted(rows.items()):
    tok = " / ".join(f"{d['agg_tok_s']:.0f}" for d in ds)
    ttft = " / ".join(f"{d['ttft_p50_ms']:.0f}" for d in ds)
    itl = " / ".join(f"{d['itl_mean_ms']:.2f}, {d['itl_p95_ms']:.1f}, {d['itl_p99_ms']:.1f}" for d in ds)
    err = sum(d["errors"] for d in ds)
    print(f"| {name} | {c} | {eng} | {tok} | {ttft} | {itl} | {err} |")
for f in sorted(glob.glob(os.path.join(D, "*.textcheck.json"))):
    t = json.load(open(f))
    print(os.path.basename(f), "garbage:", t.get("garbage", t.get("failed", "?")), "summary:", {k: v for k, v in t.items() if not isinstance(v, (list, dict))})
