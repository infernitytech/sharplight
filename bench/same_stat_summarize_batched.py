#!/usr/bin/env python3
"""Summarizes same_stat_batched.sh output: per (model, provider, batch) Sharplight mean / median step ms (greedy and download modes) per round,
llama.cpp mean step ms (t_tg / 64 of the second pass) per round, tok/s from the means and the ratio.
usage: same_stat_summarize_batched.py <out dir>"""
import glob, os, re, sys, statistics as st
root = sys.argv[1]
cell = re.compile(r"([\d.]+) / ([\d.]+), ([\d.]+); mean ([\d.]+) ± ([\d.]+)")
for f in sorted(glob.glob(os.path.join(root, "sl_*.txt"))):
    base = os.path.basename(f)[3:-4]
    sl = {}
    rnd = None
    for line in open(f):
        if line.startswith("=="):
            rnd = line.split()[-1]; continue
        m = re.match(r"\| (\d+) \|", line)
        if m:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            vals = [cell.match(c) for c in cells[3:]]
            sl.setdefault(int(m[1]), []).append((rnd, [tuple(map(float, v.groups())) for v in vals if v]))
    ll = {}
    lf = os.path.join(root, f"llama_{base}.txt")
    rnd = None; seen = {}
    if os.path.exists(lf):
        for line in open(lf):
            if line.startswith("=="):
                rnd = line.split()[-1 if 'ncmoe' not in line else -3]; seen = {}; continue
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) == 10 and c[0].isdigit():
                b = int(c[2]); seen[b] = seen.get(b, 0) + 1
                if seen[b] == 2:
                    ll.setdefault(b, []).append((rnd, float(c[6]) * 1000 / int(c[1]), float(c[7]), float(c[1])))
    print(f"## {base}")
    print("| seqs | SL greedy mean ms (rounds) | SL greedy median | SL download mean | llama mean ms (rounds) | SL tok/s | llama tok/s | ratio greedy | ratio download |")
    for b in sorted(set(sl) | set(ll)):
        g = [v[0] for _, v in sl.get(b, []) if v]
        d = [v[1] for _, v in sl.get(b, []) if len(v) > 1]
        l = [x[1] for x in ll.get(b, [])]
        gm = st.mean(x[3] for x in g) if g else None
        dm = st.mean(x[3] for x in d) if d else None
        lm = st.mean(l) if l else None
        fmt = lambda xs: ", ".join(f"{x:.2f}" for x in xs)
        r = lambda a: f"{lm / a:.2f}" if a and lm else ""
        print(f"| {b} | {gm and f'{gm:.2f}'} ({fmt([x[3] for x in g])}) | {fmt([x[1] for x in g])} | {dm and f'{dm:.2f}'} ({fmt([x[3] for x in d])}) | "
              f"{lm and f'{lm:.2f}'} ({fmt(l)}) | {gm and f'{b * 1000 / gm:.0f}'} | {lm and f'{b * 1000 / lm:.0f}'} | {r(gm)} | {r(dm)} |")
