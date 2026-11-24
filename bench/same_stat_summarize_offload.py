#!/usr/bin/env python3
"""Summarizes same_stat_offload.sh output per (model, provider): Sharplight per-run mean ± sd (prefill 512, decode 64 @ 512), the decode best /
median step for reference, llama-bench pp512 and tg64 @ d512 per round and setting, and each process's major page faults and reads
over its timed phase (sample_faults.sh).
usage: same_stat_summarize_offload.py <out dir>"""
import glob, os, re, sys
root = sys.argv[1]
def faults(path, dur, log=None, marker=None):
    """Major faults and reads per second over the timed phase. With a log line count in the samples: from the first sample after
    the log's first `marker` line (after Sharplight's warm-up; llama-bench's table header) to the last sample before the RSS falls
    (teardown). Without: the last `dur` + 3 seconds before that end (an estimate from the measured rates)."""
    rows = [l.split() for l in open(path)][1:]
    rows = [r for r in rows if "NA" not in r[:5]]
    if len(rows) < 2:
        return "n/a"
    peak = max(int(r[3]) for r in rows)
    end = max(float(r[0]) for r in rows if int(r[3]) >= 0.9 * peak)
    start = end - dur - 3
    how = "estimated"
    if log and len(rows[0]) > 7 and rows[0][7] != "NA":
        lines = open(log).read().splitlines()
        idx = next((i + 1 for i, l in enumerate(lines) if marker in l), None)
        if idx is not None:
            start = min((float(r[0]) for r in rows if r[7] != "NA" and int(r[7]) >= idx), default=start)
            how = "marked"
    last = [r for r in rows if start <= float(r[0]) <= end]
    if len(last) < 2:
        return "n/a"
    d = int(last[-1][1]) - int(last[0][1]); dt = float(last[-1][0]) - float(last[0][0]) or 1
    rd = int(last[-1][2]) - int(last[0][2])
    return f"timed phase ({how}) {dt:.0f} s: {d} majflt, {rd} MB read; whole run {rows[-1][1]} majflt, {rows[-1][2]} MB read, min cgroup cache {min(int(r[4]) for r in rows)} MB"
for f in sorted(glob.glob(os.path.join(root, "sl_*.txt"))):
    base = os.path.basename(f)[3:-4]
    print(f"## {base}")
    for log in sorted(glob.glob(os.path.join(root, f"sl_{base}_*.log"))):
        tag = log[len(os.path.join(root, f"sl_{base}_")):-4]
        s = open(log).read()
        m = re.search(r"decode: best ([\d.]+), median ([\d.]+).*per run: prefill ([\d.]+) ± ([\d.]+) tok/s of (\d+), decode ([\d.]+) ± ([\d.]+) tok/s of (\d+) \(([^)]*)\)", s)
        fl = os.path.join(root, f"faults_sl_{base}_{tag}.txt")
        if m:
            print(f"SL {tag}: pp512 {m[3]} ± {m[4]} (n={m[5]}), tg64@512 {m[6]} ± {m[7]} (n={m[8]}: {m[9]}); step best/median {m[1]} / {m[2]}; {faults(fl, 11 * 512 / float(m[3]) + 5 * 64 / float(m[6]), log, 'per step') if os.path.exists(fl) else ''}")
        else:
            print(f"SL {tag}: FAILED {re.findall(r'.*(?:rror|xception).*', s)[:2]}")
    for log in sorted(glob.glob(os.path.join(root, f"llama_{base}_*.log"))):
        tag = log[len(os.path.join(root, f"llama_{base}_")):-4]
        out = []
        hdr = None
        for line in open(log):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if c and c[0] == "model":
                hdr = c
                continue
            if hdr and len(c) == len(hdr) and re.match(r"(pp|tg)\d+", c[-2]):
                row = dict(zip(hdr, c))
                extra = "".join(f", {k} {row[k]}" for k in ("lm",) if k in row)
                out.append(f"{c[-2]} {c[-1]} (ncmoe {row.get('n_cpu_moe', '?')}{extra})")
        fl = os.path.join(root, f"faults_llama_{base}_{tag}.txt")
        rates = [float(x.split(") (")[0].rsplit(" ± ", 1)[0].split()[-1]) for x in out]
        dur = sum((5 if x.startswith("tg") else 6) * (64 if x.startswith("tg") else 512) / r for x, r in zip(out, rates)) + 5 * len(out)
        print(f"llama {tag}: " + "; ".join(out) + f"; {faults(fl, dur) if os.path.exists(fl) and out else ''}")
