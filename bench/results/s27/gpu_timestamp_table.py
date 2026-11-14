"""GPU-timestamp minimum per (model, batch, engine) from parity_bench jsonl files; prints a Markdown table (new vs old)."""
import json
import sys


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("stage") != "ok" or "gpuMinMs" not in r:
            continue
        eng = "rp" if r.get("reducedPrecision") else "split" if r.get("splitHalf") else "strict"
        key = (r["model"], r["batch"], eng)
        cur = out.get(key)
        rms = r.get("worstRms")
        if cur is None or r["gpuMinMs"] < cur[0]:
            out[key] = (r["gpuMinMs"], rms)
    return out


new = load(sys.argv[1])
old = load(sys.argv[2]) if len(sys.argv) > 2 else {}
rows = sorted({(m, b) for (m, b, _) in new}, key=lambda k: (list(dict.fromkeys(m for (m, _, _) in new)).index(k[0]), k[1]))
print("| model | batch | strict (§26 → §27) | split f16 (§26 → §27) | reduced | worst rms (strict / split / reduced) |")
print("|---|---|---|---|---|---|")


def cell(eng, m, b):
    n = new.get((m, b, eng))
    o = old.get((m, b, eng))
    if n is None:
        return "-"
    return f"{o[0]:.2f} → {n[0]:.2f}" if o else f"{n[0]:.2f}"


def rms(eng, m, b):
    n = new.get((m, b, eng))
    return f"{n[1]:.1e}" if n and n[1] is not None else "-"


for m, b in rows:
    rp = new.get((m, b, "rp"))
    print(f"| {m} | {b} | {cell('strict', m, b)} | {cell('split', m, b)} | {rp[0]:.2f} | {rms('strict', m, b)} / {rms('split', m, b)} / {rms('rp', m, b)} |" if rp else f"| {m} | {b} | {cell('strict', m, b)} | {cell('split', m, b)} | - | |")
