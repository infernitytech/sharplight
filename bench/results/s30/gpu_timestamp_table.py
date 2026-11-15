"""GPU-timestamp minimum per (model, batch, engine) from a parity_bench jsonl file, against section 27's strict and
split-f16 runs; prints a Markdown table."""
import json
import sys


def load(path, by_engine):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("stage") != "ok" or "gpuMinMs" not in r:
            continue
        if by_engine:
            eng = r["engine"]
        else:
            eng = "rp" if r.get("reducedPrecision") else "split" if r.get("splitHalf") else "strict"
        key = (r["model"], r["batch"], eng)
        cur = out.get(key)
        if cur is None or r["gpuMinMs"] < cur[0]:
            out[key] = (r["gpuMinMs"], r.get("worstRms"))
    return out


def main(new_path, old_path):
    new = load(new_path, True)
    old = load(old_path, False)
    keys = sorted({(m, b) for m, b, _ in new}, key=lambda k: list(dict.fromkeys(m for m, _, _ in new)).index(k[0]) * 100 + k[1])
    print("| model | batch | §27 strict | §27 split | sl-vulkan (strict default) | sl-vulkan-simt | sl-vulkan-rp | worst RMS default / simt |")
    print("|---|---|---|---|---|---|---|---|")
    f = lambda v: f"{v[0]:.2f}" if v else "-"
    r = lambda v: f"{v[1]:.1e}" if v and v[1] is not None else "-"
    for m, b in keys:
        d, s = new.get((m, b, "sl-vulkan")), new.get((m, b, "sl-vulkan-simt"))
        print(f"| {m} | {b} | {f(old.get((m, b, 'strict')))} | {f(old.get((m, b, 'split')))} | {f(d)} | {f(s)} | "
              f"{f(new.get((m, b, 'sl-vulkan-rp')))} | {r(d)} / {r(s)} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
