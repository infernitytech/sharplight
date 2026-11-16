"""Greedy GGUF verdict across devices: a prompt passes when one of our outputs (Vulkan, CPU) equals one of llama.cpp's (Vulkan,
CPU) over the shorter length, or when ours agree with a llama.cpp output at least as long (in characters) as llama.cpp's backends
agree with each other. llama.cpp's own backends split on near-ties the same way ours do, so this tells a near-tie from a real
divergence without logits.

Usage: python -I crossmatch.py <ours.json>... -- <llama.json>...
"""
import json
import sys


def common(x, y):
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[n]:
        n += 1
    return n


def main():
    split = sys.argv.index("--")
    ours = [json.load(open(p, encoding="utf-8")) for p in sys.argv[1:split]]
    refs = [json.load(open(p, encoding="utf-8")) for p in sys.argv[split + 1:]]
    failed = False
    for g in ours[0]["greedy"]:
        prompt = g["prompt"]
        a = [x["text"] for o in ours for x in o["greedy"] if x["prompt"] == prompt and "text" in x]
        b = [x["text"] for r in refs for x in r["greedy"] if x["prompt"] == prompt and "text" in x]
        if not b:
            continue
        ok = any(x.startswith(y) or y.startswith(x) for x in a for y in b)
        ours_len = max((common(x, y) for x in a for y in b), default=0)
        ref_len = min((common(x, y) for x in b for y in b), default=0)
        near = not ok and len(b) > 1 and ours_len >= ref_len
        failed |= not (ok or near)
        print(f"{'PASS' if ok or near else 'FAIL'} cross[{prompt[:30]}]" + (f" near-tie: llama.cpp backends agree for {ref_len} chars, ours for {ours_len}" if near or not ok else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
