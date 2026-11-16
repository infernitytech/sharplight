"""Prints the per-step logit differences of a ParitySweep --teacher run against the transformers reference's top-5 logits.

Usage: python -I teacher.py <ours.json> <reference.json> [--tie 0.1]
PASS when every step where our argmax differs from the reference's token is a near-tie (reference gap at most --tie).
"""
import json
import sys


def main():
    ours = json.load(open(sys.argv[1], encoding="utf-8"))
    tie = float(sys.argv[sys.argv.index("--tie") + 1]) if "--tie" in sys.argv else 0.1
    failed = False
    ref = {g["prompt"]: g for g in json.load(open(sys.argv[2], encoding="utf-8"))["greedy"]}
    for t in ours.get("teacher", []):
        r = ref[t["prompt"]]
        diffs = []
        flips = []
        for k, step in enumerate(t["steps"]):
            ro = dict((i, v) for i, v in r["top5"][k])
            d = max(abs(v - ro[i]) for i, v in step["at_ref"])
            diffs.append(d)
            if step["argmax"] != r["ids"][k]:
                flips.append((k, round(r["top5"][k][0][1] - (ro.get(step["argmax"]) or float("nan")), 3)))
        ok = all(gap == gap and gap <= tie for _, gap in flips)
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'} teacher {t['prompt'][:30]!r}: max|dlogit| over ref top-5: max {max(diffs):.4f} mean {sum(diffs) / len(diffs):.4f}; argmax flips (step, ref gap) {flips[:6]}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
