"""Compares a bench/ParitySweep result with a reference (reference.py or llama_reference.py) and prints one verdict per check.

Usage: python -I compare.py <ours.json> <reference.json> [--tie 0.1] [--no-greedy]
Greedy policy: identical ids pass; at the first differing step the reference's logit gap between its token and ours must be at
most --tie (a near-tie; the rest of the sequence is then not compared). A text-only reference (llama.cpp) compares decoded text.
Exit code 1 when any check fails.
"""
import json
import sys


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def show(a, b, width=160):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    s = max(0, i - 40)
    return f"@{i}: ours={a[s:s + width]!r}\n      ref ={b[s:s + width]!r}"


def main():
    ours = json.load(open(sys.argv[1], encoding="utf-8"))
    ref = json.load(open(sys.argv[2], encoding="utf-8"))
    opts = sys.argv[3:]
    tie = float(opts[opts.index("--tie") + 1]) if "--tie" in opts else 0.1
    failed = False

    def verdict(name, ok, detail=""):
        nonlocal failed
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}{(' ' + detail) if detail else ''}")

    for n, (a, b) in enumerate(zip(ours.get("texts", []), ref.get("texts", []))):
        if "error" in a or "error" in b:
            verdict(f"text{n}", False, f"error ours={a.get('error')} ref={b.get('error')}")
            continue
        k = first_diff(a["ids"], b["ids"])
        detail = "" if k is None else f"ids differ at {k}: ours={a['ids'][max(0, k - 2):k + 4]} ref={b['ids'][max(0, k - 2):k + 4]}"
        verdict(f"text{n}.encode", k is None, detail)
        verdict(f"text{n}.decode", a["decoded"] == b["decoded"], "" if a["decoded"] == b["decoded"] else show(a["decoded"], b["decoded"]))

    if "has_template" in ref and ours.get("has_template") != ref.get("has_template"):
        verdict("template.present", False, f"ours={ours.get('has_template')} ref={ref.get('has_template')}")
    for name, b in ref.get("chats", {}).items():
        a = ours.get("chats", {}).get(name, {"error": "missing"})
        if "error" in b:
            verdict(f"chat.{name}", "error" in a, f"(reference raises: {b['error'][:120]}; ours: {a.get('error', 'renders')[:120]})")
            continue
        if "error" in a:
            verdict(f"chat.{name}", False, a["error"][:300])
            continue
        verdict(f"chat.{name}.text", a["text"] == b["text"], "" if a["text"] == b["text"] else show(a["text"], b["text"]))
        ida, idb = a["ids"], b["ids"]
        bos_only = len(ida) == len(idb) + 1 and ida[1:] == idb
        verdict(f"chat.{name}.ids", ida == idb or bos_only, ("(+BOS)" if bos_only else "") if ida == idb or bos_only else f"differ at {first_diff(ida, idb)}")

    if "--no-greedy" not in opts:
        refs = {g["prompt"]: g for g in ref.get("greedy", [])}
        for a in ours.get("greedy", []):
            b = refs.get(a["prompt"])
            if b is None:
                continue
            name = f"greedy[{a['prompt'][:30]}]"
            if "error" in a or "error" in b:
                verdict(name, False, f"error ours={a.get('error')} ref={b.get('error')}")
                continue
            if "ids" not in b:
                ta, tb = a["text"].lstrip(" "), b["text"].lstrip(" ")
                ok = ta == tb or ta.startswith(tb) or tb.startswith(ta)
                verdict(name + ".text", ok, "" if ok else show(a["text"], b["text"]))
                continue
            pa, pb = a["prompt_ids"], b["prompt_ids"]
            if pa != pb:
                verdict(name + ".prompt", False, f"prompt ids differ at {first_diff(pa, pb)}")
                continue
            k = first_diff(a["ids"], b["ids"])
            if k is None:
                verdict(name, True, f"identical {len(a['ids'])} tokens")
                continue
            top = dict((t, v) for t, v in b["top5"][k]) if k < len(b.get("top5", [])) else {}
            ours_tok = a["ids"][k] if k < len(a["ids"]) else None
            gap = top[b["ids"][k]] - top[ours_tok] if ours_tok in top else None
            near = gap is not None and gap <= tie
            verdict(name, near, f"{'near-tie' if near else 'diverges'} at {k}/{len(b['ids'])} (gap {gap if gap is None else round(gap, 4)})"
                    + ("" if near else "\n      " + show(a["text"], b["text"])))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
