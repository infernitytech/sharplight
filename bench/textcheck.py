#!/usr/bin/env python3
"""Batched greedy text check for an OpenAI-compatible /v1/completions server (docs/benchmarks.md section 102).

Speed benchmarks send random words with ignore_eos and never read the text, so a numeric bug that only batched steps hit (a kernel
for 9-16 rows, a race between launches) keeps its speed and goes unseen (section 101.4). This check:

1. sends each of a few natural greedy prompts alone (the single-request reference; one sequence per step),
2. sends waves of concurrent requests cycling the same prompts, each wave staggered by a few milliseconds so that its steps mix
   prompt rows and decode rows of 1 to N sequences (and the row counts fall again as requests finish),
3. compares every batched text with its prompt's reference.

Batched steps are not bit-identical to single-sequence steps (other kernels, other summation orders, per-row activation
quantization), so greedy texts part after near ties, and on open-ended completions early: on a healthy Qwen3-8B-FP8 server a third
to a half of the batched texts part within the first eight words, fluently (section 102.1). Parting alone is therefore reported, not
judged, unless nearly every text parts at once. A request is flagged as garbage when
* it repeats a token the reference does not repeat (the same word or symbol --repeat times in a row, e.g. "!!!!!!"), or
* punctuation-only tokens make up more than --symbols of it and more than three times the reference's share, or
* it carries a script the reference does not (CJK, Cyrillic, ... in an English answer: what wrong logits sample), or
* the request failed or returned no text.
A wave fails when any request is garbage, or when more than --max-early (a fraction) of its texts part from their references
within the first --min-words words.

usage: textcheck.py --url http://127.0.0.1:18080 [--model m] [--waves 16,64] [--prompts 8] [--max-tokens 48] [--json out.json]
Exit code 0 when clean, 1 when garbage was found, 2 when the server could not be reached.
"""
import argparse, asyncio, json, re, sys, unicodedata
import httpx

PROMPTS = (
    "Write a detailed essay about the role of water and power in modern history. Explain the history, give concrete examples and discuss open questions.",
    "Explain step by step how a teacher could help a student solve a problem with their homework. Be specific and practical.",
    "Write a long story about a girl who finds a book in the forest. Describe the characters and the setting in detail.",
    "Compare cities and villages from the point of view of a doctor. List advantages, disadvantages and a final recommendation.",
    "The history of the printing press begins in",
    "Here is a short summary of how photosynthesis works:",
    "Q: What are three good reasons to learn a second language?\nA:",
    "Write a letter to a friend describing a trip to the mountains in winter.",
    "List the planets of the solar system in order from the sun and give one fact about each.",
    "Explain to a ten-year-old why the sky is blue.",
    "Describe the steps of making bread at home, from the flour to the finished loaf.",
    "A museum guide explains the most famous painting of the collection to a group of visitors:",
)

WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def words(text):
    return WORD.findall(text)


def common_words(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def longest_run(tokens):
    """Length and token of the longest run of one token repeated back to back."""
    best, tok, run = 0, None, 0
    for i, t in enumerate(tokens):
        run = run + 1 if i and t == tokens[i - 1] else 1
        if run > best:
            best, tok = run, t
    return best, tok


def symbol_share(tokens):
    """Fraction of tokens that are punctuation or symbols only."""
    return sum(1 for t in tokens if not any(c.isalnum() for c in t)) / max(len(tokens), 1)


def scripts(text):
    """The Unicode scripts of the letters of text, by the first word of their character names (LATIN, CJK, CYRILLIC, ...)."""
    out = set()
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            out.add(name.split(" ")[0] if name else "UNKNOWN")
    return out


async def complete(client, args, prompt, delay=0.0):
    await asyncio.sleep(delay)
    body = {"model": args.model, "prompt": prompt, "max_tokens": args.max_tokens, "temperature": 0}
    try:
        r = await client.post(args.url + "/v1/completions", json=body)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:160]!r}"
        return r.json()["choices"][0]["text"], None
    except Exception as e:  # report, do not crash the benchmark
        return None, f"{type(e).__name__}: {e}"


def judge(ref, text, args):
    """Flags of one batched text against its reference (empty: clean) and the number of leading words they share."""
    flags = []
    rw, tw = words(ref), words(text)
    same = common_words(rw, tw)
    if not tw and rw:
        flags.append("empty")
    elif same < min(args.min_words, len(rw), len(tw)):
        flags.append(f"early:{same}")
    run, tok = longest_run(tw)
    ref_run, _ = longest_run(rw)
    if run >= args.repeat and run > ref_run:
        flags.append(f"repeat:{tok!r}x{run}")
    share, ref_share = symbol_share(tw), symbol_share(rw)
    if share > args.symbols and share > 3 * ref_share:
        flags.append(f"symbols:{share:.2f}")
    extra = scripts(text) - scripts(ref)
    if extra:
        flags.append("script:" + ",".join(sorted(extra)))
    return flags, same


def summarize(args, n, rows):
    """Counts of a wave's judged rows, its verdict and the printed line."""
    early = sum(1 for r in rows if r["flags"] and all(f.startswith("early:") for f in r["flags"]))
    other = sum(1 for r in rows if any(not f.startswith("early:") for f in r["flags"]))
    ok = other == 0 and early <= args.max_early * n
    bad = [r for r in rows if r["flags"]]
    line = (f"textcheck wave {n}: {sum(r.get('identical', False) for r in rows)}/{n} identical to single-request, {early} part within "
            f"{args.min_words} words, {other} garbage -> {'ok' if ok else 'FAIL'}"
            + "".join(f"\n  #{r['i']} {r['flags']}: {(r['text'] or '')[:100]!r}" for r in bad if any(not f.startswith('early:') for f in r['flags']))[:2000])
    return early, other, ok, line


def rejudge(args):
    report = json.load(open(args.rejudge))
    ok = True
    for w in report["waves"]:
        for r in w["rows"]:
            ref = report["references"][r["prompt"]]
            if r["text"] is not None:
                r["flags"], r["same_words"] = judge(ref, r["text"], args)
                r["identical"] = r["text"] == ref
        early, other, wave_ok, line = summarize(args, w["requests"], w["rows"])
        ok &= wave_ok
        print(line)
    print(f"textcheck: {'clean' if ok else 'GARBAGE'}")
    return 0 if ok else 1


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--model", default="m")
    ap.add_argument("--waves", default="16,64", help="concurrent requests per wave (comma separated)")
    ap.add_argument("--prompts", type=int, default=8, help="distinct prompts (cycled by the waves)")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--stagger-ms", type=float, default=4.0, help="request i of a wave starts after i * this")
    ap.add_argument("--min-words", type=int, default=3, help="a batched text parting from its reference before this many words is early")
    ap.add_argument("--max-early", type=float, default=0.5, help="fraction of a wave's texts that may part early (near ties)")
    ap.add_argument("--symbols", type=float, default=0.25, help="largest share of punctuation-only tokens of a fluent text")
    ap.add_argument("--repeat", type=int, default=6, help="a token repeated this many times in a row (more than in the reference) is garbage")
    ap.add_argument("--json")
    ap.add_argument("--rejudge", help="judge the texts of a stored --json report again (other thresholds) instead of asking a server")
    args = ap.parse_args()
    if args.rejudge:
        return rejudge(args)
    prompts = PROMPTS[:max(1, min(args.prompts, len(PROMPTS)))]
    waves = [int(w) for w in args.waves.replace(" ", ",").split(",") if w]
    report = {"url": args.url, "max_tokens": args.max_tokens, "prompts": len(prompts), "waves": [], "ok": True}
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0), limits=httpx.Limits(max_connections=max(waves) + 4)) as client:
        refs = []
        for p in prompts:
            text, err = await complete(client, args, p)
            if err:
                print(f"textcheck: reference request failed: {err}", file=sys.stderr)
                report["ok"] = False
                report["error"] = err
                if args.json:
                    json.dump(report, open(args.json, "w"), indent=1)
                return 2
            refs.append(text)
        report["references"] = refs
        for n in waves:
            res = await asyncio.gather(*(complete(client, args, prompts[i % len(prompts)], i * args.stagger_ms / 1000) for i in range(n)))
            rows = []
            for i, (text, err) in enumerate(res):
                ref = refs[i % len(prompts)]
                if err or text is None:
                    flags, same = [f"error:{err}"], 0
                else:
                    flags, same = judge(ref, text, args)
                rows.append({"i": i, "prompt": i % len(prompts), "same_words": same, "identical": text == ref, "flags": flags, "text": text})
            early, other, ok, line = summarize(args, n, rows)
            report["ok"] &= ok
            report["waves"].append({"requests": n, "identical": sum(r["identical"] for r in rows), "early": early, "garbage": other, "ok": ok,
                                    "rows": rows})
            print(line)
    if args.json:
        json.dump(report, open(args.json, "w"), indent=1)
    print(f"textcheck: {'clean' if report['ok'] else 'GARBAGE'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
