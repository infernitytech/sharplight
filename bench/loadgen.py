#!/usr/bin/env python3
"""Closed-loop load generator for OpenAI-compatible /v1/completions (streaming).

N workers each send --per-worker requests back to back (so N requests are in flight at any time). Prompts are random
word sequences (unique per request, so prefix caches cannot help) of about --prompt-words words (~256 tokens for the
default). Reports aggregate output tok/s, TTFT p50/p95, inter-token latency (mean of per-request mean gaps, plus p50/p95
of all gaps), request latency and the actual prompt/completion token counts taken from the server's usage chunk.

Mixed load (chunked prefill): with --long-prompt-words W --long-every K every K-th request of a worker sends a W-word
prompt; the ITL p99 / max of the other streams then show how much a long prefill stalls running decodes.

usage: loadgen.py --url http://127.0.0.1:8080 --model NAME -c 16 [--per-worker 3] [--max-tokens 128] [--json out.json]
"""
import argparse, asyncio, json, random, statistics, sys, time
import httpx

WORDS = ("time year people way day man thing woman life child world school state family student group country problem hand part "
         "place case week company system program question work government number night point home water room mother area money story "
         "fact month lot right study book eye job word business issue side kind head house service friend father power hour game line "
         "end member law car city community name president team minute idea kid body information back parent face others level office "
         "door health person art war history party result change morning reason research girl guy moment air teacher force education").split()

def make_prompt(rng, words):
    return " ".join(rng.choice(WORDS) for _ in range(words))

def pct(xs, p):
    if not xs: return float("nan")
    xs = sorted(xs); k = (len(xs) - 1) * p / 100; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

async def one(client, args, prompt):
    body = {"model": args.model, "prompt": prompt, "max_tokens": args.max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    if args.ignore_eos: body["ignore_eos"] = True
    t0 = time.perf_counter(); times = []; usage = None
    async with client.stream("POST", args.url + "/v1/completions", json=body) as r:
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"}
        async for line in r.aiter_lines():
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            j = json.loads(d)
            if j.get("usage"): usage = j["usage"]
            ch = j.get("choices") or []
            if ch and ch[0].get("text"): times.append(time.perf_counter())
    t1 = time.perf_counter()
    if not times: return {"error": "no tokens"}
    gaps = [b - a for a, b in zip(times, times[1:])]
    return {"ttft": times[0] - t0, "latency": t1 - t0, "gaps": gaps, "chunks": len(times),
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": (usage or {}).get("completion_tokens", len(times))}

async def worker(client, args, rng, out, index):
    for k in range(args.per_worker):
        long = args.long_every > 0 and (k + index) % args.long_every == args.long_every - 1
        r = await one(client, args, make_prompt(rng, args.long_prompt_words if long else args.prompt_words))
        r["long"] = long
        out.append(r)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080"); ap.add_argument("--model", default="m")
    ap.add_argument("-c", "--concurrency", type=int, default=4); ap.add_argument("--per-worker", type=int, default=3)
    ap.add_argument("--prompt-words", type=int, default=250); ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--ignore-eos", action="store_true", help="send ignore_eos (vLLM, llama-server)")
    ap.add_argument("--long-prompt-words", type=int, default=0, help="words of the occasional long prompt (mixed load)")
    ap.add_argument("--long-every", type=int, default=0, help="every K-th request per worker is long (0 = never)")
    ap.add_argument("--warmup", type=int, default=1); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--json")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    limits = httpx.Limits(max_connections=args.concurrency + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=limits) as client:
        for _ in range(args.warmup):  # concurrent warm-up so kernels/graphs for batch shapes exist
            await asyncio.gather(*[one(client, args, make_prompt(rng, args.prompt_words)) for _ in range(min(args.concurrency, 4))])
        res = []
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(client, args, random.Random(args.seed * 1000 + i), res, i) for i in range(args.concurrency)])
        wall = time.perf_counter() - t0
    ok = [r for r in res if "error" not in r]; errs = [r["error"] for r in res if "error" in r]
    toks = sum(r["completion_tokens"] for r in ok)
    gaps = [g for r in ok for g in r["gaps"]]
    short = [r for r in ok if not r.get("long")]
    short_gaps = [g for r in short for g in r["gaps"]]
    per_req_itl = [statistics.mean(r["gaps"]) for r in ok if r["gaps"]]
    ttft = [r["ttft"] for r in ok]
    out = {"concurrency": args.concurrency, "requests": len(res), "errors": len(errs), "wall_s": wall,
           "agg_tok_s": toks / wall, "out_tokens": toks, "avg_out_tokens": toks / max(len(ok), 1),
           "avg_prompt_tokens": statistics.mean([r["prompt_tokens"] for r in ok if r["prompt_tokens"]] or [float("nan")]),
           "per_stream_tok_s": 1 / statistics.mean(per_req_itl) if per_req_itl else float("nan"),
           "ttft_p50_ms": pct(ttft, 50) * 1e3, "ttft_p95_ms": pct(ttft, 95) * 1e3,
           "itl_mean_ms": statistics.mean(per_req_itl) * 1e3 if per_req_itl else float("nan"),
           "itl_p50_ms": pct(gaps, 50) * 1e3, "itl_p95_ms": pct(gaps, 95) * 1e3, "itl_p99_ms": pct(gaps, 99) * 1e3,
           "itl_max_ms": max(gaps) * 1e3 if gaps else float("nan"),
           "short_ttft_p95_ms": pct([r["ttft"] for r in short], 95) * 1e3,
           "short_itl_p99_ms": pct(short_gaps, 99) * 1e3, "long_requests": len(ok) - len(short),
           "latency_p50_s": pct([r["latency"] for r in ok], 50), "error_samples": errs[:3]}
    print(json.dumps(out))
    if args.json: json.dump(out, open(args.json, "w"))

asyncio.run(main())
