#!/usr/bin/env python3
"""Prefix-caching workloads for OpenAI-compatible /v1/chat/completions servers (streaming), docs/benchmarks.md section 48.

Workloads (closed loop: --concurrency workers, each sending its requests back to back):
  shared     every request is [system: one shared ~--system-tokens system prompt, user: a unique short question]
  multiturn  every worker holds one conversation of --turns turns; each turn resends the whole history (earlier user messages and
             the replies the server generated) plus a new user message
  rag        --docs shared documents of ~--doc-tokens tokens; every request quotes two of them (random order) and asks a question,
             so only the system prompt and the first document form a cacheable prefix

Reports TTFT p50 / p99, request latency, aggregate output tok/s, and the prefix-cache hit rate from the usage chunk
(usage.prompt_tokens_details.cached_tokens / prompt_tokens; vLLM needs --enable-prompt-tokens-details). Text is random words from
a fixed list (seeded), so both engines see identical requests; thinking is turned off (chat_template_kwargs).

usage: prefix_bench.py --url http://127.0.0.1:8080 --workload shared -c 16 [--per-worker 4] [--json out.json]
"""
import argparse, asyncio, json, random, statistics, sys, time
import httpx

WORDS = ("time year people way day man thing woman life child world school state family student group country problem hand part "
         "place case week company system program question work government number night point home water room mother area money story "
         "fact month lot right study book eye job word business issue side kind head house service friend father power hour game line "
         "end member law car city community name president team minute idea kid body information back parent face others level office "
         "door health person art war history party result change morning reason research girl guy moment air teacher force education").split()

TOKENS_PER_WORD = 1.0   # these words are single tokens for the Qwen3 / Llama 3 tokenizers (with the leading space)


def words(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs); k = (len(xs) - 1) * p / 100; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


async def chat(client, args, messages):
    body = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter(); first = None; usage = None; text = []
    async with client.stream("POST", args.url + "/v1/chat/completions", json=body) as r:
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {(await r.aread())[:300]!r}"}
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    first = first or time.perf_counter()
                    text.append(piece)
    t1 = time.perf_counter()
    if first is None:
        return {"error": "no tokens"}
    u = usage or {}
    cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    return {"ttft": first - t0, "latency": t1 - t0, "prompt_tokens": u.get("prompt_tokens", 0), "cached_tokens": cached,
            "completion_tokens": u.get("completion_tokens", 0), "text": "".join(text)}


async def run_worker(client, args, index, shared, out):
    rng = random.Random(args.seed * 1000 + index)
    if args.workload == "shared":
        for _ in range(args.per_worker):
            out.append(await chat(client, args, [{"role": "system", "content": shared["system"]},
                                                 {"role": "user", "content": "Question: " + words(rng, args.question_words)}]))
    elif args.workload == "multiturn":
        history = [{"role": "system", "content": shared["system_short"]}]
        for _ in range(args.turns):
            history.append({"role": "user", "content": words(rng, args.question_words * 3)})
            r = await chat(client, args, history)
            out.append(r)
            history.append({"role": "assistant", "content": r.get("text", "")})
    else:
        docs = shared["docs"]
        for _ in range(args.per_worker):
            a, b = rng.sample(range(len(docs)), 2)
            user = f"Document 1:\n{docs[a]}\n\nDocument 2:\n{docs[b]}\n\nQuestion: {words(rng, args.question_words)}"
            out.append(await chat(client, args, [{"role": "system", "content": shared["system_short"]}, {"role": "user", "content": user}]))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080"); ap.add_argument("--model", default="m")
    ap.add_argument("--workload", choices=["shared", "multiturn", "rag"], default="shared")
    ap.add_argument("-c", "--concurrency", type=int, default=16); ap.add_argument("--per-worker", type=int, default=4)
    ap.add_argument("--turns", type=int, default=6); ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--system-tokens", type=int, default=1500); ap.add_argument("--question-words", type=int, default=30)
    ap.add_argument("--docs", type=int, default=4); ap.add_argument("--doc-tokens", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--json")
    args = ap.parse_args()
    srng = random.Random(4242)   # shared texts: identical on every engine and round
    shared = {"system": "You are a helpful assistant. Background notes: " + words(srng, int(args.system_tokens / TOKENS_PER_WORD)),
              "system_short": "You are a helpful assistant.",
              "docs": [words(srng, int(args.doc_tokens / TOKENS_PER_WORD)) for _ in range(args.docs)]}
    limits = httpx.Limits(max_connections=args.concurrency + 4, max_keepalive_connections=args.concurrency + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0), limits=limits) as client:
        # Warm-up with a prompt unrelated to the measured ones (compiles kernels, nothing shared with the workload).
        await chat(client, args, [{"role": "user", "content": "warm up " + words(random.Random(7), 50)}])
        out = []
        t0 = time.perf_counter()
        await asyncio.gather(*(run_worker(client, args, i, shared, out) for i in range(args.concurrency)))
        wall = time.perf_counter() - t0
    errors = [r["error"] for r in out if "error" in r]
    ok = [r for r in out if "error" not in r]
    ttft = [r["ttft"] * 1000 for r in ok]
    prompt = sum(r["prompt_tokens"] for r in ok); cached = sum(r["cached_tokens"] for r in ok)
    res = {"workload": args.workload, "concurrency": args.concurrency, "requests": len(ok), "errors": len(errors),
           "first_error": errors[0] if errors else None, "wall_s": wall,
           "output_tok_s": sum(r["completion_tokens"] for r in ok) / wall, "prompt_tokens_mean": prompt / max(1, len(ok)),
           "cached_tokens": cached, "prompt_tokens": prompt, "hit_rate": cached / prompt if prompt else 0.0,
           "ttft_p50_ms": pct(ttft, 50), "ttft_p99_ms": pct(ttft, 99), "ttft_mean_ms": statistics.mean(ttft) if ttft else float("nan"),
           "latency_p50_s": pct([r["latency"] for r in ok], 50)}
    print(json.dumps(res))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
