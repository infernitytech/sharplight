#!/usr/bin/env python3
"""Closed-loop load generator for OpenAI-compatible /v1/completions (streaming).

N workers each send --per-worker requests back to back (so N requests are in flight at any time). Prompts are random
word sequences (unique per request, so prefix caches cannot help) of about --prompt-words words (~256 tokens for the
default). Reports aggregate output tok/s, TTFT p50/p95, inter-token latency (mean of per-request mean gaps, plus p50/p95
of all gaps), request latency and the actual prompt/completion token counts taken from the server's usage chunk.

Mixed load (chunked prefill): with --long-prompt-words W --long-every K every K-th request of a worker sends a W-word
prompt; the ITL p99 / max of the other streams then show how much a long prefill stalls running decodes.

Shared prefix: with --shared-prefix-words W every prompt starts with the same W-word text (a system prompt), so a prefix cache
that shares KV between concurrent requests can skip it.

Natural text: with --natural every prompt is a short instruction ("Write a detailed essay about ...") with random topic words,
so the output is ordinary prose (needed for speculative decoding, whose acceptance on random-word continuations is not
representative). --prompt-words then has no effect.

Server metrics: with --metrics the Prometheus text of <url>/metrics is read right before and after the timed part (after the
warm-up) and the difference of every counter / histogram sum / histogram count is stored as "metrics_delta" in the JSON
(sharplight_step_duration_seconds_sum, vllm:spec_decode_num_accepted_tokens_total, ...), and the cumulative buckets of
sharplight_step_rows (steps per row range).

Length mix: with --length-mix every request draws its prompt length (words) and its output length (max_tokens) from
log-normal distributions (--mix-prompt-median / --mix-output-median, sigma --mix-sigma, clipped to [16, --mix-prompt-max] and
[16, --mix-output-max]; defaults median 200 / 150, sigma 0.9 / 0.8, means ~300 / ~205, like chat traces), seeded per worker in
the closed loop and per arrival in the open loop, so every engine gets the same requests.

Gauges: with --gauges the server's /metrics is polled every 0.5 s during the timed part and the mean / max of the running,
waiting and KV-usage gauges (sharplight_active_sequences / queue_depth / kv_utilization / kv_active_tokens, vllm:num_requests_running /
num_requests_waiting / kv_cache_usage_perc) are stored as "gauges".

Arrivals: by default all N workers start together (closed loop), so with equal prompt and output lengths the streams finish
together and the next prompts arrive in waves. --ramp S starts worker i after i * S / N seconds (closed loop, staggered).
--rate R makes the load open loop: --concurrency x --per-worker requests are sent at Poisson arrivals of R requests per second
(seeded), at most --max-concurrency (default --concurrency) in flight; a request that finds the limit reached waits for a free
slot, and its TTFT counts from its arrival ("ttft_*" include that queueing; "send_ttft_*" count from the send). The JSON
records the mode ("arrival": closed / ramp / poisson).

usage: loadgen.py --url http://127.0.0.1:8080 --model NAME -c 16 [--per-worker 3] [--max-tokens 128] [--json out.json]
"""
import argparse, asyncio, json, random, re, statistics, sys, time
import httpx

WORDS = ("time year people way day man thing woman life child world school state family student group country problem hand part "
         "place case week company system program question work government number night point home water room mother area money story "
         "fact month lot right study book eye job word business issue side kind head house service friend father power hour game line "
         "end member law car city community name president team minute idea kid body information back parent face others level office "
         "door health person art war history party result change morning reason research girl guy moment air teacher force education").split()

PREFIX = ""
NATURAL = False
TASKS = ("Write a detailed essay about the role of {a} and {b} in modern {c}. Explain the history, give concrete examples and discuss open questions.",
         "Explain step by step how a {a} could help a {b} solve a problem with their {c}. Be specific and practical.",
         "Write a long story about a {a} who finds a {b} in the {c}. Describe the characters and the setting in detail.",
         "Compare {a} and {b} from the point of view of a {c}. List advantages, disadvantages and a final recommendation.")

def make_prompt(rng, words):
    if NATURAL:
        return PREFIX + rng.choice(TASKS).format(a=rng.choice(WORDS), b=rng.choice(WORDS), c=rng.choice(WORDS)) + "\n\n"
    return PREFIX + " ".join(rng.choice(WORDS) for _ in range(words))

_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)$')

async def scrape(client, url):
    """Prometheus text -> {series: value} for counters (_total), histogram/summary sums and counts."""
    try:
        r = await client.get(url + "/metrics")
        out = {}
        for line in r.text.splitlines():
            m = _SAMPLE.match(line.strip())
            if not m: continue
            name, labels, val = m.group(1), m.group(2) or "", m.group(3)
            # step-size histogram buckets too (how many serving steps fall in each row range)
            bucket = name == "sharplight_step_rows_bucket"
            if not (name.endswith("_total") or name.endswith("_sum") or name.endswith("_count") or bucket): continue
            # drop per-model / per-engine labels so the keys stay stable; keep position labels (spec decode per position)
            keep = ",".join(x for x in labels.strip("{}").split(",") if x.startswith("position=") or x.startswith("reason=") or x.startswith("finished_reason=")
                            or (bucket and x.startswith("le=")))
            key = name + ("{" + keep + "}" if keep else "")
            out[key] = out.get(key, 0.0) + float(val)
        return out
    except Exception as e:  # metrics are optional
        return {"error": str(e)}

GAUGES = ("sharplight_active_sequences", "sharplight_queue_depth", "sharplight_kv_utilization", "sharplight_kv_active_tokens",
          "sharplight_kv_capacity_tokens", "sharplight_slot_utilization", "vllm:num_requests_running", "vllm:num_requests_waiting",
          "vllm:kv_cache_usage_perc")

async def sample_gauges(client, url, stop, samples):
    """Polls the gauges of GAUGES every 0.5 s until stop is set; samples[name] collects the summed values per poll."""
    while not stop.is_set():
        try:
            r = await client.get(url + "/metrics")
            seen = {}
            for line in r.text.splitlines():
                m = _SAMPLE.match(line.strip())
                if m and m.group(1) in GAUGES:
                    seen[m.group(1)] = seen.get(m.group(1), 0.0) + float(m.group(3))
            for k, v in seen.items():
                samples.setdefault(k, []).append(v)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), 0.5)
        except asyncio.TimeoutError:
            pass

def lengths(rng, args):
    """(prompt words, max tokens) of one request: fixed, or drawn from the log-normal length mix."""
    if not args.length_mix:
        return args.prompt_words, args.max_tokens
    import math
    pw = int(min(args.mix_prompt_max, max(16, rng.lognormvariate(math.log(args.mix_prompt_median), args.mix_sigma))))
    mt = int(min(args.mix_output_max, max(16, rng.lognormvariate(math.log(args.mix_output_median), args.mix_sigma_output))))
    return pw, mt

def pct(xs, p):
    if not xs: return float("nan")
    xs = sorted(xs); k = (len(xs) - 1) * p / 100; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

async def one(client, args, prompt, max_tokens=None):
    body = {"model": args.model, "prompt": prompt, "max_tokens": max_tokens or args.max_tokens, "temperature": args.temperature, "stream": True,
            "stream_options": {"include_usage": True}}
    if args.top_p < 1: body["top_p"] = args.top_p
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
        if args.length_mix:
            pw, mt = lengths(rng, args)
            r = await one(client, args, make_prompt(rng, pw), mt)
        else:
            r = await one(client, args, make_prompt(rng, args.long_prompt_words if long else args.prompt_words))
        r["long"] = long
        out.append(r)

async def open_loop(client, args, rng, out):
    """Poisson arrivals of args.rate requests per second, at most args.max_concurrency in flight."""
    total = args.concurrency * args.per_worker
    arrivals = random.Random(args.seed * 7919 + 1)
    slots = asyncio.Semaphore(args.max_concurrency or args.concurrency)

    async def send(index, arrival, request):
        async with slots:
            sent = time.perf_counter()
            if request is None:
                r = await one(client, args, make_prompt(rng, args.prompt_words))
            else:
                r = await one(client, args, request[0], request[1])
        if "error" not in r:
            r["send_ttft"] = r["ttft"]
            r["ttft"] += sent - arrival
        r["long"] = False
        out.append(r)

    tasks = []
    start = time.perf_counter()
    due = 0.0
    for index in range(total):
        delay = start + due - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        request = None
        if args.length_mix:
            pw, mt = lengths(rng, args)
            request = (make_prompt(rng, pw), mt)
        tasks.append(asyncio.create_task(send(index, start + due, request)))
        due += arrivals.expovariate(args.rate)
    await asyncio.gather(*tasks)

async def ramped(client, args, rng, out, index):
    await asyncio.sleep(index * args.ramp / args.concurrency)
    await worker(client, args, rng, out, index)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080"); ap.add_argument("--model", default="m")
    ap.add_argument("-c", "--concurrency", type=int, default=4); ap.add_argument("--per-worker", type=int, default=3)
    ap.add_argument("--prompt-words", type=int, default=250); ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--ignore-eos", action="store_true", help="send ignore_eos (vLLM, llama-server)")
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--long-prompt-words", type=int, default=0, help="words of the occasional long prompt (mixed load)")
    ap.add_argument("--long-every", type=int, default=0, help="every K-th request per worker is long (0 = never)")
    ap.add_argument("--shared-prefix-words", type=int, default=0, help="words of a prefix common to all prompts")
    ap.add_argument("--warmup", type=int, default=1); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--json")
    ap.add_argument("--natural", action="store_true", help="short natural-language instructions instead of random words")
    ap.add_argument("--metrics", action="store_true", help="store the change of the server's /metrics over the timed part")
    ap.add_argument("--ramp", type=float, default=0.0, help="closed loop: worker i starts after i * RAMP / N seconds")
    ap.add_argument("--rate", type=float, default=0.0, help="open loop: Poisson arrivals of RATE requests per second (0 = closed loop)")
    ap.add_argument("--max-concurrency", type=int, default=0, help="open loop: requests in flight at most (default --concurrency)")
    ap.add_argument("--length-mix", action="store_true", help="log-normal prompt / output lengths per request (see the docstring)")
    ap.add_argument("--mix-prompt-median", type=float, default=200); ap.add_argument("--mix-output-median", type=float, default=150)
    ap.add_argument("--mix-sigma", type=float, default=0.9); ap.add_argument("--mix-sigma-output", type=float, default=0.8)
    ap.add_argument("--mix-prompt-max", type=int, default=1536); ap.add_argument("--mix-output-max", type=int, default=768)
    ap.add_argument("--gauges", action="store_true", help="sample the running / waiting / KV gauges of /metrics during the timed part")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    global PREFIX, NATURAL
    NATURAL = args.natural
    if args.shared_prefix_words > 0:
        PREFIX = " ".join(random.Random(args.seed + 7).choice(WORDS) for _ in range(args.shared_prefix_words)) + "\n"
    limits = httpx.Limits(max_connections=args.concurrency + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=limits) as client:
        for _ in range(args.warmup):  # concurrent warm-up so kernels/graphs for batch shapes exist
            await asyncio.gather(*[one(client, args, make_prompt(rng, args.prompt_words)) for _ in range(min(args.concurrency, 4))])
        res = []
        m0 = await scrape(client, args.url) if args.metrics else None
        gauge_samples = {}; gauge_stop = asyncio.Event()
        gauge_task = asyncio.create_task(sample_gauges(client, args.url, gauge_stop, gauge_samples)) if args.gauges else None
        t0 = time.perf_counter()
        if args.rate > 0:
            await open_loop(client, args, random.Random(args.seed * 1000), res)
        else:
            run = ramped if args.ramp > 0 else worker
            await asyncio.gather(*[run(client, args, random.Random(args.seed * 1000 + i), res, i) for i in range(args.concurrency)])
        wall = time.perf_counter() - t0
        if gauge_task is not None:
            gauge_stop.set(); await gauge_task
        m1 = await scrape(client, args.url) if args.metrics else None
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
           "shared_prefix_words": args.shared_prefix_words,
           "per_stream_tok_s": 1 / statistics.mean(per_req_itl) if per_req_itl else float("nan"),
           "ttft_p50_ms": pct(ttft, 50) * 1e3, "ttft_p95_ms": pct(ttft, 95) * 1e3,
           "itl_mean_ms": statistics.mean(per_req_itl) * 1e3 if per_req_itl else float("nan"),
           "itl_p50_ms": pct(gaps, 50) * 1e3, "itl_p95_ms": pct(gaps, 95) * 1e3, "itl_p99_ms": pct(gaps, 99) * 1e3,
           "itl_max_ms": max(gaps) * 1e3 if gaps else float("nan"),
           "short_ttft_p95_ms": pct([r["ttft"] for r in short], 95) * 1e3,
           "short_itl_p99_ms": pct(short_gaps, 99) * 1e3, "long_requests": len(ok) - len(short),
           "arrival": "poisson" if args.rate > 0 else ("ramp" if args.ramp > 0 else "closed"), "rate": args.rate, "ramp_s": args.ramp,
           "max_concurrency": (args.max_concurrency or args.concurrency) if args.rate > 0 else args.concurrency,
           "ttft_p99_ms": pct(ttft, 99) * 1e3,
           "send_ttft_p50_ms": pct([r.get("send_ttft", r["ttft"]) for r in ok], 50) * 1e3,
           "send_ttft_p95_ms": pct([r.get("send_ttft", r["ttft"]) for r in ok], 95) * 1e3,
           "latency_p50_s": pct([r["latency"] for r in ok], 50), "error_samples": errs[:3], "natural": args.natural,
           "length_mix": args.length_mix, "total_prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
           "itl_p90_ms": pct(gaps, 90) * 1e3, "ttft_p90_ms": pct(ttft, 90) * 1e3,
           "tpot_mean_ms": statistics.mean([(r["latency"] - r["ttft"]) / (r["completion_tokens"] - 1) * 1e3 for r in ok if r["completion_tokens"] > 1] or [float("nan")])}
    if gauge_samples:
        out["gauges"] = {k: {"mean": statistics.mean(v), "max": max(v), "samples": len(v)} for k, v in gauge_samples.items()}
    if m0 is not None and m1 is not None and "error" not in m0 and "error" not in m1:
        out["metrics_delta"] = {k: m1[k] - m0.get(k, 0.0) for k in m1 if m1[k] - m0.get(k, 0.0) != 0}
    print(json.dumps(out))
    if args.json: json.dump(out, open(args.json, "w"))

asyncio.run(main())
