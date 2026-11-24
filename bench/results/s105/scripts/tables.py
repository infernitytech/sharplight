"""Markdown tables for benchmarks.md section 105 from sat/r2, sat/ab, sat/fixnew and sat/cap."""
import glob, json, os, re, statistics, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "<path>"


def runs(d):
    res = {}
    for f in glob.glob(d + "/serve_*_c*_r1.json"):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        res[j["concurrency"]] = j
    return res


def gauge(j, *names):
    g = j.get("gauges", {})
    for n in names:
        if n in g:
            return g[n]
    return {}


def row(j):
    run = gauge(j, "sharplight_active_sequences", "vllm:num_requests_running")
    wait = gauge(j, "sharplight_queue_depth", "vllm:num_requests_waiting")
    kv = gauge(j, "sharplight_kv_utilization", "vllm:kv_cache_usage_perc")
    md = j.get("metrics_delta", {})
    qn = md.get("sharplight_queue_wait_seconds_count") or md.get("vllm:request_queue_time_seconds_count") or 0
    qs = md.get("sharplight_queue_wait_seconds_sum") or md.get("vllm:request_queue_time_seconds_sum") or 0
    pre = sum(v for k, v in md.items() if "preempt" in k and (k.endswith("_total") or k.endswith("_sum")))
    st = j.get("steady_tok_s", float("nan"))
    return (f"{j['agg_tok_s']:.0f} / {st:.0f}", f"{j['ttft_p50_ms']:.0f} / {j['ttft_p99_ms']:.0f}",
            f"{j['itl_p50_ms']:.1f} / {j['itl_p90_ms']:.1f} / {j['itl_p99_ms']:.1f}",
            f"{run.get('mean', float('nan')):.0f} / {wait.get('mean', float('nan')):.1f}", f"{kv.get('max', float('nan')):.2f}",
            f"{1000 * qs / qn if qn else float('nan'):.0f}", f"{pre:.0f}")


def matrix():
    for m in ("qwen", "gptoss", "llama"):
        for load in ("fixed", "mix"):
            a = runs(f"{OUT}/sat/r2/{m}_sharplight_{load}")
            b = runs(f"{OUT}/sat/r2/{m}_vllm_{load}")
            if not a and not b:
                continue
            print(f"\n**{m}, {load}** (ours develop / vLLM 0.33)\n")
            print("| streams | engine | tok/s (all / steady) | TTFT p50 / p99 ms | ITL p50 / p90 / p99 ms | running / waiting (mean) | KV use max | queue ms | preemptions |")
            print("|---|---|---|---|---|---|---|---|---|")
            for c in sorted(set(a) | set(b)):
                for name, r in (("ours", a), ("vLLM", b)):
                    if c in r:
                        print(f"| {c} | {name} | " + " | ".join(row(r[c])) + " |")


def ab():
    print("\n**Serving A/B, length mix, develop vs branch (steady tok/s, ITL p50)**\n")
    for m in ("qwen", "llama", "gptoss"):
        for c in (128, 256):
            vals = {}
            for d in sorted(glob.glob(f"{OUT}/sat/ab/{m}_sharplight_mix_*")):
                tag = os.path.basename(d).split("_")[-1]
                r = runs(d)
                if c in r:
                    vals[tag] = r[c]
            if not vals:
                continue
            base = [v for k, v in vals.items() if k.startswith("base")]
            new = [v for k, v in vals.items() if k.startswith("new")]
            f = lambda xs, k: "/".join(f"{x[k]:.0f}" if k != "itl_p50_ms" else f"{x[k]:.1f}" for x in xs)
            gain = (statistics.mean(x["steady_tok_s"] for x in new) / statistics.mean(x["steady_tok_s"] for x in base) - 1) if base and new else float("nan")
            print(f"| {m} | {c} | {f(base, 'agg_tok_s')} | {f(new, 'agg_tok_s')} | {f(base, 'steady_tok_s')} | {f(new, 'steady_tok_s')} | {gain:+.1%} | {f(base, 'itl_p50_ms')} | {f(new, 'itl_p50_ms')} |")


def fixnew():
    print("\n**Fixed load, branch (steady tok/s, ITL p50)**\n")
    for m in ("qwen", "gptoss", "llama"):
        n = runs(f"{OUT}/sat/fixnew/{m}_sharplight_fixed")
        b = runs(f"{OUT}/sat/r2/{m}_sharplight_fixed")
        v = runs(f"{OUT}/sat/r2/{m}_vllm_fixed")
        for c in (128, 256):
            cells = []
            for r in (b, n, v):
                cells.append(f"{r[c]['agg_tok_s']:.0f} / {r[c].get('steady_tok_s', float('nan')):.0f} ({r[c]['itl_p50_ms']:.1f})" if c in r else "-")
            print(f"| {m} | {c} | " + " | ".join(cells) + " |")


matrix()
ab()
fixnew()
