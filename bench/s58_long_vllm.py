#!/usr/bin/env python3
"""Long-context vLLM measurements of docs/benchmarks.md section 58.8 (offline LLM API, prefix caching off, chunked prefill as vLLM
configures it by default). usage: s58_long_vllm.py <model> <name> --lens 16384,32768 --depths 32768 --streams 1,4 [--max-len N]
[--gpu-util 0.92] [--kv auto|fp8] [--reps 2] [--out f.json]
pp N   = time of one prompt of N random tokens with max_tokens=1 (also its TTFT; median of --reps).
tg@d,s = s concurrent sequences, each a prompt of d random tokens: (t(129 new tokens) - t(1 new token)) / 128 per stream, x s aggregate.
A length that does not fit is reported with its error instead of a time."""
import argparse, json, random, statistics, time
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("name")
    ap.add_argument("--lens", default="16384,32768"); ap.add_argument("--depths", default="32768"); ap.add_argument("--streams", default="1,4")
    ap.add_argument("--max-len", type=int, default=0); ap.add_argument("--gpu-util", type=float, default=0.92); ap.add_argument("--kv")
    ap.add_argument("--reps", type=int, default=2); ap.add_argument("--out")
    a = ap.parse_args()
    lens = [int(x) for x in a.lens.split(",") if x]; depths = [int(x) for x in a.depths.split(",") if x]
    streams = [int(x) for x in a.streams.split(",") if x]
    max_len = a.max_len or max(lens + depths) + 256
    kw = {"kv_cache_dtype": a.kv} if a.kv else {}
    t0 = time.time()
    llm = LLM(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=max_len, enable_prefix_caching=False, seed=0,
              max_num_seqs=max(streams), **kw)
    res = {"name": a.name, "load_s": time.time() - t0, "max_len": max_len, "kv": a.kv or "auto", "results": {}}
    rng = random.Random(1)

    def timed(n, new, count=1):
        sp = SamplingParams(temperature=0, max_tokens=new, min_tokens=new, ignore_eos=True)
        p = [{"prompt_token_ids": [rng.randrange(1000, 30000) for _ in range(n)]} for _ in range(count)]
        t = time.perf_counter(); llm.generate(p, sp, use_tqdm=False); return time.perf_counter() - t

    for _ in range(2):
        timed(512, 8)   # warm-up (CUDA graphs, autotune)
    for n in lens:
        try:
            timed(n, 1)
            pp = [timed(n, 1) for _ in range(a.reps)]
            res["results"][f"pp{n}"] = {"tok_s_median": n / statistics.median(pp), "ttft_s_median": statistics.median(pp)}
        except Exception as e:   # noqa: BLE001 - report what did not fit
            res["results"][f"pp{n}"] = {"error": str(e)[:300]}
        print(json.dumps({f"pp{n}": res["results"][f"pp{n}"]}), flush=True)
    for d in depths:
        for s in streams:
            key = f"tg128@d{d}x{s}"
            try:
                t1 = [timed(d, 1, s) for _ in range(a.reps)]; t129 = [timed(d, 129, s) for _ in range(a.reps)]
                per = [128 / (b - x) for x, b in zip(t1, t129)]
                res["results"][key] = {"tok_s_per_stream": statistics.median(per), "tok_s_aggregate": statistics.median(per) * s}
            except Exception as e:   # noqa: BLE001
                res["results"][key] = {"error": str(e)[:300]}
            print(json.dumps({key: res["results"][key]}), flush=True)
    print("RESULT " + json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
