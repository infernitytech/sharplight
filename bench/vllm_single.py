#!/usr/bin/env python3
"""Single-stream vLLM measurement (offline LLM API). usage: vllm_single.py <model-dir-or-gguf> <name> [--gpu-util 0.25] [--tokenizer X]
pp = time for a prompt of N random tokens with max_tokens=1 (prefix caching off); tg = (t(129 new tokens) - t(1 new token)) / 128 at depth N.
Prints one JSON line per measurement with min/median of --reps repetitions."""
import argparse, json, random, statistics, time
from vllm import LLM, SamplingParams
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("name"); ap.add_argument("--gpu-util", type=float, default=0.25)
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--tokenizer"); ap.add_argument("--attn", help="force a vLLM attention backend, e.g. FLASH_ATTN"); ap.add_argument("--kv"); ap.add_argument("--out"); ap.add_argument("--max-len", type=int, default=4608); ap.add_argument("--depth", type=int, default=4096, help="context depth for the deep measurements (OLMoE only has 4096 positions)")
    a = ap.parse_args()
    t0 = time.time()
    llm = LLM(a.model, tokenizer=a.tokenizer, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_len, enable_prefix_caching=False, seed=0, max_num_seqs=16, **({"attention_backend": a.attn} if a.attn else {}), **({"kv_cache_dtype": a.kv} if a.kv else {}))
    load_s = time.time() - t0
    rng = random.Random(1)
    def prompt(n): return {"prompt_token_ids": [rng.randrange(1000, 30000) for _ in range(n)]}
    def timed(n, new):
        sp = SamplingParams(temperature=0, max_tokens=new, min_tokens=new, ignore_eos=True)
        p = [prompt(n)]
        t = time.perf_counter(); llm.generate(p, sp, use_tqdm=False); return time.perf_counter() - t
    res = {"name": a.name, "load_s": load_s, "results": {}}
    for _ in range(3): timed(512, 8)  # warm-up (CUDA graphs, autotune)
    for n in (512, 2048, a.depth):
        timed(n, 1)
        pp = [timed(n, 1) for _ in range(a.reps)]
        res["results"][f"pp{n}"] = {"tok_s_median": n / statistics.median(pp), "tok_s_min": n / max(pp), "tok_s_max": n / min(pp)}
    for d in (16, a.depth):
        timed(d, 129)
        t1 = [timed(d, 1) for _ in range(a.reps)]; t129 = [timed(d, 129) for _ in range(a.reps)]
        tg = [128 / (b - a_) for a_, b in zip(t1, t129)]
        res["results"][f"tg128@d{d}"] = {"tok_s_median": statistics.median(tg), "tok_s_min": min(tg), "tok_s_max": max(tg)}
    print("RESULT " + json.dumps(res))
    if a.out: json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
