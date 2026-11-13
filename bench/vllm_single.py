#!/usr/bin/env python3
"""Single-stream vLLM measurement (offline LLM API). usage: vllm_single.py <model-dir-or-gguf> <name> [--gpu-util 0.25] [--tokenizer X]
pp = time for a prompt of N random tokens with max_tokens=1 (prefix caching off); tg = (t(129 new tokens) - t(1 new token)) / 128 at depth N.
Prints one JSON line per measurement with min/median of --reps repetitions.

Speculative decoding: --spec-method mtp --spec-tokens K passes speculative_config to vLLM (MTP heads of the checkpoint). Decode
is then measured on a natural-language prompt (--text-prompt is implied, padded with random tokens to the depth), because the
acceptance rate on continuations of random tokens is not representative; the draft / accepted token counters of the engine are
reported per point (acceptance rate = accepted / drafted, mean accepted length = 1 + accepted / drafts).
--points pp,tg selects what is measured (default both)."""
import argparse, json, random, statistics, time
from vllm import LLM, SamplingParams

TEXT = ("Write a detailed essay about the history of the printing press and its influence on science, religion and politics in Europe. "
        "Explain the technical inventions involved, name the most important people and give concrete examples of books that changed the world.\n\n")


def spec_counters(llm):
    """Sum of the vLLM spec-decode counters (drafts, draft tokens, accepted tokens), or None without speculation."""
    try:
        out = {}
        for m in llm.get_metrics():
            if m.name.startswith("vllm:spec_decode_num_") and hasattr(m, "value"):
                out[m.name] = out.get(m.name, 0) + m.value
        return out or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("name"); ap.add_argument("--gpu-util", type=float, default=0.25)
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--tokenizer"); ap.add_argument("--attn", help="force a vLLM attention backend, e.g. FLASH_ATTN"); ap.add_argument("--kv"); ap.add_argument("--out"); ap.add_argument("--max-len", type=int, default=4608); ap.add_argument("--depth", type=int, default=4096, help="context depth for the deep measurements (OLMoE only has 4096 positions)")
    ap.add_argument("--spec-method", help="speculative decoding method, e.g. mtp"); ap.add_argument("--spec-tokens", type=int, default=1)
    ap.add_argument("--text-prompt", action="store_true", help="decode after a natural-language prompt instead of random tokens")
    ap.add_argument("--points", default="pp,tg")
    a = ap.parse_args()
    kw = {}
    if a.attn: kw["attention_backend"] = a.attn
    if a.kv: kw["kv_cache_dtype"] = a.kv
    if a.spec_method:
        kw["speculative_config"] = {"method": a.spec_method, "num_speculative_tokens": a.spec_tokens}
        kw["disable_log_stats"] = False
        a.text_prompt = True
    t0 = time.time()
    llm = LLM(a.model, tokenizer=a.tokenizer, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_len, enable_prefix_caching=False, seed=0, max_num_seqs=16, **kw)
    load_s = time.time() - t0
    rng = random.Random(1)
    text_ids = llm.get_tokenizer().encode(TEXT)

    def prompt(n, text=False):
        if text:
            pad = [rng.randrange(1000, 30000) for _ in range(max(0, n - len(text_ids)))]
            return {"prompt_token_ids": pad + text_ids}
        return {"prompt_token_ids": [rng.randrange(1000, 30000) for _ in range(n)]}

    def timed(n, new, text=False):
        sp = SamplingParams(temperature=0, max_tokens=new, min_tokens=new, ignore_eos=True)
        p = [prompt(n, text)]
        t = time.perf_counter(); llm.generate(p, sp, use_tqdm=False); return time.perf_counter() - t
    res = {"name": a.name, "load_s": load_s, "spec": kw.get("speculative_config"), "results": {}}
    for _ in range(3): timed(512, 8)  # warm-up (CUDA graphs, autotune)
    pts = a.points.split(",")
    if "pp" in pts:
        for n in (512, 2048, a.depth):
            timed(n, 1)
            pp = [timed(n, 1) for _ in range(a.reps)]
            res["results"][f"pp{n}"] = {"tok_s_median": n / statistics.median(pp), "tok_s_min": n / max(pp), "tok_s_max": n / min(pp)}
    if "tg" in pts:
        for d in (16, a.depth):
            d = max(d, len(text_ids) + 1) if a.text_prompt else d
            timed(d, 129, a.text_prompt)
            c0 = spec_counters(llm)
            t1 = [timed(d, 1, a.text_prompt) for _ in range(a.reps)]; t129 = [timed(d, 129, a.text_prompt) for _ in range(a.reps)]
            c1 = spec_counters(llm)
            tg = [128 / (b - a_) for a_, b in zip(t1, t129)]
            r = {"tok_s_median": statistics.median(tg), "tok_s_min": min(tg), "tok_s_max": max(tg), "depth": d}
            if c0 is not None and c1 is not None:
                delta = {k: c1[k] - c0.get(k, 0) for k in c1}
                r["spec"] = delta
                drafted = delta.get("vllm:spec_decode_num_draft_tokens", 0); acc = delta.get("vllm:spec_decode_num_accepted_tokens", 0)
                drafts = delta.get("vllm:spec_decode_num_drafts", 0)
                if drafted: r["acceptance_rate"] = acc / drafted
                if drafts: r["mean_accepted_length"] = 1 + acc / drafts
            res["results"][f"tg128@d{d}"] = r
    print("RESULT " + json.dumps(res))
    if a.out: json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
