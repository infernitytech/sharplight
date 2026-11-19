#!/usr/bin/env python3
"""Per-kernel GPU time of vLLM's batched decode step (docs/benchmarks.md section 54): N sequences with random-token prompts of
--depth tokens generate --steps tokens each; the torch profiler (CUPTI kernel activity, no performance counters needed) records
the second, warm run. Kernels are summed by name over the decode steps (steps with one row per sequence), divided by the step
count, and printed heaviest first with the step's total.

usage: CUDA_VISIBLE_DEVICES=1 python s54_vllm_decode_profile.py <model dir> [--seqs 64] [--depth 256] [--steps 64] [--top 40]
"""
import argparse
import glob
import gzip
import json
import os
import random
import tempfile
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seqs", type=int, default=64)
    ap.add_argument("--depth", type=int, default=256)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--ctx", type=int, default=1024)
    a = ap.parse_args()

    trace_dir = tempfile.mkdtemp(prefix="vllm-prof-", dir=os.environ.get("TMPDIR"))
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(a.model, max_model_len=a.ctx, gpu_memory_utilization=0.85, max_num_seqs=max(a.seqs, 64), enable_prefix_caching=False,
              profiler_config={"profiler": "torch", "torch_profiler_dir": trace_dir})
    rng = random.Random(1)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 30000) for _ in range(a.depth)]) for _ in range(a.seqs)]
    params = SamplingParams(max_tokens=a.steps, ignore_eos=True, temperature=0)
    llm.generate(prompts, params, use_tqdm=False)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 30000) for _ in range(a.depth)]) for _ in range(a.seqs)]
    llm.start_profile()
    llm.generate(prompts, params, use_tqdm=False)
    llm.stop_profile()

    files = sorted(glob.glob(os.path.join(trace_dir, "*.json*")), key=os.path.getmtime)
    path = files[-1]
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        trace = json.load(f)
    events = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    events.sort(key=lambda e: e["ts"])
    # Steps: kernels between gaps of the timeline; the prefill steps are the few long ones at the start (they hold the
    # prompts' rows), the rest are decode steps. Use the steps after the prefill: identify by total duration.
    steps, cur, last_end = [], [], None
    for e in events:
        if last_end is not None and e["ts"] - last_end > 1000:   # > 1 ms idle: a new step
            steps.append(cur)
            cur = []
        cur.append(e)
        last_end = e["ts"] + e["dur"]
    steps.append(cur)
    totals = sorted(sum(e["dur"] for e in s) for s in steps)
    median = totals[len(totals) // 2]
    decode = [s for s in steps if sum(e["dur"] for e in s) < 1.5 * median]
    per = defaultdict(float)
    count = defaultdict(int)
    for s in decode:
        for e in s:
            per[e["name"]] += e["dur"]
            count[e["name"]] += 1
    n = len(decode)
    total = sum(per.values()) / n
    print(f"{len(steps)} steps, {n} decode steps; GPU time per decode step {total / 1000:.2f} ms (kernel sum)")
    for name, us in sorted(per.items(), key=lambda kv: -kv[1])[: a.top]:
        print(f"{us / n / 1000:8.3f} ms {100 * us / n / total:5.1f}% {count[name] / n:6.1f}x  {name[:150]}")


if __name__ == "__main__":
    main()
