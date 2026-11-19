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


def run_vllm(a):
    """Runs the workload twice and returns the trace of the second (warm) run."""
    trace_dir = tempfile.mkdtemp(prefix="vllm-prof-", dir=os.environ.get("TMPDIR"))
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(a.model, max_model_len=a.ctx, gpu_memory_utilization=0.85, max_num_seqs=max(a.seqs, 64), enable_prefix_caching=False,
              profiler_config={"profiler": "torch", "torch_profiler_dir": trace_dir})
    rng = random.Random(1)
    params = SamplingParams(max_tokens=a.steps, ignore_eos=True, temperature=0)
    for profiled in (False, True):
        prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 30000) for _ in range(a.depth)]) for _ in range(a.seqs)]
        if profiled:
            llm.start_profile()
        llm.generate(prompts, params, use_tqdm=False)
        if profiled:
            llm.stop_profile()
    files = sorted(glob.glob(os.path.join(trace_dir, "*.json*")), key=os.path.getmtime)
    return files[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seqs", type=int, default=64)
    ap.add_argument("--depth", type=int, default=256)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--trace", help="parse this torch trace (.json.gz) instead of running vLLM")
    a = ap.parse_args()

    if a.trace:
        path = a.trace
    else:
        path = run_vllm(a)
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        trace = json.load(f)
    events = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    # Decode steps: the GPU ranges vLLM annotates "execute_context_0(0)_generation_<seqs>(<seqs>)" (no prompt rows).
    marker = f"execute_context_0(0)_generation_{a.seqs}({a.seqs})"
    steps = [e for e in trace["traceEvents"] if e.get("cat") == "gpu_user_annotation" and e.get("name") == marker]
    per = defaultdict(float)
    count = defaultdict(int)
    for st in steps:
        for e in events:
            if st["ts"] <= e["ts"] < st["ts"] + st["dur"]:
                per[e["name"]] += e["dur"]
                count[e["name"]] += 1
    n = len(steps)
    total = sum(per.values()) / n
    span = sum(st["dur"] for st in steps) / n
    print(f"{n} decode steps of {a.seqs} sequences: GPU time per step {span / 1000:.2f} ms (annotated range), kernel sum {total / 1000:.2f} ms")
    for name, us in sorted(per.items(), key=lambda kv: -kv[1])[: a.top]:
        print(f"{us / n / 1000:8.3f} ms {100 * us / n / total:5.1f}% {count[name] / n:6.1f}x  {name[:150]}")


if __name__ == "__main__":
    main()
