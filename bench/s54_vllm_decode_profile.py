#!/usr/bin/env python3
"""Per-kernel GPU time of vLLM's batched decode step (docs/benchmarks.md section 54): N sequences with random-token prompts of
--depth tokens generate --steps tokens each; the torch profiler (CUPTI kernel activity, no performance counters needed) records
the second, warm run. Kernels are summed by name over the decode steps (steps with one row per sequence), divided by the step
count, and printed heaviest first with the step's total.

With --words the prompts are random English words (the vocabulary of loadgen.py, the serving benchmark's prompts) tokenized and cut
to --depth tokens instead of random token ids (section 64: random-token prompts make the generated text, and so the expert routing of
the decode steps, more concentrated than serving text). The distinct generated texts and the most common one are printed.

With --prefill N (section 56) it profiles instead the prefill of --seqs prompts of N random tokens (one output token each): the
ranges annotated "execute_context_<seqs>(<N * seqs>)_generation_0(0)", kernels summed per prefill step.

usage: CUDA_VISIBLE_DEVICES=1 python s54_vllm_decode_profile.py <model dir> [--seqs 64] [--depth 256] [--steps 64] [--top 40]
       [--prefill N]
"""
import argparse
import glob
import gzip
import json
import os
import random
import tempfile
from collections import defaultdict


def WORDS():
    """The prompt vocabulary of loadgen.py (read from its source: importing it would run its command line)."""
    import ast
    import re
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "loadgen.py")).read()
    return ast.literal_eval(re.search(r"WORDS = (\(.*?\))\.split\(\)", src, re.S).group(1)).split()


def run_vllm(a):
    """Runs the workload twice and returns the trace of the second (warm) run."""
    trace_dir = tempfile.mkdtemp(prefix="vllm-prof-", dir=os.environ.get("TMPDIR"))
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = LLM(a.model, max_model_len=a.ctx, gpu_memory_utilization=0.85, max_num_seqs=max(a.seqs, 64), enable_prefix_caching=False,
              profiler_config={"profiler": "torch", "torch_profiler_dir": trace_dir})
    rng = random.Random(1)
    params = SamplingParams(max_tokens=1 if a.prefill else a.steps, ignore_eos=True, temperature=0)
    depth = a.prefill or a.depth
    tok = llm.get_tokenizer()

    def prompt():
        if not a.words:
            return [rng.randrange(1000, 30000) for _ in range(depth)]
        words = WORDS()
        ids = []
        while len(ids) < depth:
            ids += tok.encode(" ".join(rng.choice(words) for _ in range(64)) + " ", add_special_tokens=False)
        return ids[:depth]

    for profiled in (False, True):
        prompts = [TokensPrompt(prompt_token_ids=prompt()) for _ in range(a.seqs)]
        if profiled:
            llm.start_profile()
        outs = llm.generate(prompts, params, use_tqdm=False)
        if profiled:
            llm.stop_profile()
            texts = [o.outputs[0].text for o in outs]
            common = max(set(texts), key=texts.count)
            print(f"{len(set(texts))} distinct generations of {len(texts)}; most common ({texts.count(common)}x): {common[:120]!r}")
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
    ap.add_argument("--words", action="store_true", help="random-word prompts (the serving benchmark's) instead of random token ids")
    ap.add_argument("--prefill", type=int, default=0, help="profile the prefill of --seqs prompts of this many tokens instead")
    a = ap.parse_args()

    if a.trace:
        path = a.trace
    else:
        path = run_vllm(a)
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        trace = json.load(f)
    events = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    # Decode steps: the GPU ranges vLLM annotates "execute_context_0(0)_generation_<seqs>(<seqs>)" (no prompt rows).
    marker = (f"execute_context_{a.seqs}({a.prefill * a.seqs})_generation_0(0)" if a.prefill
              else f"execute_context_0(0)_generation_{a.seqs}({a.seqs})")
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
    if n == 0:
        names = sorted({e.get("name") for e in trace["traceEvents"] if e.get("cat") == "gpu_user_annotation"})
        print("no range named", marker, "; annotated ranges:", names[:20])
        return
    print(f"{n} {'prefill' if a.prefill else 'decode'} steps of {a.seqs} sequences: GPU time per step {span / 1000:.2f} ms (annotated range), kernel sum {total / 1000:.2f} ms")
    for name, us in sorted(per.items(), key=lambda kv: -kv[1])[: a.top]:
        print(f"{us / n / 1000:8.3f} ms {100 * us / n / total:5.1f}% {count[name] / n:6.1f}x  {name[:150]}")


if __name__ == "__main__":
    main()
