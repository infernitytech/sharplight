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
       [--prefill N] [--skip N] [--gpu-util 0.85]
"""
import argparse
import re
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

    llm = LLM(a.model, max_model_len=a.ctx, gpu_memory_utilization=a.gpu_util, max_num_seqs=max(a.seqs, 64), enable_prefix_caching=False,
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
    ap.add_argument("--gpu-util", type=float, default=0.85, help="vLLM's gpu_memory_utilization (long contexts need ~0.92)")
    ap.add_argument("--trace", help="parse this torch trace (.json.gz) instead of running vLLM")
    ap.add_argument("--words", action="store_true", help="random-word prompts (the serving benchmark's) instead of random token ids")
    ap.add_argument("--prefill", type=int, default=0, help="profile the prefill of --seqs prompts of this many tokens instead")
    ap.add_argument("--skip", type=int, default=0,
                    help="leave out the first N decode steps (their routing is more concentrated: section 73; BatchBench warms up 5)")
    a = ap.parse_args()

    if a.trace:
        path = a.trace
    else:
        path = run_vllm(a)
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        trace = json.load(f)
    events = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    ranges = [e for e in trace["traceEvents"] if e.get("cat") == "gpu_user_annotation"]
    if a.prefill:
        # Prefill steps: "execute_context_<seqs>(<rows>)_generation_0(0)". The scheduler may split the prompts over several
        # steps (e.g. 7 + 1 sequences), so every prefill range is reported on its own, largest first (section 86: averaging
        # vLLM's own per-kernel table over the steps mixed 1,792- and 256-row GEMMs).
        names = sorted({e["name"] for e in ranges if re.fullmatch(r"execute_context_\d+\(\d+\)_generation_0\(0\)", e["name"])},
                       key=lambda m: -int(re.search(r"\((\d+)\)", m).group(1)))
    else:
        # Decode steps: the GPU ranges vLLM annotates "execute_context_0(0)_generation_<seqs>(<seqs>)" (no prompt rows).
        names = [f"execute_context_0(0)_generation_{a.seqs}({a.seqs})"]
    if not names:
        print("no prefill range; annotated ranges:", sorted({e.get("name") for e in ranges})[:20])
        return
    for marker in names:
        report(a, marker, ranges, events)


def report(a, marker, ranges, events):
    steps = sorted((e for e in ranges if e.get("name") == marker), key=lambda e: e["ts"])[a.skip:]
    n = len(steps)
    if n == 0:
        print("no range named", marker, "; annotated ranges:", sorted({e.get("name") for e in ranges})[:20])
        return
    per = defaultdict(float)
    count = defaultdict(int)
    for st in steps:
        for e in events:
            if st["ts"] <= e["ts"] < st["ts"] + st["dur"]:
                per[e["name"]] += e["dur"]
                count[e["name"]] += 1
    total = sum(per.values()) / n
    span = sum(st["dur"] for st in steps) / n
    print(f"{n} x {marker} (first {a.skip} left out): GPU time per step {span / 1000:.2f} ms (annotated range), kernel sum {total / 1000:.2f} ms")
    for name, us in sorted(per.items(), key=lambda kv: -kv[1])[: a.top]:
        print(f"{us / n / 1000:8.3f} ms {100 * us / n / total:5.1f}% {count[name] / n:6.1f}x  {us / max(count[name], 1):9.1f} us/call  {name[:150]}")

if __name__ == "__main__":
    main()
