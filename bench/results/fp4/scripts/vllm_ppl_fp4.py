# Perplexity of vLLM on the token ids PplBench wrote (--ids): sequences of --seq tokens, each scored with prompt_logprobs (the
# log-probability of every prompt token after the first), the same predictions PplBench averages (docs/benchmarks.md section 86).
# usage: CUDA_VISIBLE_DEVICES=<i> <venv with vLLM>/bin/python -I vllm_ppl.py <model> <ids file> [--seq 2048] [--gpu-util 0.6]
import argparse, math
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("ids")
ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--gpu-util", type=float, default=0.6)
a = ap.parse_args()
from vllm import LLM, SamplingParams, TokensPrompt
ids = [int(x) for x in open(a.ids).read().split(",")]
seqs = [ids[i * a.seq:(i + 1) * a.seq] for i in range(len(ids) // a.seq)]
llm = LLM(a.model, max_model_len=a.seq + 16, gpu_memory_utilization=a.gpu_util, enable_prefix_caching=False, linear_backend="cutlass")
outs = llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs], SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0), use_tqdm=False)
total, count = 0.0, 0
for i, (s, o) in enumerate(zip(seqs, outs)):
    nll = 0.0
    for p in range(1, len(s)):
        nll -= o.prompt_logprobs[p][s[p]].logprob
    print(f"sequence {i}: {len(s) - 1} tokens, mean NLL {nll / (len(s) - 1):.5f}, PPL {math.exp(nll / (len(s) - 1)):.4f}")
    total += nll; count += len(s) - 1
print(f"vLLM {len(seqs)} sequences of {a.seq} tokens: {count} predictions, mean NLL {total / count:.5f}, PPL {math.exp(total / count):.4f}")
