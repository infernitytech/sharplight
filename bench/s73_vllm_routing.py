"""Records the expert routing of vLLM's batched decode steps (gpt-oss-20b, Marlin MoE) as a routing file for bench/s56_moe's
ROUTES=file replay and bench/s73_marlin_replay.py: one line per MoE layer call of a decode step with --seqs rows, the items per expert.

The workload is s54_vllm_decode_profile.py's --words run (random-word prompts of --depth tokens, greedy, --steps tokens each), so
the routing is the one behind its decode-step profile. vLLM runs in-process and eagerly (no CUDA graphs: the hook must run on every
step; routing does not depend on it) with moe_align_block_size of the Marlin module wrapped to count the top-k ids.

usage: CUDA_VISIBLE_DEVICES=0 python bench/s73_vllm_routing.py <model dir> --out routes.txt [--seqs 64] [--depth 256] [--steps 64]
"""
import argparse
import os
import random
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out", required=True, help="routing file to write")
    ap.add_argument("--seqs", type=int, default=64)
    ap.add_argument("--depth", type=int, default=256)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--experts", type=int, default=32)
    a = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams, TokensPrompt
    import vllm.model_executor.layers.fused_moe.experts.marlin_moe as marlin_moe
    from s54_vllm_decode_profile import WORDS

    lines = []
    original = marlin_moe.moe_align_block_size

    def recording(topk_ids, *args, **kwargs):
        if topk_ids.shape[0] == a.seqs:
            counts = torch.bincount(topk_ids.flatten().long(), minlength=a.experts).tolist()
            lines.append(" ".join(str(c) for c in counts))
        return original(topk_ids, *args, **kwargs)

    marlin_moe.moe_align_block_size = recording
    llm = LLM(a.model, max_model_len=1024, gpu_memory_utilization=0.85, max_num_seqs=max(a.seqs, 64), enable_prefix_caching=False,
              enforce_eager=True)
    rng = random.Random(1)
    tok = llm.get_tokenizer()
    words = WORDS()

    def prompt():
        ids = []
        while len(ids) < a.depth:
            ids += tok.encode(" ".join(rng.choice(words) for _ in range(64)) + " ", add_special_tokens=False)
        return ids[:a.depth]

    params = SamplingParams(max_tokens=a.steps, ignore_eos=True, temperature=0)
    llm.generate([TokensPrompt(prompt_token_ids=prompt()) for _ in range(a.seqs)], params, use_tqdm=False)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    touched = [sum(1 for c in line.split() if c != "0") for line in lines]
    print(f"{len(lines)} layer routings of {a.seqs} rows; experts touched per layer {sum(touched) / max(1, len(touched)):.2f}")


if __name__ == "__main__":
    main()
