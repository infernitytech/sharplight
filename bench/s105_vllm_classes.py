#!/usr/bin/env python3
"""Per-class times of a vLLM step profile (output of bench/s54_vllm_decode_profile.py), docs/benchmarks.md section 105.

usage: s105_vllm_classes.py <s54 output>...

Sums the per-step kernel times of each output by class (GEMM, MoE, attention, norms, quantization, activation / rope / sampling and
the rest) from the kernel names, so they line up with the classes of bench/s105_roofline.py. The LM head is a GEMM like the others
here (vLLM's kernel names do not carry shapes).
"""
import re, sys

CLASSES = [
    ("moe", re.compile(r"moe|topk|expert|align_block", re.I)),
    ("attention", re.compile(r"flash|attn|attention|kernel_mha|paged|decode_with|prefill_with|merge_state|reshape_and_cache", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|xmma|nvjet|wgmma|marlin|matmul|splitKreduce|split_k_reduce", re.I)),
    ("norm", re.compile(r"norm", re.I)),
    ("quant", re.compile(r"quant|scaled_fp4|cvt", re.I)),
    ("act_rope_sample", re.compile(r"act_and_mul|silu|rotary|rope|sampl|argmax|softmax|gather|index|scatter|copy|elementwise|reduce", re.I)),
]
LINE = re.compile(r"^\s*([\d.]+) ms\s+([\d.]+)%\s+([\d.]+)x\s+([\d.]+) us/call\s+(.*)$")


def main():
    for path in sys.argv[1:]:
        head, cls = "", {}
        for line in open(path, encoding="utf-8", errors="replace"):
            if "GPU time per step" in line:
                head = line.strip()
            m = LINE.match(line)
            if not m:
                continue
            ms, name = float(m.group(1)), m.group(5)
            c = next((k for k, rx in CLASSES if rx.search(name)), "other")
            cls[c] = cls.get(c, 0.0) + ms
        total = sum(cls.values())
        print(f"{path}: {head}")
        for c, ms in sorted(cls.items(), key=lambda kv: -kv[1]):
            print(f"    {c:16s} {ms:7.3f} ms {ms / total:6.1%}")


if __name__ == "__main__":
    main()
