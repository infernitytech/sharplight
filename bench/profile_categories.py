#!/usr/bin/env python3
"""Groups a SHARPLIGHT_VK_PROFILE report (BatchBench --profile or `sharplight --bench` with the profile on) into op categories.
usage: profile_categories.py <profile.txt> [section substring, e.g. "batch 64" or "prefill T=2048"]
Prints, per profile section, the summed GPU time and share of: dense GEMM / matvec, GEMM glue (activation conversion /
quantization, split-K reduction), MoE expert matmuls and routing, attention, recurrent (delta rule, conv1d, scans),
norms / elementwise / rope / cache writes, sampling, other."""
import re, sys

CATS = [
    ("GEMM glue (to_f16 / quantize / split-K reduce)", r"to_f16|quantize_fp8|quantize_q8|splitk_reduce|x→f16|x→fp8|x→q8"),
    ("MoE (expert matmuls, routing, gather)", r"matmul-id|moe-|moe_|topk|softmax"),
    ("dense GEMM / matvec", r"^matmul|^matmul2"),
    ("attention", r"^attention"),
    ("recurrent (delta rule, conv1d, scan)", r"delta_rule|conv1d|ssm|scan"),
    ("sampling", r"^sample"),
    ("norms / elementwise / rope / cache", r"Norm|fused\[|rope|cache_write|ReduceSum|gather|^Rms"),
]

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else None
section, rows = None, {}
order = []
for line in open(path, encoding="utf-8"):
    if line.startswith("---"):
        section = line.strip("- \n"); order.append(section); rows[section] = []
        continue
    m = re.match(r"\s+([\d.]+) ms\s+[\d.]+%\s+(\d+)x\s+min\s+[\d.]+ us\s+(.*)$", line)
    if m and section:
        rows[section].append((float(m.group(1)), m.group(3)))
for s in order:
    if want and want not in s:
        continue
    total = sum(ms for ms, _ in rows[s])
    if total == 0:
        continue
    acc = {}
    for ms, name in rows[s]:
        cat = next((c for c, rx in CATS if re.search(rx, name)), "other")
        acc[cat] = acc.get(cat, 0) + ms
    print(f"{s}: total {total:.1f} ms")
    for c, ms in sorted(acc.items(), key=lambda x: -x[1]):
        print(f"  {ms:9.2f} ms  {100 * ms / total:5.1f}%  {c}")
