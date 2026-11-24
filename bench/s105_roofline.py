#!/usr/bin/env python3
"""Per-step roofline of BatchBench profiles (docs/benchmarks.md section 105).

usage: s105_roofline.py --profile bb.txt [--samples clocks.csv --stamps bb.err] --layers 36 --kv-heads 8 --head-dim 128
                        [--window-layers 0 --window 0] [--experts 32 --topk 4] [--depth 256] [--steps 10]  (--depth: the prefilled context; with --stamps each entry's grown context is used) [--bw 825]
                        [--peak-f16 235 --peak-fp8 474] [--json out.json]

Reads the "--- profile mixed D+P" sections of a BatchBench run (--mixed ... --profile 0,1,... --modes greedy), classifies every
kernel line (GEMM, MoE expert GEMM, LM head, attention, norms / elementwise / conversions) and estimates per step the DRAM bytes it
must move (weights once per step, expert weights for the experts a batch of T rows routes to on average, KV rows read by attention,
activations in and out) and the tensor FLOPs (2 * rows * K * N per GEMM, attention's QK and PV). From these and the kernel times it
prints the achieved GB/s against --bw (the measured read peak) and the TFLOPS against the dtype's mma.sync peak, and which one
bounds each class. With --samples (bench/s73_clock_sampler.py CSV with unix_ms) and --stamps (BatchBench --hold-ms stderr) it adds
the mean board power, the median SM clock and the share of samples at the power cap for each entry's sustained window.

The byte counts are estimates from shapes: L2 hits (the same weight tile read for several token tiles stays in L2 only partly)
are not modelled, so a GEMM over several token tiles can show more than --bw when it re-reads weights from L2.
"""
import argparse, json, math, re, statistics, sys

DTYPE_BYTES = {"f16": 2.0, "bf16": 2.0, "f32": 4.0, "f8e4m3_b128g": 1.0 + 4.0 / (128 * 128), "f8e4m3": 1.0,
               "mxfp4r": 4.25 / 8, "mxfp4rx2": 4.25 / 8, "mxfp4": 4.25 / 8, "q8_0": 34 / 32, "q4_0": 18 / 32}
LINE = re.compile(r"^\s*([\d.]+) ms\s+([\d.]+)%\s+(\d+)x\s+min\s+([\d.]+) us\s+(.*)$")
SHAPE = re.compile(r"\[(?:(\d+)x)?(\d+)→([\d+]+)\](x2)?")
ROWS = re.compile(r"\bT=(\d+)")


def classify(name):
    """(class, K, N, matrices, experts, weight dtype) of a profile line's kernel name."""
    m = SHAPE.search(name)
    head = name.split()[0]
    if head in ("matmul", "matmul2", "matmul-id") and m and not any(t in name for t in ("x→", "split-k reduce", "range", "reduce ")):
        experts = int(m.group(1)) if m.group(1) else 0
        k = int(m.group(2))
        n = sum(int(x) for x in m.group(3).split("+"))
        mats = 2 if (m.group(4) or head == "matmul2") else 1
        dtype = next((t for t in name[m.end():].split() if t in DTYPE_BYTES), "f16")
        if experts:
            return "moe", k, n, mats, experts, dtype
        return ("lm_head" if n >= 32768 else "gemm"), k, n, mats, 0, dtype
    if head == "attention":
        if "combine" in name:
            return "attn_combine", 0, 0, 0, 0, ""
        return ("attn_decode" if "attn_decode" in name or " decode" in name else "attn_prompt"), 0, 0, 0, 0, ""
    if head in ("matmul", "matmul2", "matmul-id"):
        return "convert", 0, 0, 0, 0, ""
    if "RmsNorm" in name or "LayerNorm" in name or "fused[" in head:
        return "norm", 0, 0, 0, 0, ""
    if head.startswith("rope"):
        return "rope_cache", 0, 0, 0, 0, ""
    return "other", 0, 0, 0, 0, ""


def sections(path):
    """{entry: (gpu ms of the profiled steps, [lines])} of a BatchBench output."""
    out, cur = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        h = re.match(r"--- profile (?:mixed|batch) (\S+)", line)
        if h:
            cur = h.group(1)
            out[cur] = [None, []]
            continue
        if cur is None:
            continue
        g = re.match(r"GPU time ([\d.]+) ms", line.strip())
        if g:
            out[cur][0] = float(g.group(1))
            continue
        m = LINE.match(line)
        if m:
            out[cur][1].append((float(m.group(1)), int(m.group(3)), m.group(5)))
        elif line.strip() == "" and out[cur][1]:
            cur = None
    return out


def step_times(path):
    """{entry: GPU ms min} from the table rows of a BatchBench --mixed run."""
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\|\s*(\d+)\s*\|\s*(\d+) \((\d+)\)\s*\|.*?,\s*([\d.]+)\s*\|", line)
        if m:
            d, p, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
            spec = f"{d}+{p // n if n else 0}" + (f"x{n}" if n > 1 else "")
            out[spec] = float(m.group(4))
    return out


def power_windows(samples, stamps):
    """{entry: (mean power W, median SM MHz, share at >= 195 W, median memory MHz)} of each --hold-ms window."""
    rows = []
    with open(samples, encoding="utf-8") as f:
        head = f.readline().strip().split(",")
        for line in f:
            v = line.strip().split(",")
            rows.append(dict(zip(head, v)))
    out = {}
    for line in open(stamps, encoding="utf-8", errors="replace"):
        m = re.match(r"\[stamp\] (\S+) (\d+) (\d+) steps (\d+) gpu_ms ([\d.]+)", line)
        if not m:
            continue
        a, b = int(m.group(2)) + 300, int(m.group(3))  # skip the first 300 ms (clock and power ramp)
        w = [r for r in rows if a <= float(r["unix_ms"]) <= b]
        if not w:
            continue
        p = [float(r["power_w"]) for r in w]
        out[m.group(1)] = (statistics.mean(p), statistics.median(float(r["sm_mhz"]) for r in w),
                           sum(1 for x in p if x >= 195) / len(p), statistics.median(float(r["mem_mhz"]) for r in w))
    return out


def entry_depths(stamps, initial, slots):
    """{entry: mean context of its decode rows during its profile}: every step of an entry (1 first + 3 warm-up + reps + the
    --hold-ms steps, then the 10 profiled ones) appends a token to each of its decode slots 0..D-1 (BatchBench --mixed)."""
    ctx = [initial] * slots
    out = {}
    for line in open(stamps, encoding="utf-8", errors="replace"):
        m = re.match(r"\[stamp\] (\S+) \d+ \d+ steps (\d+)", line)
        if not m:
            continue
        d = int(m.group(1).split("+")[0])
        before = 1 + 3 + 10 + int(m.group(2))
        out[m.group(1)] = statistics.mean(ctx[s] + before + 5 for s in range(d)) if d else initial
        for s in range(d):
            ctx[s] += before + 10
    return out


def analyse(args):
    secs = sections(args.profile)
    depths = entry_depths(args.stamps, args.depth, 4096) if args.stamps else {}
    times = step_times(args.profile)
    power = power_windows(args.samples, args.stamps) if args.samples and args.stamps else {}
    kv_row = 2 * args.kv_heads * args.head_dim * 2  # K and V, f16, per token per layer
    result = {}
    for entry, (gpu_total, lines) in secs.items():
        d, rest = entry.split("+")
        p_parts = rest.split("x")
        p, n = int(p_parts[0]), int(p_parts[1]) if len(p_parts) > 1 else (1 if int(p_parts[0]) > 0 else 0)
        rows = int(d) + p * n
        logits_rows = int(d) + n
        steps = args.steps
        depth = depths.get(entry, args.depth)
        cls = {}
        for total_ms, count, name in lines:
            c, k, nn, mats, experts, dtype = classify(name)
            per_step_ms = total_ms / steps
            launches = count / steps
            t = int(ROWS.search(name).group(1)) if ROWS.search(name) else rows
            b = fl = 0.0
            if c == "gemm":
                b = launches * (mats * k * nn * DTYPE_BYTES[dtype] + t * (k + nn * mats) * 2)
                fl = launches * 2.0 * t * k * nn * mats
            elif c == "lm_head":
                b = launches * (k * nn * DTYPE_BYTES[dtype] + logits_rows * nn * 4)
                fl = launches * 2.0 * logits_rows * k * nn
            elif c == "moe":
                routed = t * args.topk
                touched = experts * (1 - (1 - args.topk / experts) ** t)
                b = launches * (touched * k * nn * DTYPE_BYTES[dtype] + routed * (k + nn) * 2)
                fl = launches * 2.0 * routed * k * nn
            elif c == "attn_decode":
                full = args.layers - args.window_layers
                share = (full * depth + args.window_layers * min(depth, args.window or depth)) / args.layers
                b = launches * int(d) * share * kv_row
                fl = launches * int(d) * share * 4.0 * args.kv_heads * args.q_per_kv * args.head_dim
            elif c == "attn_prompt":
                b = launches * p * n * kv_row * 2
                fl = launches * n * 2.0 * p * p * args.kv_heads * args.q_per_kv * args.head_dim  # causal: half of 4 * p^2
            acc = cls.setdefault(c, {"ms": 0.0, "bytes": 0.0, "flops": 0.0, "peak": 0.0})
            acc["ms"] += per_step_ms
            acc["bytes"] += b
            acc["flops"] += fl
            if fl:
                acc["peak"] = max(acc["peak"], args.peak_fp8 if dtype.startswith("f8") else args.peak_f16)
        step_ms = times.get(entry, gpu_total / steps if gpu_total else sum(v["ms"] for v in cls.values()))
        tot_b = sum(v["bytes"] for v in cls.values())
        tot_f = sum(v["flops"] for v in cls.values())
        kern_ms = sum(v["ms"] for v in cls.values())
        r = {"rows": rows, "depth": depth, "step_ms": step_ms, "kernel_ms": kern_ms, "bytes_gb": tot_b / 1e9, "tflop": tot_f / 1e12,
             "gbps": tot_b / (step_ms * 1e-3) / 1e9, "tflops": tot_f / (step_ms * 1e-3) / 1e12,
             "bw_frac": tot_b / (step_ms * 1e-3) / 1e9 / args.bw, "classes": {}}
        for c, v in sorted(cls.items(), key=lambda kv: -kv[1]["ms"]):
            s = v["ms"] * 1e-3
            gb = v["bytes"] / s / 1e9 if s else 0
            tf = v["flops"] / s / 1e12 if s else 0
            r["classes"][c] = {"ms": v["ms"], "share": v["ms"] / kern_ms if kern_ms else 0, "gbps": gb, "bw_frac": gb / args.bw,
                               "tflops": tf, "mma_frac": tf / v["peak"] if v["peak"] else 0}
        if entry in power:
            r["power_w"], r["sm_mhz"], r["at_cap"], r["mem_mhz"] = power[entry]
        result[entry] = r
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True); ap.add_argument("--samples"); ap.add_argument("--stamps")
    ap.add_argument("--layers", type=int, required=True); ap.add_argument("--kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True); ap.add_argument("--q-per-kv", type=int, default=4)
    ap.add_argument("--window-layers", type=int, default=0); ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--experts", type=int, default=0); ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--depth", type=int, default=256); ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--bw", type=float, default=825.0); ap.add_argument("--peak-f16", type=float, default=235.0)
    ap.add_argument("--peak-fp8", type=float, default=474.0); ap.add_argument("--json")
    args = ap.parse_args()
    res = analyse(args)
    for entry, r in res.items():
        pw = f", {r['power_w']:.0f} W, SM {r['sm_mhz']:.0f} MHz, {r['at_cap']:.0%} at cap" if "power_w" in r else ""
        print(f"{entry}: {r['rows']} rows at ~{r['depth']:.0f} tokens, {r['step_ms']:.2f} ms, {r['bytes_gb']:.2f} GB -> {r['gbps']:.0f} GB/s "
              f"({r['bw_frac']:.0%} of {args.bw:.0f}), {r['tflops']:.1f} TFLOPS{pw}")
        for c, v in r["classes"].items():
            print(f"    {c:12s} {v['ms']:7.3f} ms {v['share']:5.1%}  {v['gbps']:5.0f} GB/s ({v['bw_frac']:4.0%})  "
                  f"{v['tflops']:6.1f} TFLOPS ({v['mma_frac']:4.0%} of MMA peak)")
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
