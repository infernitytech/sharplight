"""Replays recorded MoE routings through vLLM's Marlin MXFP4 expert GEMM (moe_wna16_marlin_gemm, W4A16 with bf16
activations, as gpt-oss-20b runs on this GPU) for a clock and power comparison against bench/s56_moe's ROUTES=file replay.

Usage: python bench/s73_marlin_replay.py ROUTES [--product gate_up|down|both] [--layers 24] [--seconds 6]

ROUTES has one routing per line (items per expert, 32 experts, 64 tokens of top-4: a 64-sequence decode step's layers, e.g.
bench/results/s67/harness/routes64_text.txt). Each routing becomes top-k ids with exactly those counts, aligned as fused_marlin_moe
aligns them (block 16 at 64 tokens), and runs on the weights of layer (line % layers) (distinct copies of random MXFP4 weights in
Marlin's layout, gpt-oss-20b's shapes as vLLM pads them for Marlin: hidden 2880 -> 3072, intermediate 2880 -> 2944, per-expert bias
rows; 9 % more weight bytes than the checkpoint's). All routings are captured in one CUDA
graph; the script prints the best replay (ms per routing) and then replays the graph back to back for --seconds (the sustained
window to sample with s73_clock_sampler.py).
"""

import argparse
import statistics
import time
import types

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin
from vllm.scalar_type import scalar_types

EXPERTS = 32
TOP_K = 4


def read_routes(path: str) -> list[list[int]]:
    routes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            counts = [int(v) for v in line.split()]
            if len(counts) == EXPERTS:
                routes.append(counts)
    return routes


def topk_ids_for(counts: list[int], device: torch.device) -> torch.Tensor:
    """Top-k ids whose per-expert counts are counts: the items of the experts (largest first) fill the slots column by column,
    so an expert's items (at most one per token) land on distinct tokens."""
    tokens = sum(counts) // TOP_K
    order = sorted(range(EXPERTS), key=lambda e: -counts[e])
    flat = [e for e in order for _ in range(counts[e])]
    ids = torch.empty((tokens, TOP_K), dtype=torch.int32)
    for i, e in enumerate(flat):
        ids[i % tokens, i // tokens] = e
    return ids.to(device)


def marlin_layer(device: torch.device, HIDDEN: int, INTERMEDIATE: int):
    w13 = torch.randint(0, 256, (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (EXPERTS, HIDDEN, INTERMEDIATE // 2), dtype=torch.uint8, device=device)
    # E8M0 120 .. 127 (2^-7 .. 1), as gpt-oss's scales and the harness's default.
    s13 = torch.randint(120, 128, (EXPERTS, 2 * INTERMEDIATE, HIDDEN // 32), dtype=torch.uint8, device=device)
    s2 = torch.randint(120, 128, (EXPERTS, HIDDEN, INTERMEDIATE // 32), dtype=torch.uint8, device=device)
    b13 = torch.zeros((EXPERTS, 2 * INTERMEDIATE), dtype=torch.bfloat16, device=device)
    b2 = torch.zeros((EXPERTS, HIDDEN), dtype=torch.bfloat16, device=device)
    layer = types.SimpleNamespace(params_dtype=torch.bfloat16)
    return prepare_moe_mxfp4_layer_for_marlin(layer, w13, w2, s13, s2, b13, b2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("routes", help="routing file (items per expert per line)")
    parser.add_argument("--product", choices=["gate_up", "down", "both"], default="both", help="which GEMM(s) to replay")
    parser.add_argument("--layers", type=int, default=24, help="distinct weight copies")
    parser.add_argument("--seconds", type=float, default=6.0, help="sustained back-to-back replay time")
    parser.add_argument("--hidden", type=int, default=3072, help="hidden size (vLLM's Marlin padding of gpt-oss's 2880)")
    parser.add_argument("--intermediate", type=int, default=2944, help="intermediate size (padded 2880)")
    parser.add_argument("--reps", type=int, default=10, help="timed replays for the best time")
    parser.add_argument("--energy", action="store_true",
                        help="read NVML's total energy counter around the sustained window (--gpu: NVML index) and print energy per "
                             "routing and per weight byte (each touched expert's padded weights read once)")
    parser.add_argument("--gpu", type=int, default=0, help="NVML index of the GPU for --energy")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)
    routes = read_routes(args.routes)
    tokens = sum(routes[0]) // TOP_K
    for counts in routes:
        if sum(counts) != tokens * TOP_K or max(counts) > tokens:
            raise SystemExit("every routing must have the same token count and at most one item per token and expert")

    HIDDEN, INTERMEDIATE = args.hidden, args.intermediate
    first = marlin_layer(device, HIDDEN, INTERMEDIATE)
    layers = [first] + [tuple(t.clone() if t is not None else None for t in first) for _ in range(args.layers - 1)]
    workspace = marlin_make_workspace_new(device, 4)
    block_m = 8
    for candidate in [8, 16, 32, 48, 64]:
        block_m = candidate
        if tokens * TOP_K / EXPERTS / candidate < 0.9:
            break

    hidden = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device=device) * 0.5
    inter = torch.randn((tokens * TOP_K, INTERMEDIATE), dtype=torch.bfloat16, device=device) * 0.5
    out13 = torch.empty((tokens * TOP_K, 2 * INTERMEDIATE), dtype=torch.bfloat16, device=device)
    out2 = torch.empty((tokens * TOP_K, HIDDEN), dtype=torch.bfloat16, device=device)
    aligned = []
    for counts in routes:
        ids = topk_ids_for(counts, device)
        weights = torch.full(ids.shape, 1.0 / TOP_K, dtype=torch.float32, device=device)
        sorted_ids, expert_ids, padded = moe_align_block_size(ids, block_m, EXPERTS, None, ignore_invalid_experts=True)
        aligned.append((weights, sorted_ids, expert_ids, padded))

    def run_all() -> None:
        for i, (weights, sorted_ids, expert_ids, padded) in enumerate(aligned):
            w13, w2, s13, s2, b13, b2 = layers[i % args.layers]
            if args.product in ("gate_up", "both"):
                ops.moe_wna16_marlin_gemm(hidden, out13, w13, b13, s13, None, None, None, workspace, sorted_ids, expert_ids,
                                          padded, weights, moe_block_size=block_m, top_k=TOP_K, mul_topk_weights=False,
                                          b_q_type=scalar_types.float4_e2m1f, size_m=tokens, size_n=2 * INTERMEDIATE,
                                          size_k=HIDDEN, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
            if args.product in ("down", "both"):
                ops.moe_wna16_marlin_gemm(inter, out2, w2, b2, s2, None, None, None, workspace, sorted_ids, expert_ids,
                                          padded, weights, moe_block_size=block_m, top_k=1, mul_topk_weights=True,
                                          b_q_type=scalar_types.float4_e2m1f, size_m=tokens * TOP_K, size_n=HIDDEN,
                                          size_k=INTERMEDIATE, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)

    run_all()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_all()
    graph.replay()
    torch.cuda.synchronize()

    flush = torch.empty(256 << 20, dtype=torch.uint8, device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(args.reps):
        flush.fill_(1)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end))
    print(f"marlin {args.product}: block_m {block_m}, {len(routes)} routings over {args.layers} layer copies: "
          f"{best / len(routes):.4f} ms per routing (best of {args.reps})")

    nvml = None
    if args.energy:
        import pynvml
        pynvml.nvmlInit()
        nvml = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
        warm = time.perf_counter() + 1.0
        while time.perf_counter() < warm:
            graph.replay()
            torch.cuda.synchronize()
        energy0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml)
        wall0 = time.perf_counter()
        clocks = []
    replays = 0
    start.record()
    deadline = time.perf_counter() + args.seconds
    while time.perf_counter() < deadline:
        for _ in range(4):
            graph.replay()
        replays += 4
        torch.cuda.synchronize()
        if nvml is not None:
            clocks.append(pynvml.nvmlDeviceGetClockInfo(nvml, pynvml.NVML_CLOCK_SM))
    end.record()
    end.synchronize()
    if nvml is not None:
        joules = (pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml) - energy0) / 1000.0
        wall = time.perf_counter() - wall0
        per_expert = 0
        if args.product in ("gate_up", "both"):
            per_expert += 2 * INTERMEDIATE * (HIDDEN // 2 + HIDDEN // 32)
        if args.product in ("down", "both"):
            per_expert += HIDDEN * (INTERMEDIATE // 2 + INTERMEDIATE // 32)
        touched = sum(sum(1 for c in counts if c > 0) for counts in routes) / len(routes)
        weight_bytes = touched * per_expert
        per_routing = joules / replays / len(routes)
        print(f"marlin {args.product}: energy: {replays} replays {wall:.2f} s, {joules / wall:.1f} W, {per_routing * 1000:.3f} mJ per routing, "
              f"{weight_bytes / 1e6:.1f} MB weights per routing ({touched:.2f} experts), {per_routing * 1e9 / weight_bytes:.4f} nJ per weight "
              f"byte, SM {statistics.median(clocks):.0f} MHz")
    print(f"marlin {args.product}: sustained {replays} replays: {start.elapsed_time(end) / replays / len(routes):.4f} ms per routing")


if __name__ == "__main__":
    main()
