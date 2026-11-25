# Per-shape timing of vLLM's block-scaled FP8 GEMM (CUTLASS sm120 blockwise via cutlass_scaled_mm), burst min and sustained mean.
import sys, torch
from vllm import _custom_ops as ops
torch.manual_seed(0)
# usage: CUDA_VISIBLE_DEVICES=<i> <venv with vLLM>/bin/python -I vllm_gemm.py
shapes = [(4096, 4096, "q/o"), (4096, 1024, "k/v"), (4096, 6144, "qkv"), (4096, 24576, "gate+up"), (12288, 4096, "down")]
def run(M, K, N):
    a = (torch.randn(M, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    w = (torch.randn(N, K, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    sa = (torch.rand(K // 128, M, device="cuda") + 0.5).t()          # column-major [M, K/128]
    sb = (torch.rand(N // 128, K // 128, device="cuda") + 0.5).t()    # [K/128, N/128]
    f = lambda: ops.cutlass_scaled_mm(a, w.t(), sa, sb, torch.bfloat16)
    for _ in range(3): f()
    torch.cuda.synchronize()
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    best = 1e9
    for i in range(20):
        flush.fill_(i)
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); f(); e1.record(); e1.synchronize()
        best = min(best, e0.elapsed_time(e1))
    reps = int(1500 / best) + 1
    for _ in range(reps // 4): f()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps): f()
    e1.record(); e1.synchronize()
    mean = e0.elapsed_time(e1) / reps
    fl = 2.0 * M * N * K
    return best, fl / best / 1e9, mean, fl / mean / 1e9
for M in (2048, 8192):
    for K, N, name in shapes:
        b, bt, m, mt = run(M, K, N)
        print(f"M={M:5d} {name:8s} K={K:5d} N={N:5d}: burst {b:.3f} ms {bt:.0f} TFLOPS, sustained {m:.3f} ms {mt:.0f} TFLOPS", flush=True)
