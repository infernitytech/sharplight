# Per-shape timing of vLLM's default FP8 GEMM on sm_120 (DeepGEMM sm120_fp8_fp4_gemm_1d1d with UE8M0 scales) and of CUTLASS's
# block-scaled kernel (cutlass_scaled_mm, f32 scales): burst min after an L2 flush and sustained mean (docs/benchmarks.md §86).
# usage: CUDA_VISIBLE_DEVICES=<i> <venv with vLLM>/bin/python -I vllm_dg_gemm.py [rows ...]
import sys, torch
from vllm import _custom_ops as ops
from vllm.utils import deep_gemm as dg
from vllm.third_party.deep_gemm.utils.math import per_token_cast_to_fp8, per_block_cast_to_fp8
torch.manual_seed(0)
shapes = [(4096, 4096, "q/o"), (4096, 1024, "k/v"), (4096, 6144, "qkv"), (4096, 24576, "gate+up"), (12288, 4096, "down")]
def timeit(f):
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
    return best, e0.elapsed_time(e1) / reps
for M in [int(a) for a in sys.argv[1:]] or [2048]:
    for K, N, name in shapes:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        xq, xs = per_token_cast_to_fp8(x, use_ue8m0=True)
        wq, ws = per_block_cast_to_fp8(w, use_ue8m0=True)
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        fl = 2.0 * M * N * K
        b, m = timeit(lambda: dg.fp8_gemm_nt((xq, xs), (wq, ws), out, is_deep_gemm_e8m0_used=True))
        ref = x.float() @ w.float().t()
        err = ((out.float() - ref).norm() / ref.norm()).item()
        print(f"M={M:5d} {name:8s} K={K:5d} N={N:5d} deepgemm: burst {b:.3f} ms {fl / b / 1e9:.0f} TFLOPS, sustained {m:.3f} ms {fl / m / 1e9:.0f} TFLOPS, rel err {err:.4f}", flush=True)
        sa = xs.t().contiguous().t()   # column-major [M, K/128]
        sb = ws.t()                    # [K/128, N/128]
        b, m = timeit(lambda: ops.cutlass_scaled_mm(xq, wq.t(), sa, sb, torch.bfloat16))
        print(f"M={M:5d} {name:8s} K={K:5d} N={N:5d} cutlass:  burst {b:.3f} ms {fl / b / 1e9:.0f} TFLOPS, sustained {m:.3f} ms {fl / m / 1e9:.0f} TFLOPS", flush=True)
