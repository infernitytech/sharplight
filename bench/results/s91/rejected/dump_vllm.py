# Dump vLLM-quantized operands (per-token x, per-block w; UE8M0 scales) and time CUTLASS's f32-scale blockwise GEMM on them.
import sys, torch, numpy as np
from vllm import _custom_ops as ops
from vllm.third_party.deep_gemm.utils.math import per_token_cast_to_fp8, per_block_cast_to_fp8
torch.manual_seed(0)
out_dir = sys.argv[1]
M = 2048
def timeit(f):
    for _ in range(3): f()
    torch.cuda.synchronize()
    best = 1e9
    for i in range(20):
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
for K, N, name in [(4096, 24576, "gu"), (12288, 4096, "dn")]:
    for ue in (True, False):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        xq, xs = per_token_cast_to_fp8(x, use_ue8m0=ue)
        wq, ws = per_block_cast_to_fp8(w, use_ue8m0=ue)
        sa = xs.t().contiguous().t(); sb = ws.t()
        b, m = timeit(lambda: ops.cutlass_scaled_mm(xq, wq.t(), sa, sb, torch.bfloat16))
        print(f"{name} ue8m0={ue} cutlass f32-scale GEMM: burst {b:.3f} ms, sustained {m:.3f} ms", flush=True)
        if not ue:
            tag = f"{out_dir}/{name}"
            xq.view(torch.uint8).cpu().numpy().tofile(tag + "_xc.bin"); xs.float().cpu().numpy().tofile(tag + "_xs.bin")
            wq.view(torch.uint8).cpu().numpy().tofile(tag + "_wc.bin"); ws.float().contiguous().cpu().numpy().tofile(tag + "_ws.bin")
            print(name, xq.shape, xs.shape, wq.shape, ws.shape, flush=True)
