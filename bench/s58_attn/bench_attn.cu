// Harness of attn_prefill_native.cu: random Q (f32) and f16 K / V cache, single sequence, n_past = NPAST, T = TQ query rows.
// Checks against a naive f32 reference kernel (NOREF=1: timing only) and times the kernel (minimum of REPS launches, L2 flushed before each).
// nvcc -O3 -arch=sm_120 -std=c++17 -DHQ=64 -DHKV=8 -DD=64 -DDV=64 -DTQ=2048 -DNPAST=0 -DWINDOW=0 -DHAS_SINKS=1 \
//   -DKFILE='"../../src/Sharplight.Shaders/Kernels/attn_prefill_native.cu"' bench_attn.cu -o b && ./b
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef TQ
#define TQ 2048
#endif
#ifndef NPAST
#define NPAST 0
#endif
#ifndef REPS
#define REPS 20
#endif
#ifndef SCALE
#define SCALE 0.125f
#endif
#define QS1 D
#define QS0 (HQ * D)
#define RT_N_PAST 0
#include KFILE

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// One thread per (query row, head): plain f32 softmax over the visible keys.
__global__ void reference(const float* q, const __half* kc, const __half* vc, const float* sinks, float* o, unsigned T, unsigned nPast)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * HQ) return;
    unsigned t = idx / HQ, h = idx % HQ, hk = h / (HQ / HKV);
    unsigned end = nPast + t + 1, beg = 0;
#if WINDOW
    beg = end > WINDOW ? end - WINDOW : 0;
#endif
    float qh[D];
    for (int d = 0; d < D; d++) qh[d] = __half2float(__float2half(q[(size_t)t * QS0 + h * QS1 + d] * SCALE));
    float m = -1e30f;
    for (unsigned j = beg; j < end; j++)
    {
        float s = 0;
        for (int d = 0; d < D; d++) s += qh[d] * __half2float(kc[((size_t)j * HKV + hk) * D + d]);
        m = fmaxf(m, s);
    }
#if HAS_SINKS
    m = fmaxf(m, sinks[h]);
#endif
    float acc[DV];
    for (int d = 0; d < DV; d++) acc[d] = 0;
    float l = 0;
    for (unsigned j = beg; j < end; j++)
    {
        float s = 0;
        for (int d = 0; d < D; d++) s += qh[d] * __half2float(kc[((size_t)j * HKV + hk) * D + d]);
        float pr = expf(s - m);
        l += pr;
        for (int d = 0; d < DV; d++) acc[d] += pr * __half2float(vc[((size_t)j * HKV + hk) * VRS + d]);
    }
#if HAS_SINKS
    l += expf(sinks[h] - m);
#endif
    for (int d = 0; d < DV; d++) o[((size_t)t * HQ + h) * DV + d] = acc[d] / l;
}

int main()
{
    const unsigned T = TQ, nPast = NPAST, ctx = nPast + T;
    std::vector<float> hq((size_t)T * HQ * D), hs(HQ);
    std::vector<__half> hk((size_t)ctx * HKV * D), hv((size_t)ctx * HKV * VRS);
    srand(1);
    auto rnd = [] { return (float)rand() / RAND_MAX * 2.0f - 1.0f; };
    for (auto& x : hq) x = rnd() * 3.0f;
    for (auto& x : hk) x = __float2half(rnd() * 3.0f);
    for (auto& x : hv) x = __float2half(rnd());
    for (auto& x : hs) x = rnd() * 4.0f;
    float *q, *o, *oref, *sinks;
    __half *kc, *vc;
    unsigned* rt;
    CK(cudaMalloc(&q, hq.size() * 4));
    CK(cudaMalloc(&o, (size_t)T * HQ * DV * 4));
    CK(cudaMalloc(&oref, (size_t)T * HQ * DV * 4));
    CK(cudaMalloc(&sinks, HQ * 4));
    CK(cudaMalloc(&kc, hk.size() * 2));
    CK(cudaMalloc(&vc, hv.size() * 2));
    CK(cudaMalloc(&rt, 16));
    CK(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(sinks, hs.data(), HQ * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(kc, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(vc, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(rt, &nPast, 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(o, 0, (size_t)T * HQ * DV * 4));

    FnParams p{};
    p.q = q; p.kc = (const unsigned short*)kc; p.vc = (const unsigned short*)vc; p.o = o; p.part = o; p.sinks = sinks; p.rt = rt;
    p.arrivals = (fn_u32*)o; p.T = T; p.row0 = 0;
    int shared = FN_STAGES * FN_STAGE * 2 + (FN_DS == 2 ? FN_XCH * 4 : 0);
    CK(cudaFuncSetAttribute(attn_prefill_native, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    dim3 grid(HQ, (T + FN_BR - 1) / FN_BR, 1);
    attn_prefill_native<<<grid, FN_THREADS, shared>>>(p);
    CK(cudaGetLastError());
    // NOREF=1: timing only (the reference takes minutes at long contexts).
    bool checked = getenv("NOREF") == nullptr;
    if (checked)
    {
        reference<<<(T * HQ + 127) / 128, 128>>>(q, kc, vc, sinks, oref, T, nPast);
    }
    CK(cudaDeviceSynchronize());
    std::vector<float> a((size_t)T * HQ * DV), b(a.size());
    CK(cudaMemcpy(a.data(), o, a.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), oref, b.size() * 4, cudaMemcpyDeviceToHost));
    double maxErr = 0, sumErr = 0;
    for (size_t i = 0; checked && i < a.size(); i++)
    {
        double e = fabs(a[i] - b[i]);
        if (!(e <= 1e9)) e = 1e9;
        maxErr = fmax(maxErr, e);
        sumErr += e;
    }

    void* flush;
    size_t flushBytes = 128u << 20;
    CK(cudaMalloc(&flush, flushBytes));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best = 1e9;
    for (int r = 0; r < REPS; r++)
    {
        CK(cudaMemsetAsync(flush, r, flushBytes));
        cudaEventRecord(e0);
        attn_prefill_native<<<grid, FN_THREADS, shared>>>(p);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        best = fminf(best, ms);
    }
    // FLOPs of the visible scores (Q·Kᵀ and P·V).
    double pairs = 0;
    for (unsigned t = 0; t < T; t++)
    {
        unsigned end = nPast + t + 1, beg = 0;
#if WINDOW
        beg = end > WINDOW ? end - WINDOW : 0;
#endif
        pairs += end - beg;
    }
    double flops = pairs * HQ * 2.0 * (D + DV);
    printf("HQ %d HKV %d D %d T %u nPast %u window %d warps %d mr %d stages %d: max err %.3g mean %.3g | %.3f ms %.1f TFLOPS\n", HQ, HKV, D, T, nPast, WINDOW,
           FN_WARPS, FN_MR, FN_STAGES, maxErr, sumErr / a.size(), best, flops / best / 1e9);
    return 0;
}
