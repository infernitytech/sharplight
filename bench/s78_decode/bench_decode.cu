// Harness of attn_decode_native.cu at long contexts (docs/benchmarks.md section 78): T sequences of L keys each in a contiguous
// multi-sequence f16 cache (SLOT_CTX = L rows per slot, HKV heads of D), gpt-oss-20b's shape by default (64 query heads, 8 KV heads,
// D = 64, sinks). Times the split kernel alone (minimum and median of REPS launches) and prints the cache read rate; CHECK=1 compares
// the merged partials of a few heads against an f32 reference on the host.
// nvcc -O3 -arch=sm_120 -std=c++17 -DT_ROWS=4 -DL_KEYS=130048 -DKFILE='"../../src/Sharplight.Shaders/Kernels/attn_decode_native.cu"' \
//   bench_decode.cu -o d && ./d
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef T_ROWS
#define T_ROWS 4
#endif
#ifndef L_KEYS
#define L_KEYS 130048
#endif
#ifndef REPS
#define REPS 20
#endif
#ifndef HQ
#define HQ 64
#endif
#ifndef HKV
#define HKV 8
#endif
#ifndef D
#define D 64
#endif
#ifndef HPW
#define HPW 8
#endif
#ifndef DEC_TARGET_WG
#define DEC_TARGET_WG 1024
#endif
#ifndef DEC_MAX_SPLITS
#define DEC_MAX_SPLITS 128
#endif
#ifndef DEC_BC
#define DEC_BC 64
#endif
#ifndef DEC_MIN_KEYS
#define DEC_MIN_KEYS 256
#endif
#define SCALE 0.125f
#define QS1 D
#define QS0 (HQ * D)
#define HAS_SINKS 1
#define MULTI_SEQ 1
#ifdef PAGED
// Paged cache: PAGE_BS-row blocks, a random permutation of the pool over the sequences (PAGE_SEQ: in order).
#ifndef PAGE_BS
#define PAGE_BS 16
#endif
#define PAGE_MB ((L_KEYS + PAGE_BS - 1) / PAGE_BS)
#define SLOT_CTX 0
#else
#define SLOT_CTX L_KEYS
#endif
#define RT_N_PAST 0
#include KFILE

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

int main()
{
#if PAGED
    const size_t rows = (size_t)T_ROWS * PAGE_MB * PAGE_BS * HKV;
#else
    const size_t rows = (size_t)T_ROWS * L_KEYS * HKV;
#endif
    const size_t cacheElems = rows * D;
    std::vector<unsigned short> hk(cacheElems), hv(cacheElems);
    unsigned s = 12345u;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 9) & 0xFFFF) / 65536.0f - 0.5f; };
    for (size_t i = 0; i < cacheElems; i++)
    {
        __half a = __float2half(rnd()), b = __float2half(rnd());
        hk[i] = *(unsigned short*)&a;
        hv[i] = *(unsigned short*)&b;
    }
    std::vector<float> hq(T_ROWS * HQ * D), hs(HQ);
    for (auto& x : hq) x = rnd() * 4;
    for (auto& x : hs) x = rnd();
    // Cache row of key j of sequence t.
    std::vector<int> pages;
#if PAGED
    pages.resize((size_t)T_ROWS * PAGE_MB);
    for (size_t i = 0; i < pages.size(); i++) pages[i] = (int)i;
#ifndef PAGE_SEQ
    for (size_t i = pages.size() - 1; i > 0; i--) { s = s * 1664525u + 1013904223u; std::swap(pages[i], pages[(s >> 8) % (i + 1)]); }
#endif
    auto cacheRow = [&](int t, int j) { return ((size_t)pages[(size_t)t * PAGE_MB + j / PAGE_BS] * PAGE_BS + j % PAGE_BS) * HKV; };
#else
    auto cacheRow = [&](int t, int j) { return ((size_t)t * L_KEYS + j) * HKV; };
#endif
    (void)cacheRow;
    std::vector<int> hseq(T_ROWS * 4);
    for (int t = 0; t < T_ROWS; t++) { hseq[t * 4] = t; hseq[t * 4 + 1] = L_KEYS - 1; hseq[t * 4 + 2] = 0; hseq[t * 4 + 3] = 0; }

    unsigned short *dk, *dv; float *dq, *ds, *dpart, *dout; int* dseq; unsigned *drt, *darr;
    const int ms = (int)std::min(std::max(DEC_TARGET_WG / std::max(T_ROWS * ND_GROUPS, 1), 1), DEC_MAX_SPLITS);
    CK(cudaMalloc(&dk, cacheElems * 2)); CK(cudaMalloc(&dv, cacheElems * 2));
    CK(cudaMalloc(&dq, hq.size() * 4)); CK(cudaMalloc(&ds, HQ * 4)); CK(cudaMalloc(&dout, hq.size() * 4));
    CK(cudaMalloc(&dpart, (size_t)ms * T_ROWS * HQ * ND_PS * 4)); CK(cudaMalloc(&dseq, hseq.size() * 4));
    CK(cudaMalloc(&drt, 256)); CK(cudaMalloc(&darr, 4096));
    CK(cudaMemcpy(dk, hk.data(), cacheElems * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv, hv.data(), cacheElems * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ds, hs.data(), HQ * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dseq, hseq.data(), hseq.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(drt, 0, 256));

    NdParams p{};
    p.q = dq; p.kc = dk; p.vc = dv; p.o = dout; p.part = dpart; p.sinks = ds; p.rt = drt; p.seq = dseq; p.runs = dseq; p.arrivals = darr;
    p.T = T_ROWS; p.row0 = 0;
#if PAGED
    int* dpages; CK(cudaMalloc(&dpages, pages.size() * 4)); CK(cudaMemcpy(dpages, pages.data(), pages.size() * 4, cudaMemcpyHostToDevice));
    p.pages = dpages;
#endif
    dim3 grid(ND_GROUPS, T_ROWS, ms);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> times;
    for (int r = 0; r < REPS + 2; r++)
    {
        CK(cudaEventRecord(e0));
        attn_decode_native<<<grid, ND_THREADS>>>(p);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        float ms_ = 0; CK(cudaEventElapsedTime(&ms_, e0, e1));
        if (r >= 2) times.push_back(ms_);
    }
    std::sort(times.begin(), times.end());
    double bytes = 2.0 * T_ROWS * L_KEYS * HKV * D * 2;
    printf("T=%d L=%d splits<=%d grid %dx%dx%d: min %.4f ms (%.0f GB/s), median %.4f ms\n", T_ROWS, L_KEYS, ms, ND_GROUPS, T_ROWS, ms, times[0],
           bytes / times[0] / 1e6, times[times.size() / 2]);
#ifdef CHECK
    // Merge the partials of heads 0, 9 and 63 of each row on the host and compare with an f32 reference.
    std::vector<float> part((size_t)ms * T_ROWS * HQ * ND_PS);
    CK(cudaMemcpy(part.data(), dpart, part.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0;
    for (int t = 0; t < T_ROWS; t++) for (int h : {0, 9, 63})
    {
        int g = h / (HQ / HKV);
        std::vector<double> sc(L_KEYS); double mx = -1e30;
        for (int j = 0; j < L_KEYS; j++)
        {
            double a = 0; size_t row = cacheRow(t, j) + g;
            for (int d = 0; d < D; d++) { __half kh = *(__half*)&hk[row * D + d]; a += (double)__half2float(__float2half(hq[(t * HQ + h) * D + d] * SCALE)) * __half2float(kh); }
            sc[j] = a; mx = std::max(mx, a);
        }
        mx = std::max(mx, (double)hs[h]);
        double l = std::exp(hs[h] - mx); std::vector<double> o(D, 0);
        for (int j = 0; j < L_KEYS; j++) { double w = std::exp(sc[j] - mx); l += w; size_t row = cacheRow(t, j) + g;
            for (int d = 0; d < D; d++) { __half vh = *(__half*)&hv[row * D + d]; o[d] += w * __half2float(vh); } }
        // Merge the partials of the ranges the kernel wrote (empty ranges have l = 0).
        double M = -1e30; for (int sp = 0; sp < ms; sp++) { const float* q = &part[(((size_t)sp * T_ROWS + t) * HQ + h) * ND_PS]; if (q[D + 1] > 0) M = std::max(M, (double)q[D]); }
        M = std::max(M, (double)hs[h]);
        double L2 = std::exp(hs[h] - M); std::vector<double> num(D, 0);
        for (int sp = 0; sp < ms; sp++) { const float* q = &part[(((size_t)sp * T_ROWS + t) * HQ + h) * ND_PS]; if (!(q[D + 1] > 0)) continue;
            double w = std::exp(q[D] - M); L2 += w * q[D + 1]; for (int d = 0; d < D; d++) num[d] += w * q[d]; }
        for (int d = 0; d < D; d++) worst = std::max(worst, std::fabs(num[d] / L2 - o[d] / l));
    }
    printf("max error against the f32 reference: %.3g\n", worst);
#endif
    return 0;
}
