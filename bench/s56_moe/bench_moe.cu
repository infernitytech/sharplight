// Harness of the hand-written grouped MXFP4 expert GEMM (src/Sharplight.Shaders/Kernels/gemm_moe_native.cu, or the file in KFILE):
// E experts of MXFP4R weights (WeightRepack.Mxfp4Split: 16-row groups of [16 rows x K / 2 code bytes][K / 32 blocks x 16 E8M0
// scales]), T tokens routed to S experts each (uniform random, or -DSKEW=a: share (1 + a i)^-1.5 for expert i, e.g. 0.15 or 0.6), the routing tiles and the
// expert-sorted f16 rows as moe_route / moe_gather_x16 build them. Checks against a reference kernel (same decode, f32 sums) and
// times the launch with events (best of 20, L2 evicted before each launch). LOOP=n: also n launches back to back at T = 2048 (the
// power-limited steady state, section 64; sample the SM clock with nvidia-smi meanwhile).
// nvcc -O3 -arch=sm_120 -std=c++17 -DK=2880 -DN=2880 -DNFULL=5760 -DDUAL=1 -DKFILE='"gemm_moe_native.cu"' bench_moe.cu -o b
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

#ifndef E
#define E 32
#endif
#ifndef S
#define S 4
#endif
#ifndef NFULL
#define NFULL N
#endif
#ifndef T_MAX
#define T_MAX 2048
#endif
#define ROW_BYTES (17ull * K / 32)
#define EXPERT_BYTES (ROW_BYTES * NFULL)
#define MAXITEMS (T_MAX * S)
#define MAXTILES (MAXITEMS / 16 + E)
#if DUAL
#define NM_EPI2(a, b, e, n) ((a) / (1.0f + __expf(-(a))) * (b))
#endif
#include KFILE

static unsigned hash(unsigned x)
{
    x *= 2654435761u; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15; return x;
}

// Decoded weight (row n of expert e, value k) exactly as the kernel decodes it.
__device__ float wval(const unsigned char* w, int e, int n, int k)
{
    const unsigned char* g = w + (size_t)e * EXPERT_BYTES + (size_t)(n >> 4) * 16 * ROW_BYTES;
    unsigned byte = g[(size_t)(n & 15) * (K / 2) + (k / 32) * 16 + (k % 16)];
    unsigned code = (k % 32) < 16 ? byte & 15 : byte >> 4;
    unsigned es = g[16 * (size_t)(K / 2) + (k / 32) * 16 + (n & 15)];
    unsigned s1, s2;
    nm_scale(es, s1, s2);
    unsigned h = nm_fp4_h2(code, s1, s2);
    return __half2float(__ushort_as_half((unsigned short)(h & 0xFFFF)));
}

__global__ void ref(const __half* xs, const unsigned char* w, const unsigned char* w2, const unsigned* scratch, float* y, int items)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x, o = blockIdx.y;
    if (n >= N || o >= items) return;
    // Sorted position o: its expert from the tiles.
    unsigned tiles = scratch[0], e = 0;
    for (unsigned t = 0; t < tiles; t++)
    {
        if (o >= (int)scratch[2 + 3 * t] && o < (int)(scratch[2 + 3 * t] + scratch[3 + 3 * t])) e = scratch[1 + 3 * t];
    }
    float a = 0, b = 0;
    for (int k = 0; k < K; k++)
    {
        float xv = __half2float(xs[(size_t)o * K + k]);
        a += xv * wval(w, e, n, k);
#if DUAL
        b += xv * wval(w2, e, n, k);
#endif
    }
    unsigned item = scratch[1 + 3 * MAXTILES + o];
#if DUAL
    y[(size_t)item * N + n] = a / (1.0f + expf(-a)) * b;
#else
    y[(size_t)item * N + n] = a;
#endif
}

template <typename T>
static T* upload(const std::vector<T>& v)
{
    T* d;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

int main(int argc, char** argv)
{
    size_t smemBytes = (size_t)NM_STAGES * NM_STAGE_BYTES;
    if (cudaFuncSetAttribute(gemm_moe_native, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes)) { printf("smem %zu too large\n", smemBytes); return 1; }
    std::vector<unsigned char> wh((size_t)E * EXPERT_BYTES);
    for (size_t i = 0; i < wh.size(); i++) wh[i] = (unsigned char)hash((unsigned)i);
    // Scales: E8M0 120..127 (2^-7 .. 1) in the scale part of every 16-row group.
    for (size_t e = 0; e < E; e++)
        for (size_t g = 0; g < NFULL / 16; g++)
            for (size_t i = 0; i < (K / 32) * 16; i++)
                wh[e * EXPERT_BYTES + g * 16 * ROW_BYTES + 16 * (K / 2) + i] = (unsigned char)(120 + hash((unsigned)(e * 7919 + g * 131 + i)) % 8);
    unsigned char* dw = upload(wh);
    const unsigned char* dw2 = dw + (size_t)N * ROW_BYTES;   // DUAL: the upper half of each expert's rows
    std::vector<__half> xh((size_t)MAXITEMS * K);
    for (size_t i = 0; i < xh.size(); i++) xh[i] = __float2half((int)(hash((unsigned)i * 3 + 1) % 2001 - 1000) / 1000.0f);
    // Range words after the rows (all 0: unscaled), as the gather writes them.
    std::vector<unsigned char> xb(xh.size() * 2 + MAXITEMS * 4, 0);
    memcpy(xb.data(), xh.data(), xh.size() * 2);
    unsigned char* dx = upload(xb);
    float* y; cudaMalloc(&y, (size_t)MAXITEMS * N * 4);
    float* yr; cudaMalloc(&yr, (size_t)MAXITEMS * N * 4);
    size_t flushBytes = 256u << 20;
    void* flush; cudaMalloc(&flush, flushBytes);
    unsigned* dscratch; cudaMalloc(&dscratch, (1 + 3 * MAXTILES + MAXITEMS) * 4);

    int Ts[] = { 32, 64, 128, 171, 256, 512, 1024, 2048 };
    for (int T : Ts)
    {
        if (T > T_MAX) continue;
        // Routing: S distinct experts per token.
        std::mt19937 rng(T);
        std::vector<std::vector<unsigned>> per(E);
        for (int t = 0; t < T; t++)
        {
            std::vector<int> ex(E);
            for (int i = 0; i < E; i++) ex[i] = i;
#ifdef SKEW
            std::vector<double> wt(E); for (int i = 0; i < E; i++) wt[i] = 1.0 / pow(1 + i * SKEW, 1.5);
            for (int s = 0; s < S; s++) { std::discrete_distribution<int> d(wt.begin(), wt.end()); int i = d(rng); per[i].push_back(t * S + s); wt[i] = 0; }
#else
            std::shuffle(ex.begin(), ex.end(), rng);
            for (int s = 0; s < S; s++) per[ex[s]].push_back(t * S + s);
#endif
        }
        std::vector<unsigned> sc(1 + 3 * MAXTILES + MAXITEMS, 0);
        unsigned tiles = 0, o = 0;
        for (int e = 0; e < E; e++)
        {
            for (size_t i = 0; i < per[e].size(); i += NM_BM)
            {
                unsigned rows = (unsigned)std::min<size_t>(NM_BM, per[e].size() - i);
                sc[1 + 3 * tiles] = e; sc[2 + 3 * tiles] = o + (unsigned)i; sc[3 + 3 * tiles] = rows; tiles++;
            }
            for (unsigned it : per[e]) sc[1 + 3 * MAXTILES + o++] = it;
        }
        sc[0] = tiles;
        cudaMemcpy(dscratch, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice);
        int items = T * S;
        NmParams p;
        p.x = dx; p.w = dw;
#if DUAL
        p.w2 = dw2;
#endif
        p.scratch = dscratch; p.y = y; p.T = T;
        dim3 grid((N + NM_BOUT - 1) / NM_BOUT, tiles);
        cudaMemset(y, 0, (size_t)items * N * 4);
        gemm_moe_native<<<grid, NM_THREADS, smemBytes>>>(p);
        cudaError_t err = cudaDeviceSynchronize();
        if (err) { printf("error %s\n", cudaGetErrorString(err)); return 1; }
        if (argc < 2)
        {
            ref<<<dim3((N + 127) / 128, items), 128>>>((const __half*)dx, dw, dw2, dscratch, yr, items);
            cudaDeviceSynchronize();
            std::vector<float> a((size_t)items * N), b((size_t)items * N);
            cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), yr, b.size() * 4, cudaMemcpyDeviceToHost);
            double maxe = 0, maxv = 0;
            for (size_t i = 0; i < a.size(); i++) { maxe = fmax(maxe, fabs(a[i] - b[i])); maxv = fmax(maxv, fabs(b[i])); }
            printf("  check T=%d: max err %.3g of max %.3g\n", T, maxe, maxv);
        }
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        if (getenv("LOOP") && T == 2048)
        {
            int n = atoi(getenv("LOOP"));
            cudaEventRecord(e0);
            for (int it = 0; it < n; it++) gemm_moe_native<<<grid, NM_THREADS, smemBytes>>>(p);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            printf("loop %d: %.3f ms per launch\n", n, ms / n);
        }
        float best = 1e9;
        for (int it = 0; it < 20; it++)
        {
            cudaMemsetAsync(flush, it, flushBytes);
            cudaEventRecord(e0);
            gemm_moe_native<<<grid, NM_THREADS, smemBytes>>>(p);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); if (it > 2 && ms < best) best = ms;
        }
        double fl = 2.0 * items * N * K * (DUAL ? 2 : 1);
        printf("T=%5d items=%6d tiles=%4u K=%d N=%d dual=%d BM=%d BN=%d: %.3f ms %.1f TFLOPS\n", T, items, tiles, K, N, DUAL, NM_BM, NM_BN, best, fl / best / 1e9);
    }
}
