// Harness of the FP8 decode GEMM (src/Sharplight.Shaders/Kernels/gemm_fp8_decode.cu): random E4M3 codes and scales in the layouts
// the kernel reads (x rows, W in 128-row groups), checked against a reference kernel, timed with events (best of ITERS after an L2
// flush, as decode streams its weights from memory) and back to back after a flush per launch (FLUSHED mean).
// nvcc -O3 -arch=sm_120 -std=c++17 -DK=4096 -DN=12288 -DDUAL=1 -DTT=16 [-DF8D_RF=.. -DF8D_KS=.. -DF8D_RT=.. -DF8D_DEPTH=..] bench_dec.cu
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#ifndef ITERS
#define ITERS 30
#endif
#ifndef TT
#define TT 16
#endif
#define Y_STRIDE N
#if DUAL
#define F8D_EPI2(a, b) ((a) / (1.0f + __expf(-(a) * 1e-4f)) * (b))
#endif
#include "../../src/Sharplight.Shaders/Kernels/gemm_fp8_decode.cu"

static unsigned hash(unsigned x) { x *= 2654435761u; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15; return x; }

__global__ void ref(const unsigned char* xc, const float* xs, const unsigned char* wc, const float* ws, const unsigned char* wc2,
                    const float* ws2, float* y, int T)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (n >= N) return;
    float acc = 0, acc2 = 0;
    for (int kb = 0; kb < K / 128; kb++)
    {
        float s = 0, s2 = 0;
        for (int k = kb * 128; k < kb * 128 + 128; k++)
        {
            __nv_fp8_e4m3 a, b, c; a.__x = xc[(size_t)t * K + k]; b.__x = wc[(size_t)n * K + k]; c.__x = wc2[(size_t)n * K + k];
            s += (float)a * (float)b; s2 += (float)a * (float)c;
        }
        acc += s * xs[t * (K / 128) + kb] * ws[n * (K / 128) + kb];
        acc2 += s2 * xs[t * (K / 128) + kb] * ws2[n * (K / 128) + kb];
    }
    y[(size_t)t * N + n] = DUAL ? acc / (1.0f + expf(-acc * 1e-4f)) * acc2 : acc;
}

struct Logical { std::vector<unsigned char> c; std::vector<float> s; };
static Logical make(size_t rows, unsigned seed)
{
    Logical l; l.c.resize(rows * K); l.s.resize(rows * (K / 128));
    for (size_t i = 0; i < l.c.size(); i++) { unsigned x = hash((unsigned)i + seed * 0x9e3779b9u); l.c[i] = (x & 0x80) | (0x28 + (x >> 8) % 0x20); }
    for (size_t i = 0; i < l.s.size(); i++) l.s[i] = 0.5f + hash((unsigned)i ^ (seed << 20)) % 1000 / 1000.0f;
    return l;
}
static std::vector<unsigned char> rowsLayout(const Logical& l, size_t rows)
{
    size_t rs = K + K / 32; std::vector<unsigned char> b(rows * rs, 0);
    for (size_t r = 0; r < rows; r++) { memcpy(&b[r * rs], &l.c[r * K], K); memcpy(&b[r * rs + K], &l.s[r * (K / 128)], K / 32); }
    return b;
}
static std::vector<unsigned char> groupedLayout(const Logical& l, size_t rows)
{
    size_t rs = K + K / 32; std::vector<unsigned char> b(rows * rs);
    for (size_t r = 0; r < rows; r++)
    {
        size_t g = r / 128 * 128 * rs, i = r % 128;
        memcpy(&b[g + i * K], &l.c[r * K], K);
        for (int kb = 0; kb < K / 128; kb++) memcpy(&b[g + 128 * K + kb * 512 + i * 4], &l.s[r * (K / 128) + kb], 4);
    }
    return b;
}
// Evicts L2 by reading a 256 MB buffer (a memset would leave dirty lines that the timed kernel writes back).
__global__ void flushRead(const uint4* p, size_t n, unsigned* out)
{
    unsigned acc = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { uint4 v = __ldcg(p + i); acc ^= v.x; }
    if (acc == 0x12345678) out[0] = acc;
}
// A launch before the GEMM: lets dependents start, then spins for SPIN_NS on a few blocks (the norm / quantize launch of a
// decode step), so the GEMM's pre-wait weight loads can be timed as a programmatic dependent.
__global__ void spin(unsigned long long ns, unsigned* out)
{
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    unsigned long long t = t0;
    while (t - t0 < ns) { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); }
    if (threadIdx.x == 0 && t == 1) out[0] = 1;
}
template <typename T> static T* upload(const std::vector<T>& v) { T* d; cudaMalloc(&d, v.size() * sizeof(T)); cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d; }

int main()
{
    const int M = TT;
    Logical x = make(M, 1), w = make(N, 2), w2 = make(N, 3);
    unsigned char* dx = upload(rowsLayout(x, M));
    unsigned char* dw = upload(groupedLayout(w, N));
    unsigned char* dw2 = upload(groupedLayout(w2, N));
    size_t flushBytes = 256u << 20; void* flush; cudaMalloc(&flush, flushBytes);
    float* y; cudaMalloc(&y, (size_t)M * N * 4);
    F8dParams p; p.x = dx; p.w = dw;
#if DUAL
    p.w2 = dw2;
#endif
    p.y = y; p.T = M;
    int perSm = 0, sms = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, gemm_fp8_decode, F8D_THREADS, 0);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    dim3 grid(F8D_TILES < perSm * sms ? F8D_TILES : perSm * sms);
    printf("  tiles %d, resident %d x %d, grid %d\n", F8D_TILES, perSm, sms, grid.x);
    gemm_fp8_decode<<<grid, F8D_THREADS>>>(p);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("error %s\n", cudaGetErrorString(e)); return 1; }
    {
        unsigned char *rc = upload(x.c), *rw = upload(w.c), *rw2 = upload(w2.c);
        float *rs = upload(x.s), *rws = upload(w.s), *rws2 = upload(w2.s), *yr; cudaMalloc(&yr, (size_t)M * N * 4);
        ref<<<dim3((N + 127) / 128, M), 128>>>(rc, rs, rw, rws, rw2, rws2, yr, M);
        cudaDeviceSynchronize();
        std::vector<float> a((size_t)M * N), b((size_t)M * N);
        cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), yr, b.size() * 4, cudaMemcpyDeviceToHost);
        double maxe = 0, maxv = 0;
        for (size_t i = 0; i < a.size(); i++) { maxe = fmax(maxe, fabs(a[i] - b[i])); maxv = fmax(maxv, fabs(b[i])); }
        printf("  check T=%d: max err %.3g of max %.3g\n", M, maxe, maxv);
    }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e9, sum = 0; int cnt = 0;
    for (int it = 0; it < ITERS; it++)
    {
#ifdef FLUSHMEMSET
        cudaMemsetAsync(flush, it, flushBytes);
#else
        flushRead<<<328, 256>>>((const uint4*)flush, flushBytes / 16, (unsigned*)y);
#endif
        cudaEventRecord(e0); gemm_fp8_decode<<<grid, F8D_THREADS>>>(p); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); if (it > 2) { best = fmin(best, ms); sum += ms; cnt++; }
    }
    double wbytes = (double)N * (K + K / 32) * (DUAL ? 2 : 1);
#ifdef SPIN_NS
    {
        cudaLaunchAttribute at; at.id = cudaLaunchAttributeProgrammaticStreamSerialization; at.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {}; cfg.gridDim = grid; cfg.blockDim = dim3(F8D_THREADS); cfg.attrs = &at; cfg.numAttrs = 1;
        for (int pdl = 0; pdl < 3; pdl++)
        {
            float b2 = 1e9;
            for (int it = 0; it < ITERS; it++)
            {
                flushRead<<<328, 256>>>((const uint4*)flush, flushBytes / 16, (unsigned*)y);
                cudaEventRecord(e0);
                spin<<<16, 256>>>(SPIN_NS, (unsigned*)flush);
                if (pdl == 1) cudaLaunchKernelEx(&cfg, gemm_fp8_decode, p);
                else if (pdl == 0) gemm_fp8_decode<<<grid, F8D_THREADS>>>(p);
                cudaEventRecord(e1); cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1); if (it > 2) b2 = fmin(b2, ms);
            }
            printf("  after a %d ns launch, %s: min %.1f us\n", SPIN_NS, pdl == 1 ? "dependent" : pdl == 0 ? "plain    " : "spin only", b2 * 1e3);
        }
    }
#endif
    printf("T=%3d K=%5d N=%5d dual=%d RF=%d KS=%d RT=%d D=%d grid=%d: min %.1f us mean %.1f us, %.0f GB/s (min)\n", M, K, N, DUAL, F8D_RF, F8D_KS, F8D_RT,
        F8D_DEPTH, grid.x, best * 1e3, sum / cnt * 1e3, wbytes / best / 1e6);
}
