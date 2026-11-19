// Harness of the hand-written FP8 GEMM (src/Sharplight.Shaders/Kernels/gemm_fp8_native.cu, or the file in KFILE): random E4M3
// codes and scales, packed in the layouts the kernel reads (OLD: per-row scales for x and W; otherwise x rows plus the dense
// scale copy, W in 128-row groups), checked against a reference kernel at T = 512, timed with events (best of 20).
// nvcc -O3 -arch=sm_120 -std=c++17 -DK=4096 -DN=12288 -DDUAL=1 -DT_MAX=8192 [-DOLD] -DKFILE='"..."' bench_fp8.cu -o b
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifndef T_MAX
#define T_MAX 8192
#endif
#define XCAP T_MAX
#define Y_STRIDE N
#if DUAL
#define F8N_EPI2(a, b) ((a) / (1.0f + __expf(-(a))) * (b))
#endif
#include KFILE

static unsigned hash(unsigned x)
{
    x *= 2654435761u; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15; return x;
}

static float dec(unsigned char c)
{
    __nv_fp8_e4m3 v; v.__x = c; return (float)v;
}

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
            s += (float)a * (float)b;
            s2 += (float)a * (float)c;
        }
        acc += s * xs[t * (K / 128) + kb] * ws[n * (K / 128) + kb];
        acc2 += s2 * xs[t * (K / 128) + kb] * ws2[n * (K / 128) + kb];
    }
    y[(size_t)t * N + n] = DUAL ? acc / (1.0f + expf(-acc)) * acc2 : acc;
}

struct Logical
{
    std::vector<unsigned char> c;
    std::vector<float> s;
};

static Logical make(size_t rows, unsigned seed)
{
    Logical l;
    l.c.resize(rows * K);
    l.s.resize(rows * (K / 128));
    for (size_t i = 0; i < l.c.size(); i++)
    {
        unsigned x = hash((unsigned)i + seed * 0x9e3779b9u);
        l.c[i] = (x & 0x80) | (0x28 + (x >> 8) % 0x20);
    }
    for (size_t i = 0; i < l.s.size(); i++)
    {
        l.s[i] = 0.5f + hash((unsigned)i ^ (seed << 20)) % 1000 / 1000.0f;
    }
    return l;
}

// Rows [K codes][K / 128 scales].
static std::vector<unsigned char> rowsLayout(const Logical& l, size_t rows, size_t cap)
{
    size_t rs = K + K / 32;
    std::vector<unsigned char> b(cap * rs + (size_t)(K / 128) * cap * 4, 0);
    for (size_t r = 0; r < rows; r++)
    {
        memcpy(&b[r * rs], &l.c[r * K], K);
        memcpy(&b[r * rs + K], &l.s[r * (K / 128)], K / 32);
        for (int kb = 0; kb < K / 128; kb++)
        {
            memcpy(&b[cap * rs + ((size_t)kb * cap + r) * 4], &l.s[r * (K / 128) + kb], 4);
        }
    }
    return b;
}

// 128-row groups [128 rows of K codes][K / 128 blocks of 128 scales].
static std::vector<unsigned char> groupedLayout(const Logical& l, size_t rows)
{
    size_t rs = K + K / 32;
    std::vector<unsigned char> b(rows * rs);
    for (size_t r = 0; r < rows; r++)
    {
        size_t g = r / 128 * 128 * rs, i = r % 128;
        memcpy(&b[g + i * K], &l.c[r * K], K);
        for (int kb = 0; kb < K / 128; kb++)
        {
            memcpy(&b[g + 128 * K + kb * 512 + i * 4], &l.s[r * (K / 128) + kb], 4);
        }
    }
    return b;
}

template <typename T>
static T* upload(const std::vector<T>& v)
{
    T* d;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

int main()
{
    cudaFuncSetAttribute(gemm_fp8_native, cudaFuncAttributeMaxDynamicSharedMemorySize, F8N_STAGES * F8N_STAGE_BYTES);
    Logical x = make(T_MAX, 1), w = make(N, 2), w2 = make(N, 3);
    unsigned char* dx = upload(rowsLayout(x, T_MAX, XCAP));
#ifdef OLD
    unsigned char* dw = upload(rowsLayout(w, N, N));
    unsigned char* dw2 = upload(rowsLayout(w2, N, N));
#else
    unsigned char* dw = upload(groupedLayout(w, N));
    unsigned char* dw2 = upload(groupedLayout(w2, N));
#endif
    float* y; cudaMalloc(&y, (size_t)T_MAX * N * 4);
    int Ms[] = { 256, 512, 1024, 2048, 4096, 8192 };
    for (int M : Ms)
    {
        if (M > T_MAX) continue;
        F8nParams p;
        p.x = dx; p.w = dw;
#if DUAL
        p.w2 = dw2;
#endif
        p.y = y; p.T = M;
        dim3 grid((M + F8N_BM - 1) / F8N_BM, (N + F8N_BOUT - 1) / F8N_BOUT);
        gemm_fp8_native<<<grid, F8N_THREADS, F8N_STAGES * F8N_STAGE_BYTES>>>(p);
        cudaError_t e = cudaDeviceSynchronize();
        if (e) { printf("error %s\n", cudaGetErrorString(e)); return 1; }
        if (M == 512)
        {
            unsigned char *rc = upload(std::vector<unsigned char>(x.c.begin(), x.c.begin() + (size_t)M * K)), *rw = upload(w.c), *rw2 = upload(w2.c);
            float *rs = upload(x.s), *rws = upload(w.s), *rws2 = upload(w2.s), *yr;
            cudaMalloc(&yr, (size_t)M * N * 4);
            ref<<<dim3((N + 127) / 128, M), 128>>>(rc, rs, rw, rws, rw2, rws2, yr, M);
            cudaDeviceSynchronize();
            std::vector<float> a((size_t)M * N), b((size_t)M * N);
            cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), yr, b.size() * 4, cudaMemcpyDeviceToHost);
            double maxe = 0, maxv = 0;
            for (size_t i = 0; i < a.size(); i++) { maxe = fmax(maxe, fabs(a[i] - b[i])); maxv = fmax(maxv, fabs(b[i])); }
            printf("  check T=%d: max err %.3g of max %.3g\n", M, maxe, maxv);
            cudaFree(rc); cudaFree(rw); cudaFree(rw2); cudaFree(rs); cudaFree(rws); cudaFree(rws2); cudaFree(yr);
        }
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e9;
        for (int it = 0; it < 20; it++)
        {
            cudaEventRecord(e0);
            gemm_fp8_native<<<grid, F8N_THREADS, F8N_STAGES * F8N_STAGE_BYTES>>>(p);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); if (it > 2 && ms < best) best = ms;
        }
        double fl = 2.0 * M * N * K * (DUAL ? 2 : 1);
        printf("T=%5d K=%5d N=%5d dual=%d: %.3f ms %.0f TFLOPS\n", M, K, N, DUAL, best, fl / best / 1e9);
    }
}
