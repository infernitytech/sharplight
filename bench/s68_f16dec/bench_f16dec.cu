// Harness of the decode form of the hand-written f16 GEMM (src/Sharplight.Shaders/Kernels/gemm_f16_native.cu, H_CFG 2, the file in
// KFILE): random f16 activations [T, K] and weights (MULTI: two or three matrices of the widths W0 / W1 / W2), with bias rows (H_BIAS),
// checked against a reference kernel, timed with events: best of ITERS after an L2 flush ("isolated"), and LAYERS distinct weight
// copies back to back, PASSES times ("sustained", ms per launch: the weights stream from memory as in a decode step).
// nvcc -O3 -arch=sm_120 -std=c++17 -DK=2880 -DW0=4096 -DW1=512 -DW2=512 -DMULTI=3 -DH_BIAS=1 -DH_CFG=2 \
//   -DKFILE='"../../src/Sharplight.Shaders/Kernels/gemm_f16_native.cu"' bench_f16dec.cu -o b && ./b 64
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifndef MULTI
#define MULTI 0
#endif
#ifndef W1
#define W1 0
#endif
#ifndef W2
#define W2 0
#endif
#define N (W0 + W1 + W2)
#define Y_STRIDE W0
#if MULTI
#define MULTI_O1 W0
#define MULTI_O2 (W0 + W1)
#endif
#ifndef ITERS
#define ITERS 30
#endif
#ifndef LAYERS
#define LAYERS 24
#endif
#ifndef PASSES
#define PASSES 20
#endif
#ifndef H_DBN
#define H_DBN 64
#endif
#include KFILE

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void reference(const __half* x, const __half* w, const float* bias, float* y, int T, int rows, int k)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (n >= rows || t >= T)
    {
        return;
    }
    float s = 0;
    for (int i = 0; i < k; i++)
    {
        s += __half2float(x[(size_t)t * k + i]) * __half2float(w[(size_t)n * k + i]);
    }
    y[(size_t)t * rows + n] = s + (bias ? bias[n] : 0.0f);
}

__global__ void fill(__half* p, size_t n, unsigned seed, float scale)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        unsigned h = (unsigned)i * 2654435761u ^ seed;
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        p[i] = __float2half(((h & 0xffff) / 65535.0f - 0.5f) * scale);
    }
}

int main(int argc, char** argv)
{
    int T = argc > 1 ? atoi(argv[1]) : 64;
    const int widths[3] = { W0, W1, W2 };
    const int members = MULTI ? MULTI : 1;
    __half* x;
    CK(cudaMalloc(&x, (size_t)64 * K * 2));
    fill<<<256, 256>>>(x, (size_t)64 * K, 1, 2.0f);
    std::vector<__half*> w(LAYERS * 3);
    float* bias[3] = {};
    float* y[3] = {};
    float* yr[3] = {};
    for (int l = 0; l < LAYERS; l++)
    {
        for (int m = 0; m < members; m++)
        {
            CK(cudaMalloc(&w[l * 3 + m], (size_t)widths[m] * K * 2));
            fill<<<256, 256>>>(w[l * 3 + m], (size_t)widths[m] * K, 100 + m, 0.1f);
        }
    }
    for (int m = 0; m < members; m++)
    {
        CK(cudaMalloc(&y[m], (size_t)64 * widths[m] * 4));
        CK(cudaMalloc(&yr[m], (size_t)64 * widths[m] * 4));
        std::vector<float> b(widths[m]);
        for (int i = 0; i < widths[m]; i++)
        {
            b[i] = 0.01f * (i % 17);
        }
        CK(cudaMalloc(&bias[m], widths[m] * 4));
        CK(cudaMemcpy(bias[m], b.data(), widths[m] * 4, cudaMemcpyHostToDevice));
    }
    auto params = [&](int l) {
        HParams p = {};
        p.x = (const unsigned char*)x;
        p.w = (const unsigned char*)w[l * 3];
        p.y = y[0];
#if H_BIAS
        p.bias = bias[0];
#endif
#if MULTI
        p.w_1 = (const unsigned char*)w[l * 3 + 1];
        p.y_1 = y[1];
#if MULTI > 2
        p.w_2 = (const unsigned char*)w[l * 3 + 2];
        p.y_2 = y[2];
#endif
#if H_BIAS
        p.bias_1 = bias[1];
#if MULTI > 2
        p.bias_2 = bias[2];
#endif
#endif
#endif
        p.T = T;
        return p;
    };
    const int smem = H_STAGES * H_STAGE_BYTES;
    CK(cudaFuncSetAttribute(gemm_f16_native, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    dim3 grid((T + H_BM - 1) / H_BM, (N + H_BOUT - 1) / H_BOUT);
    gemm_f16_native<<<grid, H_THREADS, smem>>>(params(0));
    CK(cudaDeviceSynchronize());
    double worst = 0;
    for (int m = 0; m < members; m++)
    {
        reference<<<dim3((widths[m] + 127) / 128, T), 128>>>(x, w[m], H_BIAS ? bias[m] : nullptr, yr[m], T, widths[m], K);
        CK(cudaDeviceSynchronize());
        std::vector<float> a((size_t)T * widths[m]), r((size_t)T * widths[m]);
        CK(cudaMemcpy(a.data(), y[m], a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(r.data(), yr[m], r.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < a.size(); i++)
        {
            worst = fmax(worst, fabs(a[i] - r[i]) / (fabs(r[i]) + 1e-1));
        }
    }
    char* flush;
    const size_t flushBytes = 256ull << 20;
    CK(cudaMalloc(&flush, flushBytes));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best = 1e9f;
    for (int i = 0; i < ITERS; i++)
    {
        cudaMemsetAsync(flush, i, flushBytes);
        cudaEventRecord(e0);
        gemm_f16_native<<<grid, H_THREADS, smem>>>(params(i % LAYERS));
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        best = fminf(best, ms);
    }
    float sustained = 1e9f;
    for (int pass = 0; pass < 3; pass++)
    {
        cudaEventRecord(e0);
        for (int i = 0; i < PASSES * LAYERS; i++)
        {
            gemm_f16_native<<<grid, H_THREADS, smem>>>(params(i % LAYERS));
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        sustained = fminf(sustained, ms / (PASSES * LAYERS));
    }
    CK(cudaGetLastError());
    double bytes = (double)N * K * 2;
    printf("T=%d K=%d N=%d (%d+%d+%d) BN=%d blocks=%d: isolated %.1f us (%.0f GB/s), sustained %.1f us (%.0f GB/s), max rel err %.2e\n", T, K, N,
        W0, W1, W2, H_BN, grid.x * grid.y, best * 1e3, bytes / best / 1e6, sustained * 1e3, bytes / sustained / 1e6, worst);
    return 0;
}
