#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include "gemm_f16.cuh"
__global__ void fillh(unsigned short* p, size_t n, unsigned seed)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        unsigned x = (unsigned)i * 2654435761u + seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        __half h = __float2half(((x % 2001) / 1000.0f - 1.0f));
        p[i] = *(unsigned short*)&h;
    }
}
__global__ void ref(const __half* x, const __half* w, float* y, int T, int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (n >= N) return;
    float a = 0;
    for (int k = 0; k < K; k++) a += __half2float(x[(size_t)t * K + k]) * __half2float(w[(size_t)n * K + k]);
    y[(size_t)t * N + n] = a;
}
int main()
{
    int shapes[][2] = { { 4096, 4096 }, { 4096, 1024 }, { 4096, 12288 }, { 12288, 4096 }, { 4096, 6144 }, { 4096, 24576 } };
    const char* names[] = { "q/o", "k/v", "gate", "down", "qkv", "gate_up" };
    cudaFuncSetAttribute(gemm_f16_native, cudaFuncAttributeMaxDynamicSharedMemorySize, H_SHARED);
    printf("BM=%d BN=%d stages=%d warps=%dx%d shared=%d\n", H_BM, H_BN, H_STAGES, H_WM, H_WN, H_SHARED);
    for (int M : { 512, 2048, 8192 })
        for (int s = 0; s < 6; s++)
        {
            int K = shapes[s][0], N = shapes[s][1];
            unsigned short *x, *w; float *y, *yr;
            cudaMalloc(&x, (size_t)M * K * 2); cudaMalloc(&w, (size_t)N * K * 2);
            cudaMalloc(&y, (size_t)M * N * 4); cudaMalloc(&yr, (size_t)M * N * 4);
            fillh<<<1024, 256>>>(x, (size_t)M * K, 1); fillh<<<1024, 256>>>(w, (size_t)N * K, 2);
            HParams p{ x, w, y, (unsigned)M, (unsigned)N, (unsigned)K, (unsigned)N };
            dim3 grid((M + H_BM - 1) / H_BM, (N + H_BN - 1) / H_BN);
            gemm_f16_native<<<grid, H_THREADS, H_SHARED>>>(p);
            if (cudaDeviceSynchronize()) { printf("launch error\n"); return 1; }
            if (M == 512 && s < 4)
            {
                ref<<<dim3((N + 127) / 128, M), 128>>>((__half*)x, (__half*)w, yr, M, N, K);
                cudaDeviceSynchronize();
                std::vector<float> a((size_t)M * N), b((size_t)M * N);
                cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), yr, b.size() * 4, cudaMemcpyDeviceToHost);
                double e = 0; for (size_t i = 0; i < a.size(); i++) e = fmax(e, fabs(a[i] - b[i]));
                printf("  check %s max err %.3g\n", names[s], e);
            }
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            float best = 1e9;
            for (int it = 0; it < 20; it++)
            {
                cudaEventRecord(e0); gemm_f16_native<<<grid, H_THREADS, H_SHARED>>>(p); cudaEventRecord(e1); cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1); if (it > 2 && ms < best) best = ms;
            }
            printf("M=%5d %-8s K=%5d N=%5d: %.3f ms %.0f TFLOPS\n", M, names[s], K, N, best, 2.0 * M * N * K / best / 1e9);
            cudaFree(x); cudaFree(w); cudaFree(y); cudaFree(yr);
        }
}
