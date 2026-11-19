// nvcc -O3 -arch=sm_120 -std=c++17 bench_fp8.cu -o bench_fp8 [-DF8N_...]
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include KHDR

__global__ void fill(unsigned char* p, size_t rows, size_t K, unsigned seed)
{
    size_t rs = K + K / 32;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < rows * rs; i += (size_t)gridDim.x * blockDim.x)
    {
        size_t r = i / rs, o = i % rs;
        unsigned x = (unsigned)i * 2654435761u + seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        if (o < K)
        {
            p[i] = (x & 0x80) | (0x28 + (x >> 8) % 0x20);
        }
        else if ((o - K) % 4 == 0)
        {
            float s = 0.5f + (x % 1000) / 1000.0f;
            *(float*)(p + i) = s;
        }
    }
}

__device__ float dec(unsigned char c)
{
    __nv_fp8_e4m3 v; v.__x = c; return (float)v;
}

__global__ void ref(const unsigned char* x, const unsigned char* w, const unsigned char* w2, float* y, int T, int N, int K, int dual)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (n >= N) return;
    size_t rs = K + K / 32;
    float acc = 0, acc2 = 0;
    for (int kb = 0; kb < K / 128; kb++)
    {
        float s = 0, s2 = 0;
        for (int k = kb * 128; k < kb * 128 + 128; k++)
        {
            float xv = dec(x[t * rs + k]);
            s += xv * dec(w[n * rs + k]);
            if (dual) s2 += xv * dec(w2[n * rs + k]);
        }
        float xs = *(const float*)(x + t * rs + K + kb * 4);
        acc += s * xs * *(const float*)(w + n * rs + K + kb * 4);
        if (dual) acc2 += s2 * xs * *(const float*)(w2 + n * rs + K + kb * 4);
    }
    y[(size_t)t * N + n] = dual ? acc / (1.0f + expf(-acc)) * acc2 : acc;
}

int main(int argc, char** argv)
{
    int Ms[] = { 512, 2048, 8192 };
    int shapes[][2] = { { 4096, 4096 }, { 4096, 1024 }, { 4096, 12288 }, { 12288, 4096 } };
    const char* names[] = { "q/o", "k/v", F8N_DUAL ? "gate+up" : "gate", "down" };
    cudaFree(0); cudaFuncSetAttribute(gemm_fp8_native, cudaFuncAttributeMaxDynamicSharedMemorySize, F8N_SHARED);
    printf("BM=%d BN=%d stages=%d warps=%dx%d dual=%d shared=%d\n", F8N_BM, F8N_BN, F8N_STAGES, F8N_WM, F8N_WN, F8N_DUAL, F8N_SHARED);
    for (int M : Ms)
    {
        for (int s = 0; s < 4; s++)
        {
            int K = shapes[s][0], N = shapes[s][1];
            size_t rs = K + K / 32;
            unsigned char *x, *w, *w2;
            float *y, *yr;
            cudaMalloc(&x, M * rs); cudaMalloc(&w, N * rs); cudaMalloc(&w2, N * rs);
            cudaMalloc(&y, (size_t)M * N * 4); cudaMalloc(&yr, (size_t)M * N * 4);
            fill<<<1024, 256>>>(x, M, K, 1); fill<<<1024, 256>>>(w, N, K, 2); fill<<<1024, 256>>>(w2, N, K, 3);
            F8nParams p;
            p.x = x; p.w = w;
#if F8N_DUAL
            p.w2 = w2;
#endif
#ifdef F8N_TMA
            auto enc = [](F8nTensorMap* m, const void* base, unsigned long long rows, unsigned long long rowBytes, unsigned K, unsigned box) {
                cuuint64_t dims[2] = { K, rows };
                cuuint64_t strides[1] = { rowBytes };
                cuuint32_t boxDim[2] = { 128, box };
                cuuint32_t es[2] = { 1, 1 };
                CUresult r = cuTensorMapEncodeTiled((CUtensorMap*)m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)base, dims, strides, boxDim, es,
                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
                if (r) printf("encode failed %d\n", (int)r);
            };
            enc(&p.mx, x, M, rs, K, F8N_BM);
            enc(&p.mw, w, N, rs, K, F8N_DUAL ? F8N_BN / 2 : F8N_BN);
#if F8N_DUAL
            enc(&p.mw2, w2, N, rs, K, F8N_BN / 2);
#endif
#endif
            p.y = y; p.T = M; p.N = N; p.K = K; p.ldy = N;
            dim3 grid((M + F8N_BM - 1) / F8N_BM, F8N_DUAL ? (N + F8N_BN / 2 - 1) / (F8N_BN / 2) : (N + F8N_BN - 1) / F8N_BN);
            gemm_fp8_native<<<grid, F8N_THREADS, F8N_SHARED>>>(p);
            cudaError_t e = cudaDeviceSynchronize();
            if (e) { printf("error %s\n", cudaGetErrorString(e)); return 1; }
            if (M == 512)
            {
                ref<<<dim3((N + 127) / 128, M), 128>>>(x, w, w2, yr, M, N, K, F8N_DUAL);
                cudaDeviceSynchronize();
                std::vector<float> a((size_t)M * N), b((size_t)M * N);
                cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(b.data(), yr, b.size() * 4, cudaMemcpyDeviceToHost);
                double maxe = 0, maxv = 0;
                for (size_t i = 0; i < a.size(); i++) { maxe = fmax(maxe, fabs(a[i] - b[i])); maxv = fmax(maxv, fabs(b[i])); }
                printf("  check M=%d %s: max err %.3g of max %.3g\n", M, names[s], maxe, maxv);
            }
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            float best = 1e9;
            for (int it = 0; it < 20; it++)
            {
                cudaEventRecord(e0);
                gemm_fp8_native<<<grid, F8N_THREADS, F8N_SHARED>>>(p);
                cudaEventRecord(e1); cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1); if (it > 2 && ms < best) best = ms;
            }
            double fl = 2.0 * M * N * K * (F8N_DUAL ? 2 : 1);
            printf("M=%5d %-8s K=%5d N=%5d: %.3f ms %.0f TFLOPS\n", M, names[s], K, N, best, fl / best / 1e9);
            cudaFree(x); cudaFree(w); cudaFree(w2); cudaFree(y); cudaFree(yr);
        }
    }
}
