// Section 49 of docs/benchmarks.md: cuBLASLt throughput on the Qwen3-8B prefill shapes (bf16 with f32 output, E4M3 per-tensor and
// 128x128 block scales). Build: nvcc -O2 -arch=sm_120 s49_cublaslt_probe.cu -lcublasLt -o probe
// cuBLASLt probe: heuristics + timing for bf16 (f32 out) and E4M3 (per-tensor, VEC128 x BLK128x128 scales) on Qwen3-8B shapes.
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#define CK(x) do { auto r = (x); if (r) { printf("%s failed %d line %d\n", #x, (int)r, __LINE__); return -1; } } while (0)
__global__ void fill(unsigned char* p, size_t n, unsigned seed, int es)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        unsigned x = (unsigned)i * 2654435761u + seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        // bf16: random sign / mantissa with exponent near 1; e4m3: values in (-2, 2) without NaN
        if (es == 2) { if (i % 2 == 0) p[i] = x & 0xFF; else p[i] = 0x3F ^ (x & 0x80); }
        else p[i] = (x & 0x80) | (0x30 + (x >> 8) % 0x10);
    }
}
static cublasLtHandle_t h; static void* ws; static size_t wsSize = 64 << 20;
// D[N x M col-major] = W^T (W row-major [N,K] -> col-major K x N, op T) * X (row-major [M,K] -> col-major K x M)
float run(cudaDataType_t ab, cudaDataType_t dt, int M, int N, int K, int scaleMode, bool timeIt, int* ok)
{
    *ok = 0;
    size_t es = ab == CUDA_R_16BF ? 2 : 1;
    void *A, *B, *D, *sa = nullptr, *sb = nullptr;
    cudaMalloc(&A, (size_t)N * K * es); cudaMalloc(&B, (size_t)M * K * es); cudaMalloc(&D, (size_t)M * N * 4);
    fill<<<1024, 256>>>((unsigned char*)A, (size_t)N * K * es, 1, (int)es); fill<<<1024, 256>>>((unsigned char*)B, (size_t)M * K * es, 2, (int)es);
    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, ld;
    cublasComputeType_t ct = CUBLAS_COMPUTE_32F;
    if (cublasLtMatmulDescCreate(&op, ct, CUDA_R_32F)) return -1;
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
    if (ab != CUDA_R_16BF)
    {
        // A = weights (N x K, blocks 128x128 -> scale [N/128][K/128]); B = activations (1x128 vectors -> [M][K/128])
        int ma = scaleMode, mb = scaleMode == CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F : scaleMode;
        cudaMalloc(&sa, (size_t)N * K / 16 * 4 + 1024); cudaMalloc(&sb, (size_t)M * K / 16 * 4 + 1024);
        { std::vector<float> one((size_t)N * K / 16 + 256, 1.0f); cudaMemcpy(sa, one.data(), one.size() * 4, cudaMemcpyHostToDevice); }
        { std::vector<float> one((size_t)M * K / 16 + 256, 1.0f); cudaMemcpy(sb, one.data(), one.size() * 4, cudaMemcpyHostToDevice); }
        
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &ma, sizeof(ma));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mb, sizeof(mb));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb));
    }
    cublasLtMatrixLayoutCreate(&la, ab, K, N, K);
    cublasLtMatrixLayoutCreate(&lb, ab, K, M, K);
    cublasLtMatrixLayoutCreate(&ld, dt, N, M, N);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize));
    cublasLtMatmulHeuristicResult_t res[4]; int n = 0;
    auto st = cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, ld, ld, pref, 4, res, &n);
    if (st || n == 0) { printf(" [no algo: %d]", (int)st); return -1; }
    float alpha = 1, beta = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e9;
    for (int it = 0; it < (timeIt ? 25 : 1); it++)
    {
        cudaEventRecord(e0);
        st = cublasLtMatmul(h, op, &alpha, A, la, B, lb, &beta, D, ld, D, ld, &res[0].algo, ws, wsSize, 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        if (st) { printf("  matmul status %d\n", (int)st); return -1; }
        float ms; cudaEventElapsedTime(&ms, e0, e1); if (it > 4 && ms < best) best = ms;
    }
    *ok = 1;
    cudaFree(A); cudaFree(B); cudaFree(D); if (sa) cudaFree(sa); if (sb) cudaFree(sb);
    return best;
}
int main()
{
    CK(cublasLtCreate(&h)); cudaMalloc(&ws, wsSize);
    printf("cublasLt version %zu\n", cublasLtGetVersion());
    int shapes[][2] = { {4096, 6144}, {4096, 4096}, {4096, 24576}, {12288, 4096} };
    const char* names[] = { "qkv", "o", "gate_up", "down" };
    for (int M : {512, 2048, 8192})
        for (int s = 0; s < 4; s++)
        {
            int K = shapes[s][0], N = shapes[s][1], ok; double fl = 2.0 * M * N * K;
            printf("M=%d %s K=%d N=%d:", M, names[s], K, N);
            float t = run(CUDA_R_16BF, CUDA_R_32F, M, N, K, 0, true, &ok); if (ok) printf(" bf16 %.0f", fl / t / 1e9);
            t = run(CUDA_R_8F_E4M3, CUDA_R_32F, M, N, K, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, true, &ok); if (ok) printf(" | fp8 tensor %.0f", fl / t / 1e9);
            t = run(CUDA_R_8F_E4M3, CUDA_R_32F, M, N, K, CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F, true, &ok); if (ok) printf(" | fp8 blk128 %.0f", fl / t / 1e9);
            t = run(CUDA_R_8F_E4M3, CUDA_R_16BF, M, N, K, CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F, true, &ok); if (ok) printf(" | fp8 blk128->bf16 %.0f", fl / t / 1e9);
            printf("\n");
        }
}
