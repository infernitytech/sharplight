// Harness of the hand-written K-quant decode GEMM (src/Sharplight.Shaders/Kernels/gemm_kq_native.cu, the file in KFILE): random f16
// activations [T, K] and random Q4_K / Q5_K / Q6_K blocks (KQ_FMT 4 / 5 / 6) for [N, K] weights, checked against an f32 GEMM on the
// blocks dequantized on the host, timed with events: best of ITERS after an L2 flush ("isolated"), and LAYERS weight copies back to back,
// PASSES times ("sustained", µs per launch: the weights stream from memory as in a decode step).
// nvcc -O3 -arch=sm_120 -std=c++17 -DK=4096 -DN=12288 -DKQ_FMT=4 -DKQ_BM=32 -DKQ_STAGES=3 -DKQ_SPLIT=1 \
//   -DKFILE='"../../src/Sharplight.Shaders/Kernels/gemm_kq_native.cu"' bench_kq.cu -o b && ./b 32
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define Y_STRIDE N
#ifndef ITERS
#define ITERS 30
#endif
#ifndef LAYERS
#define LAYERS 16
#endif
#ifndef PASSES
#define PASSES 10
#endif
#ifdef KQ_DUAL
#if KQ_DUAL
#define KQ_EPI2(a, b) ((a) / (1.0f + __expf(-(a))) * (b))
#endif
#endif
#include KFILE

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void reference(const __half* x, const float* w, float* y, int T, int rows, int k)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (n >= rows || t >= T)
    {
        return;
    }
    float s = 0;
    for (int i = 0; i < k; i++)
    {
        s += __half2float(x[(size_t)t * k + i]) * w[(size_t)n * k + i];
    }
    y[(size_t)t * rows + n] = s;
}

static unsigned s_rng = 12345;
static unsigned rnd()
{
    s_rng ^= s_rng << 13;
    s_rng ^= s_rng >> 17;
    s_rng ^= s_rng << 5;
    return s_rng;
}

static float h2f(unsigned short h)
{
    __half v;
    memcpy(&v, &h, 2);
    return __half2float(v);
}

static unsigned short f2h(float f)
{
    __half v = __float2half(f);
    unsigned short h;
    memcpy(&h, &v, 2);
    return h;
}

[[maybe_unused]] static void scale_min(const unsigned char* q, int j, int& sc, int& m)
{
    if (j < 4)
    {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    }
    else
    {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// One random block of KQ_FMT and its 256 dequantized values.
static void block(unsigned char* b, float* out)
{
    for (int i = 0; i < KQ_BB; i++)
    {
        b[i] = (unsigned char)rnd();
    }
#if KQ_FMT == 6
    unsigned short d = f2h(0.0005f + (rnd() % 1000) * 1e-6f);
    memcpy(b + 208, &d, 2);
    float dd = h2f(d);
    for (int i = 0; i < 256; i++)
    {
        int jj = i / 128, l = i % 128, qq = l / 32, ll = l % 32;
        int q = ((b[64 * jj + 32 * (qq & 1) + ll] >> (4 * (qq >> 1))) & 0xF) | (((b[128 + 32 * jj + ll] >> (2 * qq)) & 3) << 4);
        out[i] = dd * (signed char)b[192 + i / 16] * (q - 32);
    }
#else
    unsigned short d = f2h(0.0005f + (rnd() % 1000) * 1e-6f), dm = f2h(0.001f + (rnd() % 1000) * 1e-6f);
    memcpy(b, &d, 2);
    memcpy(b + 2, &dm, 2);
    float dd = h2f(d), dmin = h2f(dm);
    const int qs = KQ_FMT == 5 ? 48 : 16;
    for (int i = 0; i < 256; i++)
    {
        int jj = i / 64, l = i % 64, hi = l >= 32, ll = l % 32;
        int q = (b[qs + 32 * jj + ll] >> (4 * hi)) & 0xF;
#if KQ_FMT == 5
        q |= ((b[16 + ll] >> (2 * jj + hi)) & 1) << 4;
#endif
        int sc, m;
        scale_min(b + 4, 2 * jj + hi, sc, m);
        out[i] = dd * sc * q - dmin * m;
    }
#endif
}

int main(int argc, char** argv)
{
    int T = argc > 1 ? atoi(argv[1]) : 64;
    const size_t rowBytes = (size_t)(K / 256) * KQ_BB, wBytes = rowBytes * N;
    std::vector<unsigned char> hw(wBytes);
    std::vector<float> hd((size_t)N * K);
    for (size_t r = 0; r < (size_t)N; r++)
    {
        for (int s = 0; s < K / 256; s++)
        {
            block(&hw[r * rowBytes + s * KQ_BB], &hd[r * K + s * 256]);
        }
    }
#if KQ_DUAL
    // The second matrix (up) of the dual form: other random blocks.
    std::vector<unsigned char> hw2(wBytes);
    std::vector<float> hd2((size_t)N * K);
    for (size_t r = 0; r < (size_t)N; r++)
    {
        for (int s = 0; s < K / 256; s++)
        {
            block(&hw2[r * rowBytes + s * KQ_BB], &hd2[r * K + s * 256]);
        }
    }
#endif
    std::vector<__half> hx((size_t)64 * K);
    for (auto& v : hx)
    {
        v = __float2half(((rnd() % 20001) / 10000.0f - 1.0f));
    }
    __half* x;
    float *wd, *y, *yr, *part;
    unsigned* arrivals;
    CK(cudaMalloc(&x, hx.size() * 2));
    CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    std::vector<unsigned char*> w(LAYERS);
    for (int l = 0; l < LAYERS; l++)
    {
        CK(cudaMalloc(&w[l], wBytes));
        CK(cudaMemcpy(w[l], hw.data(), wBytes, cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&wd, hd.size() * 4));
    CK(cudaMemcpy(wd, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice));
#if KQ_DUAL
    std::vector<unsigned char*> w2(LAYERS);
    for (int l = 0; l < LAYERS; l++)
    {
        CK(cudaMalloc(&w2[l], wBytes));
        CK(cudaMemcpy(w2[l], hw2.data(), wBytes, cudaMemcpyHostToDevice));
    }
    float *wd2, *yr2;
    CK(cudaMalloc(&wd2, hd2.size() * 4));
    CK(cudaMemcpy(wd2, hd2.data(), hd2.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&yr2, (size_t)64 * N * 4));
#endif
    CK(cudaMalloc(&y, (size_t)64 * N * 4));
    CK(cudaMalloc(&yr, (size_t)64 * N * 4));
    CK(cudaMalloc(&part, (size_t)KQ_SPLIT * 64 * N * 8));
    CK(cudaMalloc(&arrivals, 65536 * 4));
    CK(cudaMemset(arrivals, 0, 65536 * 4));
    auto params = [&](int l) {
        KqParams p = {};
        p.x = (const unsigned char*)x;
        p.w = w[l];
        p.y = y;
        p.part = part;
#if KQ_DUAL
        p.w_1 = w2[l];
#endif
        p.arrivals = arrivals;
        p.T = T;
        return p;
    };
    const int smem = KQ_STAGES * KQ_XBYTES + KQ_WST * KQ_WBYTES;
    CK(cudaFuncSetAttribute(gemm_kq_native, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    dim3 grid((T + KQ_BM - 1) / KQ_BM, (KQ_PN + KQ_BN - 1) / KQ_BN, KQ_SPLIT);
    gemm_kq_native<<<grid, KQ_THREADS, smem>>>(params(0));
    CK(cudaDeviceSynchronize());
    reference<<<dim3((N + 127) / 128, T), 128>>>(x, wd, yr, T, N, K);
    CK(cudaDeviceSynchronize());
    std::vector<float> a((size_t)T * N), r((size_t)T * N);
    CK(cudaMemcpy(a.data(), y, a.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(r.data(), yr, r.size() * 4, cudaMemcpyDeviceToHost));
#if KQ_DUAL
    reference<<<dim3((N + 127) / 128, T), 128>>>(x, wd2, yr2, T, N, K);
    CK(cudaDeviceSynchronize());
    std::vector<float> r2((size_t)T * N);
    CK(cudaMemcpy(r2.data(), yr2, r2.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < r.size(); i++)
    {
        r[i] = r[i] / (1.0f + expf(-r[i])) * r2[i];
    }
#endif
    double worst = 0, scale = 0;
    for (size_t i = 0; i < a.size(); i++)
    {
        scale = fmax(scale, fabs(r[i]));
    }
    for (size_t i = 0; i < a.size(); i++)
    {
        worst = fmax(worst, fabs(a[i] - r[i]) / scale);
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
        gemm_kq_native<<<grid, KQ_THREADS, smem>>>(params(i % LAYERS));
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
            gemm_kq_native<<<grid, KQ_THREADS, smem>>>(params(i % LAYERS));
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        sustained = fminf(sustained, ms / (PASSES * LAYERS));
    }
    CK(cudaGetLastError());
    const double traffic = (KQ_DUAL ? 2.0 : 1.0) * wBytes;
    printf("dual=%d fmt=%d T=%d K=%d N=%d BM=%d stages=%d split=%d blocks=%d smem=%d: isolated %.1f us (%.0f GB/s), sustained %.1f us (%.0f GB/s), max err %.2e of max |y| %.2f\n",
        KQ_DUAL, KQ_FMT, T, K, N, KQ_BM, KQ_STAGES, KQ_SPLIT, grid.x * grid.y * grid.z, smem, best * 1e3, traffic / best / 1e6, sustained * 1e3, traffic / sustained / 1e6,
        worst, scale);
    return 0;
}
