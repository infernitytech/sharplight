// Copy of bench/s57_gemm/bench_fp8.cu with POW2 (power-of-two scales, for gemm_fp8_bs.cu). Harness of the hand-written FP8 GEMMs (src/Sharplight.Shaders/Kernels/gemm_fp8_native.cu or gemm_fp8_tma.cu, the file in KFILE):
// random E4M3 codes and scales in the layouts the kernels read (x rows plus the dense scale copy, W in 128-row groups; OLD: per-row
// scales of the phase-3 kernel), checked against a reference kernel at CHECK_T rows, timed with events (best of ITERS after an L2
// flush) and, with SUSTAIN=ms, back to back for that long: the steady state under the board's power limit, with the board energy
// (NVML) per flop. Options:
//   KFILE, KENTRY (entry point other than gemm_fp8_native), TMA (encode the tensor maps of gemm_fp8_tma.cu), PERSIST=n (grid of
//   at most n blocks for persistent kernels), F8N_CYCLES (kernel adds clock64() cycles per block to g_f8n_cycles: FLOPs per SM
//   cycle, which does not depend on the clock), FULLRANGE (codes over the whole E4M3 range), F8N_EPI2_SLOWPATH (the SiLU
//   epilogue on the raw products: with these magnitudes its division takes the slow path; default: scaled so that it does not),
//   NOFLUSH, PTXFILE / PTXENTRY (launch PTX from a file, e.g. what the provider's NVRTC produced), MS_A / MS_B / MS_MID (row counts).
// nvcc -O3 -arch=sm_120 -std=c++17 -DK=4096 -DN=12288 -DDUAL=1 -DKFILE='"..."' bench_fp8.cu -o b -lcuda -lnvidia-ml
#pragma push_macro("K")
#pragma push_macro("N")
#undef K
#undef N
#include <cuda.h>
#pragma pop_macro("K")
#pragma pop_macro("N")
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <nvml.h>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifndef CHECK_T
#define CHECK_T 512
#endif
#ifndef T_MAX
#define T_MAX 8192
#endif
#define XCAP T_MAX
#ifndef ITERS
#define ITERS 20
#endif
#ifndef SPLITK
#define SPLITK 1
#endif
#define Y_STRIDE N
#if DUAL
#ifdef F8N_EPI2_SLOWPATH
#define F8N_EPI2(a, b) ((a) / (1.0f + __expf(-(a))) * (b))
#else
#define F8N_EPI2(a, b) ((a) / (1.0f + __expf(-(a) * 1e-4f)) * (b))
#endif
#endif
#ifdef F8N_CYCLES
__device__ unsigned long long g_f8n_cycles;
#endif
#include KFILE
#ifndef QOUT
#define QOUT 0
#endif
#ifdef KENTRY
#define gemm_fp8_native KENTRY
#endif

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
#ifndef F8N_EPI2_SLOWPATH
    y[(size_t)t * N + n] = DUAL ? acc / (1.0f + expf(-acc * 1e-4f)) * acc2 : acc;
#else
    y[(size_t)t * N + n] = DUAL ? acc / (1.0f + expf(-acc)) * acc2 : acc;
#endif
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
#ifdef FULLRANGE
        unsigned char c = (unsigned char)(x >> 8);
        l.c[i] = (c & 0x7f) == 0x7f ? c - 1 : c;   // any E4M3 code but NaN
#else
        l.c[i] = (x & 0x80) | (0x28 + (x >> 8) % 0x20);
#endif
    }
    for (size_t i = 0; i < l.s.size(); i++)
    {
#ifdef POW2
        l.s[i] = ldexpf(1.0f, -(int)(hash((unsigned)i ^ (seed << 20)) % 4));   // UE8M0 scales (gemm_fp8_bs.cu)
#else
        l.s[i] = 0.5f + hash((unsigned)i ^ (seed << 20)) % 1000 / 1000.0f;
#endif
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

#ifdef PTXFILE
static CUfunction g_fn;
static void launchPtx(dim3 grid, F8nParams& p)
{
    void* args[] = { &p };
    CUresult r = cuLaunchKernel(g_fn, grid.x, grid.y, grid.z, F8N_THREADS, 1, 1, F8N_SHARED, 0, args, nullptr);
    if (r) printf("launch failed %d\n", (int)r);
}
#define LAUNCH(p) launchPtx(grid, p)
#else
#define LAUNCH(p) gemm_fp8_native<<<grid, F8N_THREADS, F8N_SHARED>>>(p)
#endif
int main()
{
#ifdef PTXFILE
    {
        cudaFree(0);
        FILE* f = fopen(PTXFILE, "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<char> b(n + 1, 0); fread(b.data(), 1, n, f); fclose(f);
        CUmodule m; CUresult r = cuModuleLoadData(&m, b.data()); if (r) { printf("load failed %d\n", (int)r); return 1; }
        cuModuleGetFunction(&g_fn, m, PTXENTRY);
        cuFuncSetAttribute(g_fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, F8N_SHARED);
    }
#endif
    #ifndef F8N_SHARED
#define F8N_SHARED (F8N_STAGES * F8N_STAGE_BYTES)
#endif
    cudaFuncSetAttribute(gemm_fp8_native, cudaFuncAttributeMaxDynamicSharedMemorySize, F8N_SHARED);
    Logical x = make(T_MAX, 1), w = make(N, 2), w2 = make(N, 3);
    unsigned char* dx = upload(rowsLayout(x, T_MAX, XCAP));
#ifdef OLD
    unsigned char* dw = upload(rowsLayout(w, N, N));
    unsigned char* dw2 = upload(rowsLayout(w2, N, N));
#else
    unsigned char* dw = upload(groupedLayout(w, N));
    unsigned char* dw2 = upload(groupedLayout(w2, N));
#endif
    size_t flushBytes = 256u << 20;
    void* flush; cudaMalloc(&flush, flushBytes);
    float* y; cudaMalloc(&y, (size_t)T_MAX * N * 4 * SPLITK);
#ifdef MS_A
    int Ms[] = { MS_A, MS_B };   // nvcc splits -D values at commas
#elif defined(MS_MID)
    int Ms[] = { 80, 96, 128, 160, 192, 256, 320, 384 };
#else
    int Ms[] = { 256, 512, 1024, 2048, 4096, 8192 };
#endif
    for (int M : Ms)
    {
        if (M > T_MAX) continue;
        F8nParams p;
        p.x = dx; p.w = dw;
#if DUAL
        p.w2 = dw2;
#endif
        p.y = y; p.T = M;
#ifdef TMA
        {
            auto enc = [](void* m, const void* base, int rank, cuuint64_t* dims, cuuint64_t* strides, cuuint32_t* box) {
                cuuint32_t es[3] = { 1, 1, 1 };
                CUresult r = cuTensorMapEncodeTiled((CUtensorMap*)m, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, (void*)base, dims, strides, box, es,
                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
                if (r) printf("encode failed %d\n", (int)r);
            };
            cuuint64_t rs = K + K / 32;
            cuuint64_t xd[2] = { K, XCAP }, xs[1] = { rs };
            cuuint32_t xb[2] = { 128, F8N_BM };
            enc(&p.mx, dx, 2, xd, xs, xb);
            cuuint64_t wd[3] = { K, 128, N / 128 }, wst[2] = { K, 128 * rs };
            cuuint32_t wb[3] = { 128, DUAL ? 64u : 128u, 1 };
            enc(&p.mw, dw, 3, wd, wst, wb);
#if DUAL
            enc(&p.mw2, dw2, 3, wd, wst, wb);
#endif
        }
#endif
#ifdef PERSIST
        int tilesAll = (M + F8N_BM - 1) / F8N_BM * ((N + F8N_BOUT - 1) / F8N_BOUT);
        dim3 grid(tilesAll < PERSIST ? tilesAll : PERSIST);
#else
        dim3 grid((M + F8N_BM - 1) / F8N_BM, (N + F8N_BOUT - 1) / F8N_BOUT, SPLITK);
#endif
        LAUNCH(p);
        cudaError_t e = cudaDeviceSynchronize();
        if (e) { printf("error %s\n", cudaGetErrorString(e)); return 1; }
        if (M == CHECK_T)
        {
            unsigned char *rc = upload(std::vector<unsigned char>(x.c.begin(), x.c.begin() + (size_t)M * K)), *rw = upload(w.c), *rw2 = upload(w2.c);
            float *rs = upload(x.s), *rws = upload(w.s), *rws2 = upload(w2.s), *yr;
            cudaMalloc(&yr, (size_t)M * N * 4);
            ref<<<dim3((N + 127) / 128, M), 128>>>(rc, rs, rw, rws, rw2, rws2, yr, M);
            cudaDeviceSynchronize();
            std::vector<float> a((size_t)M * N), b((size_t)M * N);
            for (int z = 0; z < SPLITK; z++)
            {
                std::vector<float> part(a.size());
                cudaMemcpy(part.data(), y + (size_t)z * M * N, a.size() * 4, cudaMemcpyDeviceToHost);
                for (size_t i = 0; i < a.size(); i++) a[i] = z == 0 ? part[i] : a[i] + part[i];
            }
            cudaMemcpy(b.data(), yr, b.size() * 4, cudaMemcpyDeviceToHost);
#if QOUT
            {
                // E4M3 rows of the outputs: host quantization of the reference (power-of-two scales per 128) against the kernel's codes.
                size_t qrs = N + N / 32;
                std::vector<unsigned char> q((size_t)QCAP * qrs + (size_t)(N / 128) * QCAP * 4);
                cudaMemcpy(q.data(), y, q.size(), cudaMemcpyDeviceToHost);
                long codeDiff = 0, scaleDiff = 0, denseDiff = 0; double worst = 0;
                for (int t = 0; t < M; t++)
                    for (int gq = 0; gq < N / 128; gq++)
                    {
                        float amax = 0;
                        for (int i = 0; i < 128; i++) amax = fmaxf(amax, fabsf(b[(size_t)t * N + gq * 128 + i]));
                        float d = amax > 0 ? exp2f(ceilf(log2f(amax / 448.0f))) : 1.0f;
                        float ds, dd; memcpy(&ds, &q[(size_t)t * qrs + N + gq * 4], 4); memcpy(&dd, &q[(size_t)QCAP * qrs + ((size_t)gq * QCAP + t) * 4], 4);
                        scaleDiff += ds != d; denseDiff += dd != ds;
                        for (int i = 0; i < 128; i++)
                        {
                            float r = b[(size_t)t * N + gq * 128 + i];
                            __nv_fp8_e4m3 c(r / d);
                            unsigned char got = q[(size_t)t * qrs + gq * 128 + i];
                            codeDiff += c.__x != got;
                            worst = fmax(worst, fabs(dec(got) * ds - r) / (448.0 * d));
                        }
                    }
                printf("  qout T=%d: %ld / %ld codes differ from the host quantization of the reference, %ld scales, %ld dense copies; worst |dequantized - ref| %.3g of the group range\n",
                    M, codeDiff, (long)M * N, scaleDiff, denseDiff, worst);
            }
#endif
            double maxe = 0, maxv = 0;
#if !QOUT
            for (size_t i = 0; i < a.size(); i++) { maxe = fmax(maxe, fabs(a[i] - b[i])); maxv = fmax(maxv, fabs(b[i])); }
#endif
            printf("  check T=%d: max err %.3g of max %.3g\n", M, maxe, maxv);
            cudaFree(rc); cudaFree(rw); cudaFree(rw2); cudaFree(rs); cudaFree(rws); cudaFree(rws2); cudaFree(yr);
        }
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e9;
        for (int it = 0; it < ITERS; it++)
        {
#ifndef NOFLUSH
            cudaMemsetAsync(flush, it, flushBytes);   // evict the weights from L2: decode streams them from memory
#endif
#ifdef XHOT
            cudaMemcpyAsync(dx + (size_t)XCAP * (K + K / 32) / 2, dx, (size_t)M * (K + K / 32) / 4, cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(dx, dx + (size_t)XCAP * (K + K / 32) / 2, (size_t)M * (K + K / 32) / 4, cudaMemcpyDeviceToDevice);
#endif
            cudaEventRecord(e0);
            LAUNCH(p);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); if (it > 2 && ms < best) best = ms;
        }
        double fl = 2.0 * M * N * K * (DUAL ? 2 : 1);
#ifdef SUSTAIN
        {
            // Back-to-back launches for ~SUSTAIN ms: the steady state under the board's power limit.
            int reps = (int)(SUSTAIN / best) + 1;
            for (int w = 0; w < reps / 4; w++) LAUNCH(p);
            static nvmlDevice_t nd; static bool ni = false;
            if (!ni) { nvmlInit(); nvmlDeviceGetHandleByIndex(0, &nd); ni = true; }
            cudaDeviceSynchronize();
            unsigned long long en0, en1; nvmlDeviceGetTotalEnergyConsumption(nd, &en0);
            cudaEventRecord(e0);
            for (int w = 0; w < reps; w++) LAUNCH(p);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            nvmlDeviceGetTotalEnergyConsumption(nd, &en1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            printf("  sustained: %.3f ms %.0f TFLOPS, %.1f W, %.3f pJ/flop\n", ms / reps, fl / (ms / reps) / 1e9, (en1 - en0) / ms, (en1 - en0) * 1e9 / (fl * reps));
#ifdef F8N_CYCLES
            unsigned long long z = 0, c;
            cudaMemcpyToSymbol(g_f8n_cycles, &z, 8);
            LAUNCH(p);
            cudaMemcpyFromSymbol(&c, g_f8n_cycles, 8);
            printf("  cycles %llu blocks %d: %.0f flop per SM cycle (%.1f %% of 2048)\n", c, grid.x * grid.y, fl / (double)c, fl / (double)c / 20.48);
#endif
        }
#endif
        double wbytes = (double)N * (K + K / 32) * (DUAL ? 2 : 1);
        printf("T=%5d K=%5d N=%5d dual=%d split=%d: %.3f ms %.0f TFLOPS %.0f GB/s of weights\n", M, K, N, DUAL, SPLITK, best, fl / best / 1e9, wbytes / best / 1e6);
    }
}
