// Harness of the hand-written grouped MXFP4 expert GEMM (src/Sharplight.Shaders/Kernels/gemm_moe_native.cu, or the file in KFILE):
// E experts of MXFP4R weights (WeightRepack.Mxfp4Split: 16-row groups of [16 rows x K / 2 code bytes][K / 32 blocks x 16 E8M0
// scales]), T tokens routed to S experts each (uniform random, or -DSKEW=a: share (1 + a i)^-1.5 for expert i, e.g. 0.15 or 0.6), the routing tiles and the
// expert-sorted f16 rows as moe_route / moe_gather_x16 build them. BIAS=1: the epilogue adds per-expert bias rows (zero) as
// gpt-oss-20b's does; PREFETCH=1 with the epilogue's L1 prefetches. Checks against a reference kernel (same decode, f32 sums) and
// times the launch with events (best of 20, L2 evicted before each launch: by a memset, or with EVICT=read by reading a 256 MB
// buffer, which leaves no dirty lines to write back). LOOP=n: also n launches back to back at T = 2048 (the
// power-limited steady state, section 64; sample the SM clock with nvidia-smi meanwhile). ROUTES=file replays recorded routings, also
// back to back over distinct weight copies (LAYERS=n, the sustained clocks of a decode step; see routes()); SUSTAINED_ONLY=1 skips the
// isolated launches (a clean window for bench/s73_clock_sampler.py); PER_ROUTE=1 prints each isolated launch's tiles, blocks and time.
// nvcc -O3 -arch=sm_120 -std=c++17 -DK=2880 -DN=2880 -DNFULL=5760 -DDUAL=1 -DKFILE='"gemm_moe_native.cu"' bench_moe.cu -o b
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <dlfcn.h>

// ENERGY=seconds: NVML's total energy counter (mJ, libnvidia-ml loaded at run time) read around a sustained back-to-back window
// without eviction, so energy per routing and per weight byte compare with bench/s73_marlin_replay.py --energy. NVML_GPU: the
// NVML (physical) index of the GPU.
struct NvmlEnergy
{
    void* lib = nullptr;
    void* dev = nullptr;
    int (*energy)(void*, unsigned long long*) = nullptr;
    int (*clock)(void*, int, unsigned*) = nullptr;
    bool open()
    {
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!lib) { return false; }
        auto init = (int (*)())dlsym(lib, "nvmlInit_v2");
        auto handle = (int (*)(unsigned, void**))dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
        energy = (int (*)(void*, unsigned long long*))dlsym(lib, "nvmlDeviceGetTotalEnergyConsumption");
        clock = (int (*)(void*, int, unsigned*))dlsym(lib, "nvmlDeviceGetClockInfo");
        unsigned idx = getenv("NVML_GPU") ? atoi(getenv("NVML_GPU")) : 0;
        return init && handle && energy && init() == 0 && handle(idx, &dev) == 0;
    }
    unsigned long long mj() { unsigned long long v = 0; energy(dev, &v); return v; }
    unsigned sm() { unsigned v = 0; clock(dev, 1, &v); return v; }
    unsigned mem() { unsigned v = 0; clock(dev, 2, &v); return v; }
};

#ifndef E
#define E 32
#endif
#ifndef S
#define S 4
#endif
#ifndef BIAS
#define BIAS 0
#endif
#ifndef SCALE_BASE
#define SCALE_BASE 120
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
#if DUAL && BIAS
// As gpt-oss-20b's generated epilogue: the per-expert bias rows of both halves, then SwiGLU.
#define NM_EPI_FIELDS float* gt0;
#define NM_EPI2(a, b, e, n) nm_epi_bias((a) + p.gt0[(unsigned long long)(e) * NFULL + (n)], (b) + p.gt0[(unsigned long long)(e) * NFULL + N + (n)])
#if PREFETCH
#define NM_EPI_PREFETCH2(e, n) (nm_prefetch(p.gt0 + (unsigned long long)(e) * NFULL + (n)), nm_prefetch(p.gt0 + (unsigned long long)(e) * NFULL + N + (n)))
#endif
static __device__ __forceinline__ float nm_epi_bias(float a, float b)
{
    return a / (1.0f + __expf(-a)) * b;
}
#elif DUAL
#define NM_EPI2(a, b, e, n) ((a) / (1.0f + __expf(-(a))) * (b))
#elif BIAS
#define NM_EPI_FIELDS float* moe_bias;
#define NM_EPI1(m0, e, n, t, s) ((m0) + p.moe_bias[(unsigned long long)(e) * N + (n)])
#if PREFETCH
#define NM_EPI_PREFETCH1(e, n, t, s) nm_prefetch(p.moe_bias + (unsigned long long)(e) * N + (n))
#endif
#endif
#include KFILE

#if DUAL
// RD=1 (section 97's bisect): the energy window launches read_energy.cu's KMODE copy ring instead of the GEMM, on the same routing
// tiles, weights, grid and dynamic shared memory: each block copies the 32 code bytes per K step of its 128 dual weight rows
// (gemm_moe_native's row mapping, 256-byte L2 hints) into a ring of 4 stages, waits, synchronizes, reads one word.
// RD=2: also the tile's scale rows (gemm_moe_native's scale copies), RD=3: also the activation rows (its activation copies),
// RD=4: both, RD=5: both and the E8M0 table of the prologue. The ring's stage stride is gemm_moe_native's stage size.
// F (bits): 1 the codes stored [block][row] in the stage as gemm_moe_native does (else row-major), 2 one barrier per step (the
// wait's), 4 the codes at offset 4096 of the stage (after the activations), 8 row-major with each row's two halves swapped, 16 the
// halves 64 bytes apart in one 128-byte line ([r / 4][block][r % 4]), 32 the halves 128 bytes apart ([r / 8][block][r % 8]).
template <int M, int F = 0>
__global__ void __launch_bounds__(256, 2) rd_h(NmParams p)
{
    extern __shared__ __align__(128) unsigned char rsm[];
    const unsigned tile = blockIdx.y;
    if (tile >= p.scratch[0])
    {
        return;
    }
    const unsigned char* b0 = p.w + (unsigned long long)p.scratch[1 + 3 * tile] * EXPERT_BYTES;
    const unsigned char* b1 = p.w2 + (unsigned long long)p.scratch[1 + 3 * tile] * EXPERT_BYTES;
    const int nk = (K / 2) / 32;
    const unsigned rowOff = p.scratch[2 + 3 * tile], tileRows = p.scratch[3 + 3 * tile];
    const unsigned ar = threadIdx.x >> 3, ac = threadIdx.x & 7;
    const bool aLive = ar < tileRows;
    const unsigned char* aSrc = p.x + (unsigned long long)(rowOff + (aLive ? ar : 0)) * K * 2 + ac * 16;
    const bool sLive = threadIdx.x < 16;
    const unsigned char* sSrc;
    {
        unsigned q = threadIdx.x % 4, m = threadIdx.x / 4 % 2, b = threadIdx.x / 8 % 2;
        sSrc = (m ? b1 : b0) + (unsigned long long)(blockIdx.x * 4 + q) * 16 * ROW_BYTES + 16 * (K / 2) + b * 16;
    }
    __shared__ unsigned tab[256];
    if (M == 5)
    {
        tab[threadIdx.x] = threadIdx.x * 0x10001u;
    }
    auto load = [&](int s, int kb) {
        for (int i = 0; i < 1; i++)
        {
            int idx = threadIdx.x, r = idx / 2, c = idx % 2;
            int n = blockIdx.x * 64 + (r >> 4) * 8 + (r & 7);
            const unsigned char* half = ((r >> 3) & 1) ? b1 : b0;
            const void* src = half + (size_t)(n >> 4) * 16 * ROW_BYTES + (size_t)(n & 15) * (K / 2) + kb * 32 + c * 16;
            unsigned st = (unsigned)__cvta_generic_to_shared(rsm + s * 8448);
            if ((M == 3 || M >= 4) && aLive)
            {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(st + 4096 + ar * 128 + ((ac ^ (ar & 7)) << 4)), "l"(aSrc + kb * 128));
            }
            unsigned cb = idx & 1, cr = idx >> 1;
            unsigned cd = (F & 1) ? (cb * 128 + cr) * 16 : (F & 8) ? (idx ^ 1) * 16 : (F & 16) ? ((cr >> 2) * 8 + cb * 4 + (cr & 3)) * 16
                        : (F & 32) ? ((cr >> 3) * 16 + cb * 8 + (cr & 7)) * 16 : idx * 16;
            asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" :: "r"(st + ((F & 4) ? 4096u : 0u) + cd), "l"(src));
            if ((M == 2 || M >= 4) && sLive)
            {
                asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" :: "r"(st + 8192 + threadIdx.x * 16), "l"(sSrc + kb * 32));
            }
        }
    };
    for (int s = 0; s < 3; s++) { load(s, s); asm volatile("cp.async.commit_group;\n" ::); }
    unsigned acc = 0;
    for (int kb = 0; kb < nk; kb++)
    {
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        acc ^= *(unsigned*)(rsm + (kb % 4) * 8448 + threadIdx.x * 4) + (M == 5 ? tab[acc & 255] : 0u);
        if (!(F & 2))
        {
            __syncthreads();
        }
        int nx = kb + 3;
        if (nx < nk) { load(nx % 4, nx); }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    if (acc == 0x12345678) { p.y[0] = (float)acc; }
}
#endif

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
    // The code's magnitude bits at half bits 9..11 and its sign at bit 15 (fp4_h2 of matmul.slang), then * s1 * s2.
    unsigned h = nm_hmul2(nm_hmul2(((code & 7u) << 9) | ((code & 8u) << 12), s1), s2);
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

// Reads a buffer (EVICT=read: the L2 eviction without leaving it full of dirty lines, as a preceding kernel's reads would).
__global__ void evict_read(const uint4* b, size_t n, unsigned* out)
{
    uint4 a = make_uint4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        uint4 v = __ldcs(b + i);
        a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
    }
    if ((a.x ^ a.y ^ a.z ^ a.w) == 0x9e3779b9u)
    {
        out[0] = a.x;
    }
}

template <typename T>
static T* upload(const std::vector<T>& v)
{
    T* d;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

// ROUTES=file: one routing per line (the items of each of the E experts, e.g. a decode step's layers on text), each timed as one
// launch (read eviction, best of 10) and summed. ROUTE_M: the routing tile height (default NM_BM); MIN_ROWS / MAX_ROWS: launch
// only the tiles with that many rows (a tile class); CLAMP=1: tiles of more than NM_BM rows run NM_BM of them (the time of one
// read per expert, wrong products).
static int routes(const char* file, const unsigned char* dx, const unsigned char* dw, const unsigned char* dw2, unsigned* dscratch, float* y,
                  float* dbias, void* flush, size_t flushBytes, size_t smemBytes)
{
    FILE* f = fopen(file, "r");
    if (!f) { printf("cannot open %s\n", file); return 1; }
    unsigned routeM = getenv("ROUTE_M") ? atoi(getenv("ROUTE_M")) : NM_BM;
    unsigned minRows = getenv("MIN_ROWS") ? atoi(getenv("MIN_ROWS")) : 1, maxRows = getenv("MAX_ROWS") ? atoi(getenv("MAX_ROWS")) : routeM;
    bool clamp = getenv("CLAMP") != nullptr;
    bool sustainedOnly = getenv("SUSTAINED_ONLY") != nullptr;
    int limit = getenv("LINES") ? atoi(getenv("LINES")) : 1 << 30;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    double total = 0;
    int lines = 0, launched = 0;
    unsigned counts[E];
    while (lines < limit)
    {
        int got = 0;
        for (int e = 0; e < E; e++) got += fscanf(f, "%u", &counts[e]);
        if (got != E) break;
        std::vector<unsigned> sc(1 + 3 * MAXTILES + MAXITEMS, 0);
        unsigned tiles = 0, o = 0;
        for (int e = 0; e < E; e++)
        {
            for (unsigned i = 0; i < counts[e]; i += routeM)
            {
                unsigned rows = std::min(routeM, counts[e] - i);
                if (rows >= minRows && rows <= maxRows)
                {
                    sc[1 + 3 * tiles] = e; sc[2 + 3 * tiles] = o + i; sc[3 + 3 * tiles] = clamp ? std::min(rows, (unsigned)NM_BM) : rows; tiles++;
                }
            }
            for (unsigned i = 0; i < counts[e]; i++, o++) sc[1 + 3 * MAXTILES + o] = o;
        }
        sc[0] = tiles;
        lines++;
        if (tiles == 0) continue;
        if (sustainedOnly) { launched++; continue; }
        cudaMemcpy(dscratch, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice);
        NmParams p;
        p.x = dx; p.w = dw;
#if DUAL
        p.w2 = dw2;
#endif
        p.scratch = dscratch; p.y = y; p.T = o / S;
#if BIAS && DUAL
        p.gt0 = dbias;
#elif BIAS
        p.moe_bias = dbias;
#endif
        dim3 grid((N + NM_BOUT - 1) / NM_BOUT, tiles);
        float best = 1e9;
        for (int it = 0; it < 10; it++)
        {
            evict_read<<<1024, 256>>>((const uint4*)flush, flushBytes / 16, (unsigned*)y);
            cudaEventRecord(e0);
            gemm_moe_native<<<grid, NM_THREADS, smemBytes>>>(p);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); if (it > 1 && ms < best) best = ms;
        }
        total += best;
        launched++;
        if (getenv("PER_ROUTE")) printf("route %d tiles %u blocks %u ms %.4f\n", lines - 1, tiles, grid.x * tiles, best);
    }
    // LAYERS=n: the routings again back to back without eviction, each on the weights of layer (line % n) (n distinct copies, as
    // a decode step's layers read theirs: page-table reach and the sustained clocks of a step).
    if (getenv("LAYERS"))
    {
        int layers = atoi(getenv("LAYERS"));
        std::vector<unsigned char*> copies(layers);
        for (int l = 0; l < layers; l++)
        {
            if (cudaMalloc(&copies[l], (size_t)E * EXPERT_BYTES)) { printf("cannot allocate layer %d\n", l); return 1; }
            cudaMemcpy(copies[l], dw, (size_t)E * EXPERT_BYTES, cudaMemcpyDeviceToDevice);
        }
        std::vector<std::vector<unsigned>> all;
        rewind(f);
        for (int l = 0; l < lines; l++)
        {
            std::vector<unsigned> sc(1 + 3 * MAXTILES + MAXITEMS, 0);
            unsigned tiles = 0, o = 0;
            for (int e = 0; e < E; e++) { if (fscanf(f, "%u", &counts[e]) != 1) counts[e] = 0; }
            for (int e = 0; e < E; e++)
            {
                for (unsigned i = 0; i < counts[e]; i += routeM)
                {
                    unsigned rows = std::min(routeM, counts[e] - i);
                    if (rows >= minRows && rows <= maxRows)
                    {
                        sc[1 + 3 * tiles] = e; sc[2 + 3 * tiles] = o + i; sc[3 + 3 * tiles] = clamp ? std::min(rows, (unsigned)NM_BM) : rows; tiles++;
                    }
                }
                for (unsigned i = 0; i < counts[e]; i++, o++) sc[1 + 3 * MAXTILES + o] = o;
            }
            sc[0] = tiles;
            all.push_back(sc);
        }
        size_t words = 1 + 3 * MAXTILES + MAXITEMS;
        unsigned* dall; cudaMalloc(&dall, all.size() * words * 4);
        for (size_t l = 0; l < all.size(); l++) cudaMemcpy(dall + l * words, all[l].data(), words * 4, cudaMemcpyHostToDevice);
        float best = 1e9;
        int passes = getenv("PASSES") ? atoi(getenv("PASSES")) : 6;
        for (int it = 0; it < passes; it++)
        {
            evict_read<<<1024, 256>>>((const uint4*)flush, flushBytes / 16, (unsigned*)y);
            cudaEventRecord(e0);
            for (size_t l = 0; l < all.size(); l++)
            {
                NmParams p;
                p.x = dx; p.w = copies[l % layers];
#if DUAL
                p.w2 = copies[l % layers] + (size_t)N * ROW_BYTES;
#endif
                p.scratch = dall + l * words; p.y = y; p.T = 64;
#if BIAS && DUAL
                p.gt0 = dbias;
#elif BIAS
                p.moe_bias = dbias;
#endif
                dim3 grid((N + NM_BOUT - 1) / NM_BOUT, std::max(1u, all[l][0]));
                gemm_moe_native<<<grid, NM_THREADS, smemBytes>>>(p);
            }
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); if (it > 0 && ms < best) best = ms;
        }
        printf("back to back over %d layer copies: %.4f ms per routing\n", layers, best / lines);
        NvmlEnergy nv;
        if (getenv("ENERGY") && nv.open())
        {
            double seconds = atof(getenv("ENERGY"));
            const int rdMode = getenv("RD") ? atoi(getenv("RD")) : 0;
#if DUAL
            cudaFuncSetAttribute(rd_h<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
            cudaFuncSetAttribute(rd_h<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
            cudaFuncSetAttribute(rd_h<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
            cudaFuncSetAttribute(rd_h<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
            cudaFuncSetAttribute(rd_h<5>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
            cudaFuncSetAttribute(rd_h<1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
            cudaFuncSetAttribute(rd_h<1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
            cudaFuncSetAttribute(rd_h<1, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
            cudaFuncSetAttribute(rd_h<1, 7>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
            cudaFuncSetAttribute(rd_h<1, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
            cudaFuncSetAttribute(rd_h<1, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
            cudaFuncSetAttribute(rd_h<1, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
#endif
            auto launchAll = [&]() {
                for (size_t l = 0; l < all.size(); l++)
                {
                    NmParams p;
                    p.x = dx; p.w = copies[l % layers];
#if DUAL
                    p.w2 = copies[l % layers] + (size_t)N * ROW_BYTES;
#endif
                    p.scratch = dall + l * words; p.y = y; p.T = 64;
#if BIAS && DUAL
                    p.gt0 = dbias;
#elif BIAS
                    p.moe_bias = dbias;
#endif
                    dim3 grid((N + NM_BOUT - 1) / NM_BOUT, std::max(1u, all[l][0]));
#if DUAL
                    if (rdMode)
                    {
                        switch (rdMode)
                        {
                        case 2: rd_h<2><<<grid, 256, smemBytes>>>(p); break;
                        case 3: rd_h<3><<<grid, 256, smemBytes>>>(p); break;
                        case 4: rd_h<4><<<grid, 256, smemBytes>>>(p); break;
                        case 5: rd_h<5><<<grid, 256, smemBytes>>>(p); break;
                        case 11: rd_h<1, 1><<<grid, 256, smemBytes>>>(p); break;
                        case 12: rd_h<1, 2><<<grid, 256, smemBytes>>>(p); break;
                        case 14: rd_h<1, 4><<<grid, 256, smemBytes>>>(p); break;
                        case 17: rd_h<1, 7><<<grid, 256, smemBytes>>>(p); break;
                        case 18: rd_h<1, 8><<<grid, 256, smemBytes>>>(p); break;
                        case 116: rd_h<1, 16><<<grid, 256, smemBytes>>>(p); break;
                        case 132: rd_h<1, 32><<<grid, 256, smemBytes>>>(p); break;
                        default: rd_h<1><<<grid, 256, smemBytes>>>(p); break;
                        }
                        continue;
                    }
#endif
                    gemm_moe_native<<<grid, NM_THREADS, smemBytes>>>(p);
                }
            };
            // One second of warm-up at the sustained clocks, then the measured window (whole passes).
            auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 1.0) { launchAll(); cudaDeviceSynchronize(); }
            unsigned long long e0mj = nv.mj();
            unsigned smSum = 0, smN = 0, memSum = 0;
            int passesE = 0;
            cudaEventRecord(e0);
            t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds)
            {
                launchAll(); launchAll(); cudaDeviceSynchronize(); passesE += 2; smSum += nv.sm(); memSum += nv.mem(); smN++;
            }
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            unsigned long long e1mj = nv.mj();
            double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            double j = (e1mj - e0mj) / 1000.0, per = ms / passesE / all.size();
            // Weight bytes read per routing: each routing tile reads its expert's codes and scales of the matrix (or both of DUAL).
            double tilesSum = 0;
            for (auto& r : all) { tilesSum += r[0]; }
            double bytes = tilesSum / all.size() * (double)N * ROW_BYTES * (DUAL ? 2 : 1);
            printf("energy: %d passes %.2f s: %.4f ms per routing, %.1f W, %.3f mJ per routing, %.1f MB weights per routing (tile reads), "
                   "%.4f nJ per weight byte, SM %u MHz, memory %u MHz\n",
                   passesE, wall, per, j / wall, j * 1000.0 / passesE / all.size(), bytes / 1e6, j * 1e9 / passesE / all.size() / bytes,
                   smN ? smSum / smN : 0, smN ? memSum / smN : 0);
        }
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err) { printf("error %s\n", cudaGetErrorString(err)); return 1; }
    printf("routes %d (%d launched), ROUTE_M=%u rows %u-%u clamp=%d BM=%d BN=%d stages=%d: %.4f ms per routing\n", lines, launched, routeM, minRows,
           maxRows, clamp ? 1 : 0, NM_BM, NM_BN, NM_STAGES, total / lines);
    return 0;
}

int main(int argc, char** argv)
{
#ifdef NM_SMEM_BYTES
    size_t smemBytes = (size_t)NM_SMEM_BYTES;   // a kernel variant with its own shared-memory layout (bench/results/s96/rejected)
#else
    size_t smemBytes = (size_t)NM_STAGES * NM_STAGE_BYTES;
#endif
    if (cudaFuncSetAttribute(gemm_moe_native, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes)) { printf("smem %zu too large\n", smemBytes); return 1; }
    std::vector<unsigned char> wh((size_t)E * EXPERT_BYTES);
    for (size_t i = 0; i < wh.size(); i++) wh[i] = (unsigned char)hash((unsigned)i);
    // Scales: E8M0 SCALE_BASE .. SCALE_BASE + 7 (default 120 .. 127: 2^-7 .. 1; 126 also checks the factors above 1) in the scale part
    // of every 16-row group.
    for (size_t e = 0; e < E; e++)
        for (size_t g = 0; g < NFULL / 16; g++)
            for (size_t i = 0; i < (K / 32) * 16; i++)
                wh[e * EXPERT_BYTES + g * 16 * ROW_BYTES + 16 * (K / 2) + i] = (unsigned char)(SCALE_BASE + hash((unsigned)(e * 7919 + g * 131 + i)) % 8);
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
    void* flush; cudaMalloc(&flush, flushBytes); cudaMemset(flush, 1, flushBytes);
    // BIAS: zero bias rows (E x NFULL f32), read by the epilogue as gpt-oss's are.
    float* dbias; cudaMalloc(&dbias, (size_t)E * NFULL * 4); cudaMemset(dbias, 0, (size_t)E * NFULL * 4);
    unsigned* dscratch; cudaMalloc(&dscratch, (1 + 3 * MAXTILES + MAXITEMS) * 4);

    if (getenv("ROUTES"))
    {
        return routes(getenv("ROUTES"), dx, dw, dw2, dscratch, y, dbias, flush, flushBytes, smemBytes);
    }

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
#if BIAS && DUAL
        p.gt0 = dbias;
#elif BIAS
        p.moe_bias = dbias;
#endif
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
            if (getenv("EVICT") && strcmp(getenv("EVICT"), "read") == 0)
            {
                evict_read<<<1024, 256>>>((const uint4*)flush, flushBytes / 16, (unsigned*)y);
            }
            else
            {
                cudaMemsetAsync(flush, it, flushBytes);
            }
            cudaEventRecord(e0);
            gemm_moe_native<<<grid, NM_THREADS, smemBytes>>>(p);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); if (it > 2 && ms < best) best = ms;
        }
        double fl = 2.0 * items * N * K * (DUAL ? 2 : 1);
        printf("T=%5d items=%6d tiles=%4u K=%d N=%d dual=%d BM=%d BN=%d: %.3f ms %.1f TFLOPS\n", T, items, tiles, K, N, DUAL, NM_BM, NM_BN, best, fl / best / 1e9);
    }
}
