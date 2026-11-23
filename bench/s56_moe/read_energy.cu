// Energy per byte of weight-copy patterns at the power limit (docs/benchmarks.md section 96): blocks of 256 threads copy a tile of
// ROWS weight rows (K = 2880 MXFP4R rows of 1440 code bytes in 16-row groups, as gemm_moe_native.cu's decode tiles read them) into a
// ring of S shared stages with cp.async, one barrier per stage, back to back over a 4 GB buffer (no L2 reuse), for SECONDS; NVML's
// energy counter is read around the window. PAT:
//   0 rows: C code bytes per row and stage, row-strided (gemm_moe_native.cu's form), HINT=256|128|0 the L2 prefetch hint
//   1 contiguous: the same bytes per stage as one contiguous run per block (a repacked layout read in 16 bytes per thread)
//   2 rows, hint only on the copy at the start of each 32-byte row chunk (the other copy of the pair without a hint)
// Random data unless CONST_DATA=1 (constant bytes draw less power). MMA=1 adds a dependent use of every copied word (an xor), so the copies are consumed from shared memory as a kernel would.
// nvcc -O3 -arch=sm_120 -std=c++17 -DPAT=0 -DC=32 -DS=4 read_energy.cu -o re -ldl && SECONDS_RUN=5 ./re
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <random>
#include <vector>
// KFILE=<gemm_moe_native.cu or a variant>: the same host loop launches the expert GEMM instead of rd (section 97's bisect): GEMM_TILES
// routing tiles (default 28) of GEMM_ROWS items each (default 9, a 64-sequence decode tile), each tile one random expert of the
// buffer, the decode launch's gate / up shape (K = N = 2880, dual, 32-row tiles of 128 weight rows, 4 stages, 2 blocks per SM).
#ifdef KFILE
#define K 2880
#define N 2880
#define NFULL 5760
#define DUAL 1
#define ROW_BYTES (17ull * K / 32)
#define EXPERT_BYTES (ROW_BYTES * NFULL)
#define T_MAX 2048
#define MAXITEMS (T_MAX * 4)
#define MAXTILES (MAXITEMS / 16 + 32)
#define NM_EPI2(a, b, e, n) ((a) / (1.0f + __expf(-(a))) * (b))
#define S 4
#include KFILE
#undef S
#endif
#ifndef PAT
#define PAT 0
#endif
#ifndef C
#define C 32
#endif
#ifndef S
#define S 4
#endif
#ifndef HINT
#define HINT 256
#endif
#ifndef ROWS
#define ROWS 128
#endif
#ifndef MMA
#define MMA 0
#endif
// WORK=n: n dependent FMAs per thread and stage (SM load that brings the clock down to the power limit, as the GEMM's decode and MMAs).
#ifndef WORK
#define WORK 0
#endif
#ifndef AHEAD
#define AHEAD 0
#endif
#ifndef KMODE
#define KMODE 0
#endif
#ifndef MINB
#define MINB 2
#endif
#define ROWB 1530
#define KB 1440
#define TILE_BYTES ((size_t)ROWS * ROWB)

static __device__ __forceinline__ void cp16(unsigned dst, const void* src, bool hint)
{
#if HINT == 256
    if (hint) { asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" :: "r"(dst), "l"(src)); return; }
#elif HINT == 128
    if (hint) { asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dst), "l"(src)); return; }
#endif
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}

__global__ void __launch_bounds__(256, MINB) rd(const unsigned char* w, const unsigned* order, unsigned* out)
{
    extern __shared__ __align__(128) unsigned char sm[];
#if KMODE
    // KMODE=1: grid (45 column tiles, 28 routing tiles), each routing tile one 5760-row expert (order: expert per tile)
    const unsigned char* base = w + (size_t)order[blockIdx.y] * 5760 * ROWB;
#else
    const unsigned char* base = w + (size_t)order[blockIdx.x] * TILE_BYTES;
#endif
    constexpr int perStage = ROWS * C / 16;
    constexpr int per = (perStage + 255) / 256;
    const int nk = KB / C;
    auto load = [&](int s, int kb) {
#pragma unroll
        for (int i = 0; i < per; i++)
        {
            int idx = threadIdx.x + i * 256;
            if (idx >= perStage) { break; }
            unsigned dst = (unsigned)__cvta_generic_to_shared(sm + s * ROWS * C + idx * 16);
#if PAT == 1
            const void* src = base + (size_t)kb * ROWS * C + idx * 16;
            cp16(dst, src, true);
#else
            int r = idx / (C / 16), c = idx % (C / 16);
#if KMODE
            // gemm_moe_native's dual gate / up rows: 8-row groups alternate between the expert's two halves, column tile blockIdx.x
            int n = blockIdx.x * 64 + (r >> 4) * 8 + (r & 7);
            const unsigned char* half = ((r >> 3) & 1) ? base + (size_t)2880 * ROWB : base;
            const void* src = half + (size_t)(n >> 4) * 16 * ROWB + (size_t)(n & 15) * KB + kb * C + c * 16;
#else
            const void* src = base + (size_t)(r / 16) * 16 * ROWB + (size_t)(r % 16) * KB + kb * C + c * 16;
#endif
            cp16(dst, src, PAT == 2 ? (c & 1) == 0 : true);
#endif
        }
    };
    for (int s = 0; s < S - 1; s++) { if (s < nk) { load(s, s); } asm volatile("cp.async.commit_group;\n" ::); }
    unsigned acc = 0;
    for (int kb = 0; kb < nk; kb++)
    {
        // AHEAD=1: stage kb + 1 complete as well (gemm_moe_native decodes the next stage's first block at the end of a step).
        asm volatile("cp.async.wait_group %0;\n" :: "n"(S - 2 - AHEAD));
        __syncthreads();
#if MMA
        const unsigned* st = (const unsigned*)(sm + (kb % S) * ROWS * C);
        for (int i = threadIdx.x; i < ROWS * C / 4; i += 256) { acc ^= st[i]; }
#else
        acc ^= *(unsigned*)(sm + (kb % S) * ROWS * C + threadIdx.x * 4 % (ROWS * C));
#endif
#if WORK
        {
            float f = __uint_as_float(acc | 0x3f800000u);
#pragma unroll 8
            for (int i = 0; i < WORK; i++) { f = fmaf(f, 0.999f, 0.001f); }
            acc ^= __float_as_uint(f) & 1u;
        }
#endif
        __syncthreads();
        int nx = kb + S - 1;
        if (nx < nk) { load(nx % S, nx); }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    if (acc == 0x12345678) { out[0] = acc; }
}

// Random bytes (a hash of the word index): DRAM and crossbar power depend on the data, and weights are not constant.
__global__ void fill(unsigned* w, size_t n)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        unsigned x = (unsigned)i * 2654435761u;
        x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        w[i] = x;
    }
}

int main()
{
    // BUF_GB: buffer size (footprint over the run); SHUFFLE=1: each launch reads its tiles in a random order of the whole buffer
    // (as routing tiles pick experts across the layers' weights), else consecutive tiles rotating through it.
    size_t bytes = (size_t)(getenv("BUF_GB") ? atoi(getenv("BUF_GB")) : 4) << 30, tiles = bytes / TILE_BYTES;
    unsigned char* w;
    if (cudaMalloc(&w, bytes)) { printf("alloc failed\n"); return 1; }
    if (getenv("CONST_DATA")) { cudaMemset(w, 3, bytes); } else { fill<<<4096, 256>>>((unsigned*)w, bytes / 4); }
    unsigned* o; cudaMalloc(&o, 4);
    // SMEM_KB: dynamic shared memory per block (pads the ring: fewer resident blocks per SM, as the GEMM's 34 KB stages).
    int smem = getenv("SMEM_KB") ? atoi(getenv("SMEM_KB")) * 1024 : S * ROWS * C;
    cudaFuncSetAttribute(rd, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    const unsigned blocks = 1260;   // a gate / up decode launch: 45 column tiles x 28 routing tiles
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    auto init = (int (*)())dlsym(lib, "nvmlInit_v2");
    auto handle = (int (*)(unsigned, void**))dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
    auto energy = (int (*)(void*, unsigned long long*))dlsym(lib, "nvmlDeviceGetTotalEnergyConsumption");
    auto clock = (int (*)(void*, int, unsigned*))dlsym(lib, "nvmlDeviceGetClockInfo");
    void* dev = nullptr;
    init();
    handle(getenv("NVML_GPU") ? atoi(getenv("NVML_GPU")) : 0, &dev);
    double seconds = getenv("SECONDS_RUN") ? atof(getenv("SECONDS_RUN")) : 5.0;
    const int orders = 64;
    std::vector<unsigned> hostOrder((size_t)orders * blocks);
    bool shuffle = getenv("SHUFFLE") != nullptr;
    std::mt19937 rng(1);
    size_t off = 0;
    for (int l = 0; l < orders; l++)
    {
        for (unsigned b = 0; b < blocks; b++)
        {
            hostOrder[(size_t)l * blocks + b] = KMODE ? (unsigned)(rng() % (bytes / (5760ull * ROWB)))
                                                      : shuffle ? (unsigned)(rng() % tiles) : (unsigned)((off + b) % tiles);
        }
        off += blocks;
    }
    unsigned* order;
    cudaMalloc(&order, hostOrder.size() * 4);
    cudaMemcpy(order, hostOrder.data(), hostOrder.size() * 4, cudaMemcpyHostToDevice);
#ifdef KFILE
    // The GEMM's routing words per order: tile i = {expert order[i], rows i * GEMM_ROWS .., GEMM_ROWS rows}, items in order.
    const unsigned gt = getenv("GEMM_TILES") ? atoi(getenv("GEMM_TILES")) : 28, gr = getenv("GEMM_ROWS") ? atoi(getenv("GEMM_ROWS")) : 9;
    const size_t words = 1 + 3 * MAXTILES + MAXITEMS;
    std::vector<unsigned> sc((size_t)orders * words, 0);
    const unsigned experts = (unsigned)(bytes / EXPERT_BYTES);
    for (int l = 0; l < orders; l++)
    {
        unsigned* q = sc.data() + (size_t)l * words;
        q[0] = gt;
        for (unsigned t = 0; t < gt; t++)
        {
            q[1 + 3 * t] = (unsigned)(rng() % experts); q[2 + 3 * t] = t * gr; q[3 + 3 * t] = gr;
        }
        for (unsigned i = 0; i < gt * gr; i++) { q[1 + 3 * MAXTILES + i] = i; }
    }
    unsigned* dsc; cudaMalloc(&dsc, sc.size() * 4); cudaMemcpy(dsc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice);
    unsigned char* gx; cudaMalloc(&gx, (size_t)MAXITEMS * (K * 2 + 4)); cudaMemset(gx, 0, (size_t)MAXITEMS * (K * 2 + 4));
    float* gy; cudaMalloc(&gy, (size_t)MAXITEMS * N * 4);
    size_t gsmem = getenv("SMEM_KB") ? atoi(getenv("SMEM_KB")) * 1024 : (size_t)NM_STAGES * NM_STAGE_BYTES;
    cudaFuncSetAttribute(gemm_moe_native, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gsmem);
    auto run = [&](double sec, unsigned long long& launches) {
        auto t0 = std::chrono::steady_clock::now();
        launches = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < sec)
        {
            for (int i = 0; i < orders; i++)
            {
                NmParams p;
                p.x = gx; p.w = w; p.w2 = w + (size_t)N * ROW_BYTES; p.scratch = dsc + (size_t)i * words; p.y = gy; p.T = 64;
                gemm_moe_native<<<dim3((N + NM_BOUT - 1) / NM_BOUT, gt), NM_THREADS, gsmem>>>(p);
                launches++;
            }
            cudaDeviceSynchronize();
        }
    };
#else
    // Each launch reads a different set of tiles (one of 64 orders).
    auto run = [&](double sec, unsigned long long& launches) {
        auto t0 = std::chrono::steady_clock::now();
        launches = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < sec)
        {
            for (int i = 0; i < orders; i++)
            {
#if KMODE
                rd<<<dim3(45, 28), 256, smem>>>(w, order + (size_t)i * blocks, o);
#else
                rd<<<blocks, 256, smem>>>(w, order + (size_t)i * blocks, o);
#endif
                launches++;
            }
            cudaDeviceSynchronize();
        }
    };
#endif
    unsigned long long n;
    run(1.0, n);
    unsigned long long e0 = 0, e1 = 0;
    energy(dev, &e0);
    auto t0 = std::chrono::steady_clock::now();
    run(seconds, n);
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    energy(dev, &e1);
    unsigned smc = 0, memc = 0;
    clock(dev, 1, &smc);
    clock(dev, 2, &memc);
    cudaError_t err = cudaGetLastError();
#ifdef KFILE
    // Weight bytes the tiles read (codes and scales of 128 rows per block, as the harness's "tile reads").
    double moved = (double)n * gt * 2.0 * N * ROW_BYTES;
    printf("GEMM %u tiles x %u rows: ", gt, gr);
#else
    double moved = (double)n * blocks * ROWS * KB;   // code bytes
#endif
    double j = (e1 - e0) / 1000.0;
    printf("PAT=%d C=%d S=%d HINT=%d ROWS=%d MMA=%d WORK=%d: %.1f GB/s, %.1f W, %.4f nJ per byte, SM %u MHz, memory %u MHz %s\n", PAT, C, S, HINT, ROWS, MMA, WORK,
           moved / wall / 1e9, j / wall, j * 1e9 / moved, smc, memc, err ? cudaGetErrorString(err) : "");
    return 0;
}
