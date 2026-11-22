// FP8 decode GEMM written in CUDA C++ for the CUDA provider (DeviceFeatures.NativeGemmFp8): y[t, n] = sum_k x[t, k] * W[n, k] for
// batches of a few rows (at most 8 * F8D_NT), E4M3 activations and weights in 128-value scale groups along K. At these sizes the
// product is bound by how fast the weights stream from memory, so the kernel has no shared-memory stages and no block barrier in
// its main loop: every warp reads its own weight rows straight into registers (16-byte loads, F8D_DEPTH K blocks ahead) and runs
// the m16n8k32 E4M3 MMAs with the weights as the A operand (16 weight rows per fragment) and the activation rows as B (8 tokens
// per fragment).
//
// The product of a K block does not depend on the order of its 128 values, so both operands use the same permutation of K inside
// each 64-value chunk: lane (g, c) of a warp loads 16 contiguous bytes at chunk offset 16 c of weight rows g and g + 8 and of
// token g, and their 4-byte words fill the fragments of the chunk's two MMAs (words 0, 1 the first MMA's k 4c..4c+3 and
// 16+4c..16+4c+3, words 2, 3 the second's).
//
// Operands:
//  * x: the rows of quantize_fp8 ([K codes][K / 128 f32 scales] per row, row stride K + K / 32).
//  * W (and W2): the 128-row groups of WeightRepack.Fp8B128Grouped (f8e4m3_b128g): per group [128 rows of K codes][K / 128 blocks
//    of 128 f32 scales]. Requires N % 128 == 0.
//
// Work split: a warp owns F8D_RF fragments of 16 weight rows (DUAL: F8D_RF / 2 of w and the same rows of w2, so that it holds both
// results of an output and stores F8D_EPI2(a, b), the generated epilogue) over 1 / F8D_KS of K; the F8D_KS warps of one row tile
// add their sums through shared memory in a fixed order (ascending K ranges). A block runs F8D_RT row tiles. Per K block a warp
// sums the block's MMAs in a fresh accumulator and adds it scaled by sx[t, block] * sw[n, block], as gemm_fp8_native.cu does.
//
// Blocks walk the block tiles (F8D_RT row tiles) from blockIdx.x in steps of gridDim.x, so a grid of the GPU's resident blocks
// runs as one wave. The weights do not depend on earlier launches: launched as a programmatic dependent, a block loads its first
// F8D_DEPTH K blocks of weights (and asks L2 for the next F8D_L2 K blocks of its rows) before griddepcontrol.wait, so the weight
// stream starts while the launch that writes x still runs.
//
// Specialization macros: K, N, Y_STRIDE, DUAL, F8D_EPI2 (DUAL), F8D_NT (token fragments: 1, 2 or 4), F8D_RF, F8D_KS, F8D_RT,
// F8D_DEPTH (K blocks in flight per warp), F8D_L2 (K blocks prefetched to L2 before the wait, 0 for none). Requires
// K % (128 * F8D_KS) == 0, N % 128 == 0, N % F8D_TILE_ROWS == 0. Grid: x = at most the block tiles.
#ifndef DUAL
#define DUAL 0
#endif
#ifndef F8D_NT
#define F8D_NT 2
#endif
#ifndef F8D_RF
#define F8D_RF 2
#endif
#ifndef F8D_KS
#define F8D_KS 1
#endif
#ifndef F8D_RT
#define F8D_RT 4
#endif
#ifndef F8D_DEPTH
#define F8D_DEPTH 1
#endif
#ifndef F8D_L2
#define F8D_L2 0
#endif
#if DUAL && (F8D_RF % 2) != 0
#error "gemm_fp8_decode: DUAL needs an even F8D_RF"
#endif
#define F8D_THREADS (F8D_RT * F8D_KS * 32)
#define F8D_RS ((unsigned long long)K + K / 32)
#define F8D_GROUP_BYTES (128ull * F8D_RS)
#define F8D_NKB (K / 128 / F8D_KS)
#if DUAL
#define F8D_OUT_FRAGS (F8D_RF / 2)
#else
#define F8D_OUT_FRAGS F8D_RF
#endif
// Outputs (weight rows of w) per warp and per block tile.
#define F8D_ROWS (16 * F8D_OUT_FRAGS)
#define F8D_TILE_ROWS (F8D_ROWS * F8D_RT)
#define F8D_TILES (N / F8D_TILE_ROWS)

typedef unsigned char f8d_u8;
typedef unsigned int f8d_u32;
typedef unsigned long long f8d_u64;

struct F8dParams
{
    const f8d_u8* x;
    const f8d_u8* w;
#if DUAL
    const f8d_u8* w2;
#endif
    float* y;
    f8d_u32 T;
};

// A weight load: read once, kept out of L1.
static __device__ __forceinline__ uint4 f8d_ldw(const f8d_u8* p)
{
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// An activation load: every warp of the K range reads the same rows.
static __device__ __forceinline__ uint4 f8d_ldx(const f8d_u8* p)
{
    return __ldg((const uint4*)p);
}

static __device__ __forceinline__ void f8d_mma(float* c, f8d_u32 a0, f8d_u32 a1, f8d_u32 a2, f8d_u32 a3, f8d_u32 b0, f8d_u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Waits for the launch this one depends on (its completion and memory); a no-op without a programmatic dependency.
static __device__ __forceinline__ void f8d_wait_previous()
{
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

static __device__ __forceinline__ void f8d_prefetch_l2(const void* p, f8d_u32 bytes)
{
#if __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(p), "r"(bytes) : "memory");
#endif
}

// One K block's operands of a lane: weight rows g and g + 8 of each fragment and token g + 8 j, both 64-value chunks; the scales of
// rows g, g + 8 and of tokens 2 c, 2 c + 1 (+ 8 j).
struct F8dStage
{
    uint4 w[F8D_RF][2][2];
    uint4 x[F8D_NT][2];
    float sw[F8D_RF][2];
    float sx[F8D_NT][2];
};

extern "C" __global__ void __launch_bounds__(F8D_THREADS) gemm_fp8_decode(F8dParams p)
{
    const f8d_u32 lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const f8d_u32 rt = warp / F8D_KS, ks = warp % F8D_KS;
    const f8d_u32 g = lane >> 2, c = lane & 3;
    const f8d_u32 kb0 = ks * F8D_NKB;

    // Per fragment: the code and scale pointers of row g of the block's current tile (row g + 8: K and 32 bytes further).
    const f8d_u8* wc[F8D_RF];
    const f8d_u8* wsc[F8D_RF];
    auto rows = [&](f8d_u32 tile) {
        const f8d_u32 n0 = (tile * F8D_RT + rt) * F8D_ROWS;
#pragma unroll
        for (int f = 0; f < F8D_RF; f++)
        {
#if DUAL
            const f8d_u8* w = f < F8D_OUT_FRAGS ? p.w : p.w2;
            const f8d_u32 n = n0 + (f % F8D_OUT_FRAGS) * 16 + g;
#else
            const f8d_u8* w = p.w;
            const f8d_u32 n = n0 + f * 16 + g;
#endif
            const f8d_u8* group = w + (n >> 7) * F8D_GROUP_BYTES;
            wc[f] = group + (f8d_u64)(n & 127) * K + c * 16;
            wsc[f] = group + 128ull * K + (n & 127) * 4;
        }
    };

    const f8d_u8* xc[F8D_NT];
    const f8d_u8* xsc[F8D_NT][2];
#pragma unroll
    for (int j = 0; j < F8D_NT; j++)
    {
        xc[j] = p.x + (f8d_u64)min(g + j * 8, p.T - 1) * F8D_RS + c * 16;
        xsc[j][0] = p.x + (f8d_u64)min(2 * c + j * 8, p.T - 1) * F8D_RS + K;
        xsc[j][1] = p.x + (f8d_u64)min(2 * c + 1 + j * 8, p.T - 1) * F8D_RS + K;
    }

    auto loadW = [&](F8dStage& s, f8d_u32 kb) {
#pragma unroll
        for (int f = 0; f < F8D_RF; f++)
        {
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
                s.w[f][h][0] = f8d_ldw(wc[f] + h * 8 * K + kb * 128);
                s.w[f][h][1] = f8d_ldw(wc[f] + h * 8 * K + kb * 128 + 64);
            }
            s.sw[f][0] = __ldg((const float*)(wsc[f] + kb * 512));
            s.sw[f][1] = __ldg((const float*)(wsc[f] + kb * 512 + 32));
        }
    };
    auto loadX = [&](F8dStage& s, f8d_u32 kb) {
#pragma unroll
        for (int j = 0; j < F8D_NT; j++)
        {
            s.x[j][0] = f8d_ldx(xc[j] + kb * 128);
            s.x[j][1] = f8d_ldx(xc[j] + kb * 128 + 64);
            s.sx[j][0] = __ldg((const float*)(xsc[j][0] + kb * 4));
            s.sx[j][1] = __ldg((const float*)(xsc[j][1] + kb * 4));
        }
    };

#if F8D_KS > 1
    __shared__ float red[F8D_RT][F8D_KS - 1][F8D_RF * F8D_NT * 4][32];
#endif
    F8dStage st[F8D_DEPTH];
    bool first = true;
    for (f8d_u32 tile = blockIdx.x; tile < F8D_TILES; tile += gridDim.x)
    {
        rows(tile);
#pragma unroll
        for (int d = 0; d < F8D_DEPTH; d++)
        {
            if (d < F8D_NKB)
            {
                loadW(st[d], kb0 + d);
            }
        }
        if (first)
        {
#if F8D_L2 > 0
            // Lane l < 16 F8D_RF asks L2 for K blocks [F8D_DEPTH, F8D_DEPTH + F8D_L2) of its range of weight row l of the warp.
            if (lane < 16 * F8D_RF && F8D_NKB > F8D_DEPTH)
            {
                const f8d_u32 f = lane >> 4, r = lane & 15;
                const f8d_u32 kbs = min(F8D_L2, F8D_NKB - F8D_DEPTH);
                f8d_prefetch_l2(wc[f] - (g * K + c * 16) + (f8d_u64)r * K + (kb0 + F8D_DEPTH) * 128, kbs * 128);
            }
#endif
            f8d_wait_previous();
            first = false;
        }
#pragma unroll
        for (int d = 0; d < F8D_DEPTH; d++)
        {
            if (d < F8D_NKB)
            {
                loadX(st[d], kb0 + d);
            }
        }

        float acc[F8D_RF][F8D_NT][4];
#pragma unroll
        for (int f = 0; f < F8D_RF; f++)
        {
#pragma unroll
            for (int j = 0; j < F8D_NT; j++)
            {
#pragma unroll
                for (int e = 0; e < 4; e++)
                {
                    acc[f][j][e] = 0.0f;
                }
            }
        }

        for (f8d_u32 i = 0; i < F8D_NKB; i += F8D_DEPTH)
        {
#pragma unroll
            for (int d = 0; d < F8D_DEPTH; d++)
            {
                if (i + d < F8D_NKB)
                {
                    F8dStage& s = st[d];
                    float part[F8D_RF][F8D_NT][4];
#pragma unroll
                    for (int f = 0; f < F8D_RF; f++)
                    {
#pragma unroll
                        for (int j = 0; j < F8D_NT; j++)
                        {
#pragma unroll
                            for (int e = 0; e < 4; e++)
                            {
                                part[f][j][e] = 0.0f;
                            }
#pragma unroll
                            for (int ch = 0; ch < 2; ch++)
                            {
                                const uint4 lo = s.w[f][0][ch], hi = s.w[f][1][ch], xv = s.x[j][ch];
                                f8d_mma(part[f][j], lo.x, hi.x, lo.y, hi.y, xv.x, xv.y);
                                f8d_mma(part[f][j], lo.z, hi.z, lo.w, hi.w, xv.z, xv.w);
                            }
                        }
                    }
#pragma unroll
                    for (int f = 0; f < F8D_RF; f++)
                    {
#pragma unroll
                        for (int j = 0; j < F8D_NT; j++)
                        {
                            acc[f][j][0] = fmaf(part[f][j][0], s.sx[j][0] * s.sw[f][0], acc[f][j][0]);
                            acc[f][j][1] = fmaf(part[f][j][1], s.sx[j][1] * s.sw[f][0], acc[f][j][1]);
                            acc[f][j][2] = fmaf(part[f][j][2], s.sx[j][0] * s.sw[f][1], acc[f][j][2]);
                            acc[f][j][3] = fmaf(part[f][j][3], s.sx[j][1] * s.sw[f][1], acc[f][j][3]);
                        }
                    }
                    if (i + d + F8D_DEPTH < F8D_NKB)
                    {
                        loadW(s, kb0 + i + d + F8D_DEPTH);
                        loadX(s, kb0 + i + d + F8D_DEPTH);
                    }
                }
            }
        }

#if F8D_KS > 1
        // The K split's partial sums, added in ascending K order by the row tile's first warp.
        if (ks > 0)
        {
#pragma unroll
            for (int f = 0; f < F8D_RF; f++)
            {
#pragma unroll
                for (int j = 0; j < F8D_NT; j++)
                {
#pragma unroll
                    for (int e = 0; e < 4; e++)
                    {
                        red[rt][ks - 1][(f * F8D_NT + j) * 4 + e][lane] = acc[f][j][e];
                    }
                }
            }
        }
        __syncthreads();
        if (ks == 0)
        {
#pragma unroll
            for (int q = 0; q < F8D_KS - 1; q++)
            {
#pragma unroll
                for (int f = 0; f < F8D_RF; f++)
                {
#pragma unroll
                    for (int j = 0; j < F8D_NT; j++)
                    {
#pragma unroll
                        for (int e = 0; e < 4; e++)
                        {
                            acc[f][j][e] += red[rt][q][(f * F8D_NT + j) * 4 + e][lane];
                        }
                    }
                }
            }
        }
#endif

        // Lane outputs per fragment f, token fragment j: (row g, tokens 2 c, 2 c + 1) and (row g + 8, the same tokens).
        if (ks == 0)
        {
            const f8d_u32 n0 = (tile * F8D_RT + rt) * F8D_ROWS;
#pragma unroll
            for (int f = 0; f < F8D_OUT_FRAGS; f++)
            {
#pragma unroll
                for (int j = 0; j < F8D_NT; j++)
                {
#pragma unroll
                    for (int e = 0; e < 4; e++)
                    {
                        const f8d_u32 n = n0 + f * 16 + g + (e >> 1) * 8, t = j * 8 + 2 * c + (e & 1);
                        if (t < p.T)
                        {
#if DUAL
                            p.y[(f8d_u64)t * Y_STRIDE + n] = F8D_EPI2(acc[f][j][e], acc[f + F8D_OUT_FRAGS][j][e]);
#else
                            p.y[(f8d_u64)t * Y_STRIDE + n] = acc[f][j][e];
#endif
                        }
                    }
                }
            }
        }
#if F8D_KS > 1
        // The next tile's partial sums reuse red.
        __syncthreads();
#endif
    }
}
