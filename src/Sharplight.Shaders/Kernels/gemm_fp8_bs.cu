// Block-scaled form of gemm_fp8_tma.cu for the CUDA provider (DeviceFeatures.BlockScaledFp8, compute capability 12.x, compiled for
// the family-specific architecture compute_XYf): y[t, n] = sum_k x[t, k] * W[n, k] on the same operands, layouts, tensor maps, tile
// order, copy warp and barrier ring as gemm_fp8_tma.cu, but every scale must be a power of two (the activation rows of quantize_fp8 /
// the E4M3-writing norms with E4M3_UE8M0, the weights of WeightRepack.Fp8B128GroupedUe8m0). The consumer warps pass the scales'
// exponent fields as UE8M0 scale vectors to the block-scaled MMA (mma.sync ... kind::mxf8f6f4.block_scale.scale_vec::1X: one scale
// per 32 values of K for each row of A and each column of B; ours repeat per 128), which applies them in the tensor core, so the
// products of the whole K range accumulate in one f32 accumulator per output: no fresh accumulator and no multiply-add promotion per
// K block (gemm_fp8_tma.cu: a multiply-add per MMA, plus the products of the scales). The results differ from gemm_fp8_tma.cu's on the same operands
// only in the order and rounding of the f32 sums.
//
// QOUT (DUAL only): instead of f32 outputs, y points to E4M3 activation rows of the outputs as quantize_fp8 writes them with E4M3_UE8M0
// ([QCAP rows of N codes + N / 32 bytes of f32 scales][N / 128][QCAP] dense scales), the next FP8 GEMM's x: the codes and power-of-two
// scales quantize_fp8 would compute from the f32 outputs, bit for bit. A 128-value scale group spans two 64-output tiles, so a block
// walks pairs of tiles (n0, n0 + 64) of one token tile back to back: the first tile's outputs stay in registers, each row's absolute
// maximum is combined in shared memory (two barriers of the computing warps per pair), and the second tile stores both.
//
// Specialization macros: K, N, Y_STRIDE, XCAP, DUAL, F8N_EPI2 (DUAL), QOUT and QCAP. Requires K % 512 == 0, N % 128 == 0, XCAP % 128 == 0 and an
// even Y_STRIDE. Grid: x = persistent blocks (at most the number of tiles), 288 threads, F8N_SHARED bytes of dynamic shared memory.
#ifndef DUAL
#define DUAL 0
#endif
#define F8N_BM 128
#define F8N_BN 128
#define F8N_STAGES 3
#define F8N_WM 2
#define F8N_WN 4
#define F8N_CONSUMERS (F8N_WM * F8N_WN * 32)
#define F8N_THREADS (F8N_CONSUMERS + 32)
#define F8N_TM (F8N_BM / F8N_WM)
#define F8N_TN (F8N_BN / F8N_WN)
#define F8N_FM (F8N_TM / 16)
#define F8N_FN (F8N_TN / 8)
#define F8N_STAGE_BYTES ((F8N_BM + F8N_BN) * 128)
#define F8N_SCALE_BYTES ((F8N_BM + F8N_BN) * 4)
// Three code stages, two scale slots, eight barriers (full[3], empty[3], scales[2]).
#ifndef QOUT
#define QOUT 0
#endif
#if QOUT
// The rows' absolute maxima of a tile pair (128 floats) after the barriers.
#define F8N_SHARED (F8N_STAGES * F8N_STAGE_BYTES + 2 * F8N_SCALE_BYTES + 64 + 512)
// A tile pair covers 128 outputs: the walk's unit.
#define F8N_WALK 128
#define F8N_QRS ((unsigned long long)N + N / 32)
#else
#define F8N_SHARED (F8N_STAGES * F8N_STAGE_BYTES + 2 * F8N_SCALE_BYTES + 64)
#define F8N_WALK F8N_BOUT
#endif
#define F8N_RS ((unsigned long long)K + K / 32)
#define F8N_GROUP_BYTES (128ull * F8N_RS)
#define F8N_TILE_ORDER 16
#if DUAL
#define F8N_BOUT (F8N_BN / 2)
#else
#define F8N_BOUT F8N_BN
#endif

typedef unsigned char f8n_u8;
typedef unsigned int f8n_u32;
typedef unsigned long long f8n_u64;

struct __align__(64) F8nTensorMap
{
    f8n_u64 opaque[16];
};

struct F8nParams
{
    const f8n_u8* x;
    const f8n_u8* w;
#if DUAL
    const f8n_u8* w2;
#endif
    float* y;
    f8n_u32 T;
    F8nTensorMap mx;
    F8nTensorMap mw;
#if DUAL
    F8nTensorMap mw2;
#endif
};

static __device__ __forceinline__ f8n_u32 f8n_smem(const void* p)
{
    return (f8n_u32)__cvta_generic_to_shared(p);
}

static __device__ __forceinline__ void f8n_ldsm4(f8n_u32 addr, f8n_u32& r0, f8n_u32& r1, f8n_u32& r2, f8n_u32& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// c += (A * 2^(sa - 127)) (B * 2^(sb - 127)): the block-scaled E4M3 MMA of sm_120 (kind::mxf8f6f4, one UE8M0 scale per 32 values
// of K for each row of A and each column of B; scale_vec::1X, byte 0 of the scale registers, thread pairs 0 / 1 of each quad).
static __device__ __forceinline__ void f8n_mma(float* c, const f8n_u32* a, f8n_u32 b0, f8n_u32 b1, f8n_u32 sa, f8n_u32 sb)
{
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "r"(sa), "h"((unsigned short)0), "h"((unsigned short)0), "r"(sb), "h"((unsigned short)0), "h"((unsigned short)0));
}

static __device__ __forceinline__ void f8n_mbar_init(f8n_u32 bar, f8n_u32 count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count));
}

static __device__ __forceinline__ void f8n_mbar_expect(f8n_u32 bar, f8n_u32 bytes)
{
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void f8n_mbar_arrive(f8n_u32 bar)
{
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}

static __device__ __forceinline__ void f8n_mbar_wait(f8n_u32 bar, f8n_u32 parity)
{
    asm volatile("{\n .reg .pred p;\n F8N_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra F8N_WAIT_%=;\n}\n"
                 :: "r"(bar), "r"(parity) : "memory");
}

// Destination space of the bulk and tensor copies: the block's own shared memory (PTX 8.6, CUDA 12.8). The cluster form that older
// compilers take makes ptxas check at run time that the address is the block's, with a fallback call that needs local memory.
#if __CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)
#define F8N_DST "shared::cta"
#else
#define F8N_DST "shared::cluster"
#endif

static __device__ __forceinline__ void f8n_tma2(f8n_u32 dst, const F8nTensorMap* map, int c0, int c1, f8n_u32 bar)
{
    asm volatile("cp.async.bulk.tensor.2d." F8N_DST ".global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 :: "r"(dst), "l"((f8n_u64)map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

static __device__ __forceinline__ void f8n_tma3(f8n_u32 dst, const F8nTensorMap* map, int c0, int c1, int c2, f8n_u32 bar)
{
    asm volatile("cp.async.bulk.tensor.3d." F8N_DST ".global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
                 :: "r"(dst), "l"((f8n_u64)map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}

static __device__ __forceinline__ void f8n_bulk(f8n_u32 dst, const void* src, f8n_u32 bytes, f8n_u32 bar)
{
    asm volatile("{\n .reg .u64 g;\n cvta.to.global.u64 g, %1;\n cp.async.bulk." F8N_DST ".global.mbarrier::complete_tx::bytes [%0], [g], %2, [%3];\n}\n"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// Byte offset of 16-byte chunk c of row r in a stage tile of 128-byte rows, as the 128-byte swizzle of the tensor copies places it
// (chunk c ^ (r % 8); the tiles are 1024-byte aligned).
static __device__ __forceinline__ f8n_u32 f8n_swz(f8n_u32 r, f8n_u32 c)
{
    return r * 128u + ((c ^ (r & 7u)) << 4);
}

// Scales of weight row n at K block 0 (the 128 rows of a group are adjacent per block, blocks 512 bytes apart).
static __device__ __forceinline__ const f8n_u8* f8n_wscales(const f8n_u8* w, f8n_u32 n)
{
    return w + (n >> 7) * F8N_GROUP_BYTES + 128ull * K + (n & 127) * 4u;
}

// Token-tile and weight-tile origin of tile id (groups of F8N_TILE_ORDER token tiles walk all weight tiles).
static __device__ __forceinline__ void f8n_tile(f8n_u32 id, f8n_u32 tilesX, f8n_u32& t0, f8n_u32& n0)
{
    const f8n_u32 tilesY = N / F8N_WALK, groupBlocks = F8N_TILE_ORDER * tilesY;
    const f8n_u32 first = id / groupBlocks * F8N_TILE_ORDER, groupRows = min(tilesX - first, (f8n_u32)F8N_TILE_ORDER);
    t0 = (first + id % groupBlocks % groupRows) * F8N_BM;
    n0 = id % groupBlocks / groupRows * F8N_WALK;
}

// Origin of this block's local tile v: tile blockIdx.x + v gridDim.x of the walk, or (QOUT) half v % 2 of tile pair v / 2.
static __device__ __forceinline__ void f8n_local(f8n_u32 v, f8n_u32 tilesX, f8n_u32& t0, f8n_u32& n0)
{
#if QOUT
    f8n_tile(blockIdx.x + (v >> 1) * gridDim.x, tilesX, t0, n0);
    n0 += (v & 1) * F8N_BOUT;
#else
    f8n_tile(blockIdx.x + v * gridDim.x, tilesX, t0, n0);
#endif
}

#if QOUT
// UE8M0 scale of a row group with absolute maximum amax: 2^ceil(log2(amax / 448)), 1 for an all-zero group (e4m3.slang, E4M3_UE8M0).
static __device__ __forceinline__ float f8n_qscale(float amax)
{
    if (!(amax > 0.0f))
    {
        return 1.0f;
    }

    const f8n_u32 bits = __float_as_uint(amax / 448.0f);
    const f8n_u32 e = ((bits >> 23) & 0xFFu) + ((bits & 0x7FFFFFu) != 0 ? 1u : 0u);
    return __uint_as_float(min(max(e, 1u), 254u) << 23);
}

// The E4M3 codes of a and b (rounded to nearest even, saturated at 448), a in the low byte.
static __device__ __forceinline__ unsigned short f8n_e4m3x2(float a, float b)
{
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(b), "f"(a));
    return r;
}
#endif

// B tile row of the first row of fragment j of warp column wn.
static __device__ __forceinline__ f8n_u32 f8n_brow(f8n_u32 wn, f8n_u32 j)
{
#if DUAL
    return (j & 1) * (F8N_BN / 2) + wn * (F8N_TN / 2) + (j >> 1) * 8;
#else
    return wn * F8N_TN + j * 8;
#endif
}

extern "C" __global__ void __launch_bounds__(F8N_THREADS) gemm_fp8_bs(const __grid_constant__ F8nParams p)
{
    extern __shared__ __align__(1024) f8n_u8 smem[];
    const f8n_u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const f8n_u32 tilesX = (p.T + F8N_BM - 1) / F8N_BM, tiles = tilesX * (N / F8N_WALK);
    const f8n_u32 nk = K / 128;
    const f8n_u32 myTiles = (blockIdx.x < tiles ? (tiles - blockIdx.x + gridDim.x - 1) / gridDim.x : 0) * (QOUT ? 2 : 1);
    const f8n_u32 total = myTiles * nk;
    const f8n_u8* sx = p.x + (f8n_u64)XCAP * F8N_RS;
    const f8n_u32 scaleRing = f8n_smem(smem + F8N_STAGES * F8N_STAGE_BYTES);
    // full[s] at bars + 8 s, empty[s] at bars + 24 + 8 s, scales[s] at bars + 48 + 8 s.
    const f8n_u32 bars = scaleRing + 2 * F8N_SCALE_BYTES;
    if (tid == 0)
    {
        for (int s = 0; s < F8N_STAGES; s++)
        {
            f8n_mbar_init(bars + s * 8, 1);
            f8n_mbar_init(bars + 24 + s * 8, F8N_CONSUMERS / 32);
        }
        f8n_mbar_init(bars + 48, 1);
        f8n_mbar_init(bars + 56, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
#if QOUT
    float* rowMax = (float*)(smem + F8N_STAGES * F8N_STAGE_BYTES + 2 * F8N_SCALE_BYTES + 64);
    if (tid < F8N_BM)
    {
        rowMax[tid] = 0.0f;
    }
#endif
    __syncthreads();

    if (warp == F8N_WM * F8N_WN)
    {
        if (lane != 0)
        {
            return;
        }

        // Producer: the codes of iteration m once iteration m - 3 released its stage, with them the scales of iteration m - 1
        // (their slot was last read in iteration m - 3 as well).
        f8n_u32 ct0 = 0, cn0 = 0, codeLt = 0, codeKb = 0, st0 = 0, sn0 = 0, scaleLt = 0, scaleKb = 0;
        for (f8n_u32 m = 0; m <= total; m++)
        {
            if (m >= F8N_STAGES)
            {
                f8n_mbar_wait(bars + 24 + (m % F8N_STAGES) * 8, ((m / F8N_STAGES) - 1) & 1);
            }

            if (m < total)
            {
                if (codeKb == 0)
                {
                    f8n_local(codeLt, tilesX, ct0, cn0);
                }

                const f8n_u32 s = m % F8N_STAGES, dst = f8n_smem(smem + s * F8N_STAGE_BYTES), bar = bars + s * 8;
                f8n_mbar_expect(bar, F8N_STAGE_BYTES);
                f8n_tma2(dst, &p.mx, codeKb * 128, ct0, bar);
#if DUAL
                f8n_tma3(dst + F8N_BM * 128, &p.mw, codeKb * 128, cn0 & 127, cn0 >> 7, bar);
                f8n_tma3(dst + (F8N_BM + F8N_BN / 2) * 128, &p.mw2, codeKb * 128, cn0 & 127, cn0 >> 7, bar);
#else
                f8n_tma3(dst + F8N_BM * 128, &p.mw, codeKb * 128, 0, cn0 >> 7, bar);
#endif
                if (++codeKb == nk)
                {
                    codeKb = 0;
                    codeLt++;
                }
            }

            if (m >= 1)
            {
                if (scaleKb == 0)
                {
                    f8n_local(scaleLt, tilesX, st0, sn0);
                }

                const f8n_u32 slot = (m - 1) & 1, dst = scaleRing + slot * F8N_SCALE_BYTES, bar = bars + 48 + slot * 8;
                f8n_mbar_expect(bar, F8N_SCALE_BYTES);
                f8n_bulk(dst, sx + ((f8n_u64)scaleKb * XCAP + st0) * 4, F8N_BM * 4, bar);
#if DUAL
                f8n_bulk(dst + F8N_BM * 4, f8n_wscales(p.w, sn0) + scaleKb * 512, F8N_BN * 2, bar);
                f8n_bulk(dst + F8N_BM * 4 + F8N_BN * 2, f8n_wscales(p.w2, sn0) + scaleKb * 512, F8N_BN * 2, bar);
#else
                f8n_bulk(dst + F8N_BM * 4, f8n_wscales(p.w, sn0) + scaleKb * 512, F8N_BN * 4, bar);
#endif
                if (++scaleKb == nk)
                {
                    scaleKb = 0;
                    scaleLt++;
                }
            }
        }

        return;
    }

    const f8n_u32 wm = warp / F8N_WN, wn = warp % F8N_WN;
    const f8n_u32 g = lane >> 2, c4 = lane & 3;
    float acc[F8N_FM][F8N_FN][4];
#pragma unroll
    for (int i = 0; i < F8N_FM; i++)
    {
#pragma unroll
        for (int j = 0; j < F8N_FN; j++)
        {
#pragma unroll
            for (int e = 0; e < 4; e++)
            {
                acc[i][j][e] = 0.0f;
            }
        }
    }

#if QOUT
    // The first tile's outputs of the current pair.
    float held[F8N_FM][2][F8N_FN / 2][2];
#endif
    for (f8n_u32 lt = 0; lt < myTiles; lt++)
    {
        f8n_u32 t0, n0;
        f8n_local(lt, tilesX, t0, n0);
        for (f8n_u32 kb = 0; kb < nk; kb++)
        {
            const f8n_u32 it = lt * nk + kb;
            f8n_mbar_wait(bars + 48 + (it & 1) * 8, (it >> 1) & 1);
            const float* xs = (const float*)(smem + F8N_STAGES * F8N_STAGE_BYTES + (it & 1) * F8N_SCALE_BYTES);
            const float* ws = xs + F8N_BM;
            // UE8M0 scales: the exponent fields of the power-of-two f32 scales. A: lanes 4g, 4g + 2 give row g, 4g + 1, 4g + 3 row
            // g + 8 of a fragment; B: every lane of quad g column g.
            f8n_u32 xe[F8N_FM], we[F8N_FN];
#pragma unroll
            for (int i = 0; i < F8N_FM; i++)
            {
                xe[i] = __float_as_uint(xs[wm * F8N_TM + i * 16 + g + (c4 & 1) * 8]) >> 23;
            }
#pragma unroll
            for (int j = 0; j < F8N_FN; j++)
            {
                we[j] = __float_as_uint(ws[f8n_brow(wn, j) + g]) >> 23;
            }

            const f8n_u32 s = it % F8N_STAGES;
            f8n_mbar_wait(bars + s * 8, (it / F8N_STAGES) & 1);
            const f8n_u32 sa = f8n_smem(smem + s * F8N_STAGE_BYTES), sb = sa + F8N_BM * 128;
            f8n_u32 b[4][F8N_FN][2];
#pragma unroll
            for (int kk = 0; kk < 4; kk++)
            {
#pragma unroll
                for (int j = 0; j < F8N_FN; j += 2)
                {
                    f8n_u32 r = f8n_brow(wn, j + (lane >> 4)) + (lane & 7);
                    f8n_ldsm4(sb + f8n_swz(r, kk * 2 + ((lane >> 3) & 1)), b[kk][j][0], b[kk][j][1], b[kk][j + 1][0], b[kk][j + 1][1]);
                }
            }

#pragma unroll
            for (int i = 0; i < F8N_FM; i++)
            {
#pragma unroll
                for (int kk = 0; kk < 4; kk++)
                {
                    f8n_u32 a[4];
                    f8n_u32 r = wm * F8N_TM + i * 16 + (lane & 15);
                    f8n_ldsm4(sa + f8n_swz(r, kk * 2 + (lane >> 4)), a[0], a[1], a[2], a[3]);
#pragma unroll
                    for (int j = 0; j < F8N_FN; j++)
                    {
                        f8n_mma(acc[i][j], a, b[kk][j][0], b[kk][j][1], xe[i], we[j]);
                    }
                }
            }

            // Stage s and scale slot it % 2 are read (the fragments and scales are in registers).
            __syncwarp();
            if (lane == 0)
            {
                f8n_mbar_arrive(bars + 24 + s * 8);
            }
        }

#if QOUT
        // The tile's outputs (half lt % 2 of the pair) and each row's absolute maximum over them, combined across the quad and the
        // warp columns in rowMax.
        float out[F8N_FM][2][F8N_FN / 2][2];
#pragma unroll
        for (int i = 0; i < F8N_FM; i++)
        {
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
                float m = 0.0f;
#pragma unroll
                for (int j = 0; j < F8N_FN; j += 2)
                {
                    out[i][h][j / 2][0] = F8N_EPI2(acc[i][j][h * 2], acc[i][j + 1][h * 2]);
                    out[i][h][j / 2][1] = F8N_EPI2(acc[i][j][h * 2 + 1], acc[i][j + 1][h * 2 + 1]);
                    m = fmaxf(m, fmaxf(fabsf(out[i][h][j / 2][0]), fabsf(out[i][h][j / 2][1])));
                    acc[i][j][h * 2] = acc[i][j][h * 2 + 1] = acc[i][j + 1][h * 2] = acc[i][j + 1][h * 2 + 1] = 0.0f;
                }

                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                if (c4 == 0)
                {
                    atomicMax((unsigned int*)&rowMax[wm * F8N_TM + i * 16 + g + h * 8], __float_as_uint(m));
                }
            }
        }

        if ((lt & 1) == 0)
        {
#pragma unroll
            for (int i = 0; i < F8N_FM; i++)
            {
#pragma unroll
                for (int h = 0; h < 2; h++)
                {
#pragma unroll
                    for (int jj = 0; jj < F8N_FN / 2; jj++)
                    {
                        held[i][h][jj][0] = out[i][h][jj][0];
                        held[i][h][jj][1] = out[i][h][jj][1];
                    }
                }
            }

            continue;
        }

        // Both halves' maxima are in (the first half's atomics precede every warp's second), then read; the next pair's first
        // half cannot reach its atomics before every warp has passed this pair's stores (its K loop needs the stages they free).
        asm volatile("bar.sync 1, %0;\n" :: "r"(F8N_CONSUMERS) : "memory");
        float scale[F8N_FM][2];
#pragma unroll
        for (int i = 0; i < F8N_FM; i++)
        {
            scale[i][0] = f8n_qscale(rowMax[wm * F8N_TM + i * 16 + g]);
            scale[i][1] = f8n_qscale(rowMax[wm * F8N_TM + i * 16 + g + 8]);
        }

        asm volatile("bar.sync 1, %0;\n" :: "r"(F8N_CONSUMERS) : "memory");
        if (tid < F8N_BM)
        {
            rowMax[tid] = 0.0f;
        }

        const f8n_u32 pair0 = n0 - F8N_BOUT;
        f8n_u8* q = (f8n_u8*)p.y;
#pragma unroll
        for (int i = 0; i < F8N_FM; i++)
        {
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
                const f8n_u32 t = t0 + wm * F8N_TM + i * 16 + g + h * 8;
                if (t < p.T)
                {
                    f8n_u8* qr = q + (f8n_u64)t * F8N_QRS;
                    const float inv = 1.0f / scale[i][h];
#pragma unroll
                    for (int jj = 0; jj < F8N_FN / 2; jj++)
                    {
                        const f8n_u32 n = wn * (F8N_TN / 2) + jj * 8 + c4 * 2;
                        *(unsigned short*)(qr + pair0 + n) = f8n_e4m3x2(held[i][h][jj][0] * inv, held[i][h][jj][1] * inv);
                        *(unsigned short*)(qr + n0 + n) = f8n_e4m3x2(out[i][h][jj][0] * inv, out[i][h][jj][1] * inv);
                    }

                    if (wn == 0 && c4 == 0)
                    {
                        *(float*)(qr + N + (pair0 / 128) * 4) = scale[i][h];
                        *(float*)(q + (f8n_u64)QCAP * F8N_QRS + ((f8n_u64)(pair0 / 128) * QCAP + t) * 4) = scale[i][h];
                    }
                }
            }
        }
#else
#pragma unroll
        for (int i = 0; i < F8N_FM; i++)
        {
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
                const f8n_u32 t = t0 + wm * F8N_TM + i * 16 + g + h * 8;
                if (t < p.T)
                {
                    float* yr = p.y + (f8n_u64)t * Y_STRIDE;
#if DUAL
#pragma unroll
                    for (int j = 0; j < F8N_FN; j += 2)
                    {
                        const f8n_u32 n = n0 + wn * (F8N_TN / 2) + (j / 2) * 8 + c4 * 2;
                        *(float2*)(yr + n) = make_float2(F8N_EPI2(acc[i][j][h * 2], acc[i][j + 1][h * 2]),
                            F8N_EPI2(acc[i][j][h * 2 + 1], acc[i][j + 1][h * 2 + 1]));
                    }
#else
#pragma unroll
                    for (int j = 0; j < F8N_FN; j++)
                    {
                        const f8n_u32 n = n0 + wn * F8N_TN + j * 8 + c4 * 2;
                        *(float2*)(yr + n) = make_float2(acc[i][j][h * 2], acc[i][j][h * 2 + 1]);
                    }
#endif
                }

#pragma unroll
                for (int j = 0; j < F8N_FN; j++)
                {
                    acc[i][j][h * 2] = 0.0f;
                    acc[i][j][h * 2 + 1] = 0.0f;
                }
            }
        }
#endif
    }
}
