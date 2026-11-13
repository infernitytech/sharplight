// Expert-indexed matrix product (MatMulId): y[item, n] = sum_k x[row(item), k] * W[ids[item], n, k]
// where item = t * S + s, row(item) = PER_SLOT ? item : t (x: f32 rows with stride X_STRIDE), y dense [T*S, N].
// Specialization macros: WFMT, K, N, E (experts), S (slots), PER_SLOT, X_STRIDE, ROW_BYTES, EXPERT_BYTES, SG, plus
// per kernel: MAXTILES / TILE_M (routing), tile layout macros (grouped GEMMs).
//
// Small batches run a per-item matvec. Larger batches first group the items by expert (moe_route builds a list of
// expert tiles of up to TILE_M items plus the items sorted by expert), then one GEMM tile processes the rows of
// one expert with a single pass over that expert's weights.
//
// Routing scratch (uint): [0] = tile count, [1 + 3 i .. 3 i + 3] = tile i {expert, row offset, rows},
// [ROW_BASE + o] = item index at sorted position o. ROW_BASE = 1 + 3 * MAXTILES.
#include "common.slang"

#define ROW_BASE (1 + 3 * MAXTILES)

float dot8m(float w[8], float4 a, float4 b)
{
    return w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
}

#ifdef KERNEL_MOE_MATVEC
struct MvParams
{
#if HAS_GRID
    uint* grid;
#endif
    float* x;
    uint8_t* w;
    uint* ids;
    float* y;
    uint T;
};

#ifndef MV_ROWS_PER_WG
#define MV_ROWS_PER_WG 4
#endif
#ifndef MV_UNROLL
#define MV_UNROLL 2
#endif

// One subgroup per output row of one (token, slot) item.
[shader("compute")]
[numthreads(SG * MV_ROWS_PER_WG, 1, 1)]
void moe_matvec(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform MvParams p)
{
    GRID_BIND(p);
    uint lane = ltid.x % SG;
    uint row = gid.x * MV_ROWS_PER_WG + ltid.x / SG;
    uint item = gid.y;
    if (row >= N) return;
    uint e = p.ids[item];
    if (e >= E)
    {
        if (lane == 0) p.y[uint64_t(item) * N + row] = 0.0;
        return;
    }
    uint8_t* wb = p.w + uint64_t(e) * EXPERT_BYTES;
    float* xp = p.x + uint64_t(PER_SLOT ? item : item / S) * X_STRIDE;
    float acc = 0;
    [unroll(MV_UNROLL)]
    for (uint g = lane; g < K / 8; g += SG)
    {
        float w[8];
        load8(wb, ROW_BYTES, row, g, w);
        float4* xq = (float4*)(xp + g * 8);
        acc += dot8m(w, xq[0], xq[1]);
    }
    float s = WaveActiveSum(acc);
    if (lane == 0) p.y[uint64_t(item) * N + row] = s;
}
#endif

#ifdef KERNEL_MOE_ROUTE
struct RouteParams
{
    uint* ids;
    uint* scratch;
    uint T;
};

#define RT_THREADS 256
groupshared uint rCnt[E];
groupshared uint rStart[E];
groupshared uint rCur[E];
groupshared uint rTileBase[E];

// Groups the T*S items by expert. One workgroup.
[shader("compute")]
[numthreads(RT_THREADS, 1, 1)]
void moe_route(uint3 ltid : SV_GroupThreadID, uniform RouteParams p)
{
    uint tid = ltid.x;
    uint items = p.T * S;
    for (uint e = tid; e < E; e += RT_THREADS) { rCnt[e] = 0; rCur[e] = 0; }
    GroupMemoryBarrierWithGroupSync();
    for (uint i = tid; i < items; i += RT_THREADS)
    {
        uint e = p.ids[i];
        if (e < E) { uint o; InterlockedAdd(rCnt[e], 1u, o); }
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0)
    {
        uint off = 0, t = 0;
        for (uint e = 0; e < E; e++)
        {
            uint c = rCnt[e];
            rStart[e] = off;
            rTileBase[e] = t;
            t += (c + TILE_M - 1) / TILE_M;
            off += c;
        }
        p.scratch[0] = t;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint e = tid; e < E; e += RT_THREADS)
    {
        uint c = rCnt[e];
        uint nt = (c + TILE_M - 1) / TILE_M;
        for (uint j = 0; j < nt; j++)
        {
            uint o = 1 + 3 * (rTileBase[e] + j);
            p.scratch[o] = e;
            p.scratch[o + 1] = rStart[e] + j * TILE_M;
            p.scratch[o + 2] = min(uint(TILE_M), c - j * TILE_M);
        }
    }
    for (uint i = tid; i < items; i += RT_THREADS)
    {
        uint e = p.ids[i];
        if (e < E)
        {
            uint o;
            InterlockedAdd(rCur[e], 1u, o);
            p.scratch[ROW_BASE + rStart[e] + o] = i;
        }
    }
}
#endif

struct GemmParams
{
#if HAS_GRID
    uint* grid;
#endif
    float* x;
    uint8_t* w;
    uint* scratch;
    float* y;
    uint T;
};

#ifdef KERNEL_MOE_GEMM
// Scalar tiled GEMM: 64 items x 64 rows per workgroup, 256 threads, 4x4 micro tiles.
#define BM 64
#define BN 64
#define BK 32
#define TM 4
#define TN 4
groupshared float sX[BK][BM + 4];
groupshared float sW[BK][BN + 4];
groupshared uint sItem[BM];

[shader("compute")]
[numthreads(256, 1, 1)]
void moe_gemm(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform GemmParams p)
{
    GRID_BIND(p);
    uint tid = ltid.x;
    uint tile = gid.y;
    if (tile >= p.scratch[0]) return;
    uint e = p.scratch[1 + 3 * tile], rowOff = p.scratch[2 + 3 * tile], nrows = p.scratch[3 + 3 * tile];
    uint n0 = gid.x * BN;
    uint8_t* wb = p.w + uint64_t(e) * EXPERT_BYTES;
    if (tid < BM) sItem[tid] = tid < nrows ? p.scratch[ROW_BASE + rowOff + tid] : 0xFFFFFFFFu;
    GroupMemoryBarrierWithGroupSync();
    uint tx = tid % (BN / TN);
    uint ty = tid / (BN / TN);

    float acc[TM][TN];
    [ForceUnroll] for (int i = 0; i < TM; i++) [ForceUnroll] for (int j = 0; j < TN; j++) acc[i][j] = 0;

    for (uint k0 = 0; k0 < K; k0 += BK)
    {
        {
            uint r = tid / 4, gq = tid % 4;
            float w[8];
            if (n0 + r < N && k0 / 8 + gq < K / 8) load8(wb, ROW_BYTES, n0 + r, k0 / 8 + gq, w);
            else { [ForceUnroll] for (int i = 0; i < 8; i++) w[i] = 0; }
            [ForceUnroll] for (int i = 0; i < 8; i++) sW[gq * 8 + i][r] = w[i];
        }
        {
            uint r = tid / 4, gq = tid % 4;
            uint item = sItem[r];
            float4 a = 0, b = 0;
            if (item != 0xFFFFFFFFu && k0 / 8 + gq < K / 8)
            {
                float4* xp = (float4*)(p.x + uint64_t(PER_SLOT ? item : item / S) * X_STRIDE + k0 + gq * 8);
                a = xp[0]; b = xp[1];
            }
            sX[gq * 8 + 0][r] = a.x; sX[gq * 8 + 1][r] = a.y; sX[gq * 8 + 2][r] = a.z; sX[gq * 8 + 3][r] = a.w;
            sX[gq * 8 + 4][r] = b.x; sX[gq * 8 + 5][r] = b.y; sX[gq * 8 + 6][r] = b.z; sX[gq * 8 + 7][r] = b.w;
        }
        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (int kk = 0; kk < BK; kk++)
        {
            float xa[TM], wv[TN];
            [ForceUnroll] for (int i = 0; i < TM; i++) xa[i] = sX[kk][ty * TM + i];
            [ForceUnroll] for (int j = 0; j < TN; j++) wv[j] = sW[kk][tx * TN + j];
            [ForceUnroll] for (int i = 0; i < TM; i++) [ForceUnroll] for (int j = 0; j < TN; j++) acc[i][j] += xa[i] * wv[j];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [ForceUnroll]
    for (int i = 0; i < TM; i++)
    {
        uint item = sItem[ty * TM + i];
        if (item == 0xFFFFFFFFu) continue;
        [ForceUnroll]
        for (int j = 0; j < TN; j++)
        {
            uint n = n0 + tx * TN + j;
            if (n < N) p.y[uint64_t(item) * N + n] = acc[i][j];
        }
    }
}
#endif

#ifdef KERNEL_MOE_GEMM_CM
// Tensor-core grouped GEMM (16x16x16 f16 -> f32). Workgroup = WARPS_M x WARPS_N subgroups, each owning FRAG_M x FRAG_N
// accumulators: a GBM items x GBN rows tile of one expert (GBM = TILE_M of the routing). Shared tiles are double
// buffered with register prefetch (one barrier per K step); results are staged through shared memory and scattered
// to the item rows. Requires K % 32 == 0.
using namespace linalg;
#define GBM (WARPS_M * FRAG_M * 16)
#define GBN (WARPS_N * FRAG_N * 16)
#define GTHREADS (WARPS_M * WARPS_N * 32)
#define GBK 32
#define GQ (GBK / 8)
#define GAS (GBK + 8)
#define GA_PER_T (GBM * GBK / 8 / GTHREADS)
#define GB_PER_T (GBN * GBK / 8 / GTHREADS)
typealias MatA2 = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB2 = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias Acc2 = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

groupshared uint4 gA[2 * GBM * GAS / 8];
groupshared uint4 gB[2 * GBN * GAS / 8];
groupshared float gC[WARPS_M * WARPS_N * 256];
groupshared uint sItem[GBM];

uint4 load8h(uint8_t* w, uint row, uint g)
{
#if WFMT == FMT_Q4_K || WFMT == FMT_Q5_K || WFMT == FMT_Q6_K
    return load8h_kquant(w, ROW_BYTES, row, g);
#else
    float v[8];
    load8(w, ROW_BYTES, row, g, v);
    return uint4(f32tof16(v[0]) | (f32tof16(v[1]) << 16), f32tof16(v[2]) | (f32tof16(v[3]) << 16),
                 f32tof16(v[4]) | (f32tof16(v[5]) << 16), f32tof16(v[6]) | (f32tof16(v[7]) << 16));
#endif
}

[shader("compute")]
[numthreads(GTHREADS, 1, 1)]
void moe_gemm_cm(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform GemmParams p)
{
    GRID_BIND(p);
    uint tid = ltid.x;
    uint sg = tid / 32, lane = tid % 32;
    uint tile = gid.y;
    if (tile >= p.scratch[0]) return;
    uint e = p.scratch[1 + 3 * tile], rowOff = p.scratch[2 + 3 * tile], nrows = p.scratch[3 + 3 * tile];
    uint n0 = gid.x * GBN;
    uint wm = (sg / WARPS_N) * (FRAG_M * 16);
    uint wn = (sg % WARPS_N) * (FRAG_N * 16);
    uint8_t* wbase = p.w + uint64_t(e) * EXPERT_BYTES;
    for (uint r = tid; r < GBM; r += GTHREADS) sItem[r] = r < nrows ? p.scratch[ROW_BASE + rowOff + r] : 0xFFFFFFFFu;
    GroupMemoryBarrierWithGroupSync();

    Acc2 acc[FRAG_M][FRAG_N];
    [ForceUnroll] for (int i = 0; i < FRAG_M; i++) [ForceUnroll] for (int j = 0; j < FRAG_N; j++) acc[i][j] = Acc2(0.0);

    uint4 xs[GA_PER_T];
    uint4 ws[GB_PER_T];

    #define FETCH(k0)                                                                                   \
    [ForceUnroll] for (uint it = 0; it < GA_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / GQ, q = idx % GQ;                                     \
        uint item = sItem[r];                                                                           \
        float4 a = 0, b = 0;                                                                            \
        if (item != 0xFFFFFFFFu)                                                                        \
        {                                                                                               \
            float4* xp = (float4*)(p.x + uint64_t(PER_SLOT ? item : item / S) * X_STRIDE + (k0) + q * 8); \
            a = xp[0]; b = xp[1];                                                                       \
        }                                                                                               \
        xs[it] = uint4(f32tof16(a.x) | (f32tof16(a.y) << 16), f32tof16(a.z) | (f32tof16(a.w) << 16),    \
                       f32tof16(b.x) | (f32tof16(b.y) << 16), f32tof16(b.z) | (f32tof16(b.w) << 16));   \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < GB_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / GQ, q = idx % GQ;                                     \
        ws[it] = 0;                                                                                     \
        if (N % GBN == 0 || n0 + r < N) ws[it] = load8h(wbase, n0 + r, (k0) / 8 + q);                   \
    }
    #define STAGE(buf)                                                                                  \
    [ForceUnroll] for (uint it = 0; it < GA_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / GQ, q = idx % GQ;                                     \
        gA[((buf) * GBM * GAS + r * GAS + q * 8) / 8] = xs[it];                                         \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < GB_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / GQ, q = idx % GQ;                                     \
        gB[((buf) * GBN * GAS + r * GAS + q * 8) / 8] = ws[it];                                         \
    }

    FETCH(0u)
    STAGE(0u)
    GroupMemoryBarrierWithGroupSync();

    uint cur = 0;
    for (uint k0 = 0; k0 < K; k0 += GBK)
    {
        bool more = k0 + GBK < K;
        if (more) { FETCH(k0 + GBK) }
        uint ba = cur * GBM * GAS, bb = cur * GBN * GAS;
        [ForceUnroll]
        for (uint kk = 0; kk < GBK; kk += 16)
        {
            MatB2 b[FRAG_N];
            [ForceUnroll] for (int j = 0; j < FRAG_N; j++) b[j] = MatB2.Load<CoopMatMatrixLayout.ColumnMajor>(gB, (bb + (wn + j * 16) * GAS + kk) / 8, GAS / 8);
            [ForceUnroll]
            for (int i = 0; i < FRAG_M; i++)
            {
                MatA2 a = MatA2.Load<CoopMatMatrixLayout.RowMajor>(gA, (ba + (wm + i * 16) * GAS + kk) / 8, GAS / 8);
                [ForceUnroll] for (int j = 0; j < FRAG_N; j++) acc[i][j] = coopMatMulAdd<float, false>(a, b[j], acc[i][j]);
            }
        }
        if (more) { STAGE(cur ^ 1) }
        GroupMemoryBarrierWithGroupSync();
        cur ^= 1;
    }
    #undef FETCH
    #undef STAGE

    // Each subgroup stages its accumulators through its own shared slice and scatters the rows.
    [ForceUnroll]
    for (int i = 0; i < FRAG_M; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j++)
        {
            acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(gC, sg * 256, 16u);
            GroupMemoryBarrierWithGroupSync();
            [ForceUnroll]
            for (uint q = 0; q < 8; q++)
            {
                uint w = lane + q * 32;
                uint item = sItem[wm + i * 16 + w / 16];
                uint n = n0 + wn + j * 16 + w % 16;
                if (item != 0xFFFFFFFFu && n < N) p.y[uint64_t(item) * N + n] = gC[sg * 256 + w];
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
#endif
