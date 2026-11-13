// y[t, n] = sum_k x[t, k] * W[n, k]      (x: f32 rows with stride X_STRIDE, W: WFMT rows of K values)
// Specialization macros: WFMT, K, N, X_STRIDE, Y_STRIDE, ROW_BYTES, SG
#include "common.slang"

// DUAL: two weight matrices of identical shape/format share x; the outputs are combined per element by
// `float epi2(float a, float b)` (defined by the including generated source, e.g. silu(a) * b) — matvec paths only.
#ifndef DUAL
#define DUAL 0
#endif
// MOE (gemm_cm2 only): expert-grouped GEMM of a MatMulId. W holds E experts of N rows (EXPERT_BYTES apart); y gets one row per
// (token, slot) item. Each workgroup row of the grid is one expert tile of the routing scratch built by moe_route (moe.slang):
// [0] = tile count, [1 + 3i ..] = tile i {expert, row offset, rows}, [1 + 3 MAXTILES + o] = item at sorted position o. Item i
// reads activation row PER_SLOT ? i : i / S.
#ifndef MOE
#define MOE 0
#endif

struct Params
{
#if HAS_GRID
    uint* grid;
#endif
    float* x;
    uint8_t* w;
#if DUAL
    uint8_t* w2;
#endif
#if MOE
    uint* scratch;
#endif
    float* y;
    uint T;
};

float dot8(float w[8], float4 a, float4 b)
{
    return w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
}

// ---------------------------------------------------------------------------------------------
// Matrix-vector path (small T, memory bound). One subgroup per output row; each lane walks groups
// of 8 weights. Up to MV_COLS tokens share every weight load.
#ifndef MV_COLS
#define MV_COLS 8
#endif
#ifndef MV_ROWS_PER_WG
#define MV_ROWS_PER_WG 4
#endif
#ifndef MV_UNROLL
#define MV_UNROLL 1
#endif

[shader("compute")]
[numthreads(SG * MV_ROWS_PER_WG, 1, 1)]
void matvec(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    GRID_BIND(p);
    uint lane = ltid.x % SG;
    uint row = gid.x * MV_ROWS_PER_WG + ltid.x / SG;
    if (row >= N) return;

    float acc[MV_COLS];
    [ForceUnroll] for (int c = 0; c < MV_COLS; c++) acc[c] = 0;

#if DUAL
    float acc2[MV_COLS];
    [ForceUnroll] for (int c = 0; c < MV_COLS; c++) acc2[c] = 0;
#endif
#if WFMT == FMT_Q8_0R && MV_Q8R_WIDE && !DUAL
    // Repacked Q8_0: lane l reads 16 consecutive int8 (one 16-byte load, fully coalesced across the
    // subgroup) per iteration; two lanes share a block scale.
    uint8_t* wr = rowStart(p.w, ROW_BYTES, row);
    uint8_t* qbase = wr + ROW_BYTES / 17;
    [unroll(4)]
    for (uint h = lane; h < K / 16; h += SG)
    {
        float d = h2f(*(uint16_t*)(wr + (h >> 1) * 2));
        uint4 q = *(uint4*)(qbase + uint64_t(h) * 16);
        [ForceUnroll]
        for (int c = 0; c < MV_COLS; c++)
        {
            if (c < p.T)
            {
                float4* xp = (float4*)(p.x + uint64_t(c) * X_STRIDE + h * 16);
                float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                float s = x0.x * float(int(q.x << 24) >> 24) + x0.y * float(int(q.x << 16) >> 24) + x0.z * float(int(q.x << 8) >> 24) + x0.w * float(int(q.x) >> 24)
                        + x1.x * float(int(q.y << 24) >> 24) + x1.y * float(int(q.y << 16) >> 24) + x1.z * float(int(q.y << 8) >> 24) + x1.w * float(int(q.y) >> 24)
                        + x2.x * float(int(q.z << 24) >> 24) + x2.y * float(int(q.z << 16) >> 24) + x2.z * float(int(q.z << 8) >> 24) + x2.w * float(int(q.z) >> 24)
                        + x3.x * float(int(q.w << 24) >> 24) + x3.y * float(int(q.w << 16) >> 24) + x3.z * float(int(q.w << 8) >> 24) + x3.w * float(int(q.w) >> 24);
                acc[c] += d * s;
            }
        }
    }
#else
    [unroll(MV_UNROLL)]
    for (uint g = lane; g < K / 8; g += SG)
    {
        float w[8];
        load8(p.w, ROW_BYTES, row, g, w);
#if DUAL
        float w2[8];
        load8(p.w2, ROW_BYTES, row, g, w2);
#endif
        [ForceUnroll]
        for (int c = 0; c < MV_COLS; c++)
        {
            if (c < p.T)
            {
                float4* xp = (float4*)(p.x + uint64_t(c) * X_STRIDE + g * 8);
                float4 a = xp[0], b = xp[1];
                acc[c] += dot8(w, a, b);
#if DUAL
                acc2[c] += dot8(w2, a, b);
#endif
            }
        }
    }
#endif
    [ForceUnroll]
    for (int c = 0; c < MV_COLS; c++)
    {
        if (c < p.T)
        {
            float s = WaveActiveSum(acc[c]);
#if DUAL
            s = epi2(s, WaveActiveSum(acc2[c]));
#endif
            if (lane == 0) p.y[uint64_t(c) * Y_STRIDE + row] = s;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Split-K matrix-vector path for small N: one workgroup (8 subgroups) per output row, K spread over
// all 256 threads, partial sums reduced through shared memory. Keeps the GPU busy when N/4 rows
// would give too few workgroups.
#ifdef KERNEL_MATVEC_SK
#ifndef MV_SK_SG
#define MV_SK_SG 8
#endif
groupshared float skPart[MV_COLS][MV_SK_SG];
#if DUAL
groupshared float skPart2[MV_COLS][MV_SK_SG];
#endif

[shader("compute")]
[numthreads(SG * MV_SK_SG, 1, 1)]
void matvec_sk(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    GRID_BIND(p);
    uint tid = ltid.x, lane = tid % SG, sg = tid / SG;
    uint row = gid.x;
    float acc[MV_COLS];
    [ForceUnroll] for (int c = 0; c < MV_COLS; c++) acc[c] = 0;
#if DUAL
    float acc2[MV_COLS];
    [ForceUnroll] for (int c = 0; c < MV_COLS; c++) acc2[c] = 0;
#endif
    [unroll(MV_UNROLL)]
    for (uint g = tid; g < K / 8; g += SG * MV_SK_SG)
    {
        float w[8];
        load8(p.w, ROW_BYTES, row, g, w);
#if DUAL
        float w2[8];
        load8(p.w2, ROW_BYTES, row, g, w2);
#endif
        [ForceUnroll]
        for (int c = 0; c < MV_COLS; c++)
        {
            if (c < p.T)
            {
                float4* xp = (float4*)(p.x + uint64_t(c) * X_STRIDE + g * 8);
                float4 a = xp[0], b = xp[1];
                acc[c] += dot8(w, a, b);
#if DUAL
                acc2[c] += dot8(w2, a, b);
#endif
            }
        }
    }
    [ForceUnroll]
    for (int c = 0; c < MV_COLS; c++)
    {
        float s = WaveActiveSum(acc[c]);
        if (lane == 0) skPart[c][sg] = s;
#if DUAL
        float s2 = WaveActiveSum(acc2[c]);
        if (lane == 0) skPart2[c][sg] = s2;
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid < MV_COLS && tid < p.T)
    {
        float s = 0;
        [ForceUnroll] for (int i = 0; i < MV_SK_SG; i++) s += skPart[tid][i];
#if DUAL
        float s2 = 0;
        [ForceUnroll] for (int i = 0; i < MV_SK_SG; i++) s2 += skPart2[tid][i];
        s = epi2(s, s2);
#endif
        p.y[uint64_t(tid) * Y_STRIDE + row] = s;
    }
}
#endif

// ---------------------------------------------------------------------------------------------
// Tiled GEMM path (prefill). Workgroup computes a BM(tokens) x BN(rows) tile, 256 threads,
// each thread a TM x TN micro tile. K is walked in steps of BK=32 (dequantized into shared memory).
// GEMM_SPLITK > 1: workgroup z sums only its share of K into partials [GEMM_SPLITK][T][N] (see splitk_reduce).
// GEMM_BM: tokens per tile, 64 or 16 (few tokens: less padding work, more workgroups).
#ifndef GEMM_SPLITK
#define GEMM_SPLITK 1
#endif
#ifndef GEMM_BM
#define GEMM_BM 64
#endif
#define BM GEMM_BM
#define BN 64
#define BK 32
#define TM (BM / 16)
#define TN 4

groupshared float sX[BK][BM + 4];
groupshared float sW[BK][BN + 4];

[shader("compute")]
[numthreads(256, 1, 1)]
void gemm(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    GRID_BIND(p);
    uint tid = ltid.x;
    uint n0 = gid.x * BN;
    uint t0 = gid.y * BM;
    uint tx = tid % (BN / TN);   // 16 threads along N
    uint ty = tid / (BN / TN);   // 16 threads along T

    float acc[TM][TN];
    [ForceUnroll] for (int i = 0; i < TM; i++) [ForceUnroll] for (int j = 0; j < TN; j++) acc[i][j] = 0;

#if GEMM_SPLITK > 1
    const uint kShare = ((K + BK - 1) / BK + GEMM_SPLITK - 1) / GEMM_SPLITK * BK;
    uint kBegin = gid.z * kShare;
    uint kEnd = min(uint(K), kBegin + kShare);
    float* yb = p.y + uint64_t(gid.z) * p.T * N;
    const uint yStride = N;
#else
    uint kBegin = 0;
    uint kEnd = K;
    float* yb = p.y;
    const uint yStride = Y_STRIDE;
#endif
    for (uint k0 = kBegin; k0 < kEnd; k0 += BK)
    {
        // W tile: BN rows x BK values = 64 rows * 4 groups of 8 = 256 loads (one per thread).
        {
            uint r = tid / 4, gq = tid % 4;
            float w[8];
            if (n0 + r < N && k0 / 8 + gq < K / 8) load8(p.w, ROW_BYTES, n0 + r, k0 / 8 + gq, w);
            else { [ForceUnroll] for (int i = 0; i < 8; i++) w[i] = 0; }
            [ForceUnroll] for (int i = 0; i < 8; i++) sW[gq * 8 + i][r] = w[i];
        }
        // X tile: BM tokens x BK values, 8 per thread (BM = 64: every thread; BM = 16: the first 64).
        if (tid < BM * 4)
        {
            uint r = tid / 4, gq = tid % 4;
            uint t = t0 + r;
            float4 a = 0, b = 0;
            if (t < p.T && k0 / 8 + gq < K / 8)
            {
                float4* xp = (float4*)(p.x + uint64_t(t) * X_STRIDE + k0 + gq * 8);
                a = xp[0]; b = xp[1];
            }
            sX[gq * 8 + 0][r] = a.x; sX[gq * 8 + 1][r] = a.y; sX[gq * 8 + 2][r] = a.z; sX[gq * 8 + 3][r] = a.w;
            sX[gq * 8 + 4][r] = b.x; sX[gq * 8 + 5][r] = b.y; sX[gq * 8 + 6][r] = b.z; sX[gq * 8 + 7][r] = b.w;
        }
        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (int kk = 0; kk < BK; kk++)
        {
            float xa[TM], wb[TN];
            [ForceUnroll] for (int i = 0; i < TM; i++) xa[i] = sX[kk][ty * TM + i];
            [ForceUnroll] for (int j = 0; j < TN; j++) wb[j] = sW[kk][tx * TN + j];
            [ForceUnroll] for (int i = 0; i < TM; i++) [ForceUnroll] for (int j = 0; j < TN; j++) acc[i][j] += xa[i] * wb[j];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [ForceUnroll]
    for (int i = 0; i < TM; i++)
    {
        uint t = t0 + ty * TM + i;
        if (t >= p.T) continue;
        [ForceUnroll]
        for (int j = 0; j < TN; j++)
        {
            uint n = n0 + tx * TN + j;
            if (n < N) yb[uint64_t(t) * yStride + n] = acc[i][j];
        }
    }
}

#ifdef KERNEL_GEMM_CM
// ---------------------------------------------------------------------------------------------
// Tensor-core GEMM (VK_KHR_cooperative_matrix, 16x16x16 f16 x f16 + f32). Workgroup = 8 subgroups
// computing a CBM(tokens) x CBN(rows) tile; subgroup (sm, sn) owns a 32x32 block = 2x2 accumulators.
// Per K step of CBK, x (f32) and W (any WFMT, dequantized via load8) are staged into shared memory
// as f16. Requires K % CBK == 0; partial N / T tiles are handled (zero-filled loads, staged stores).
using namespace linalg;
#define CBM 64
#define CBN 128
#define CBK 32
#define SAS (CBK + 8)
typealias MatA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias Acc = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

groupshared half cA[CBM * SAS];
groupshared half cB[CBN * SAS];
groupshared float cC[8 * 256];

[shader("compute")]
[numthreads(256, 1, 1)]
void gemm_cm(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    GRID_BIND(p);
    uint tid = ltid.x;
    uint sg = tid / 32;
    uint n0 = gid.x * CBN;
    uint t0 = gid.y * CBM;
    uint wm = (sg / 4) * 32;
    uint wn = (sg % 4) * 32;

    Acc acc00 = Acc(0.0), acc01 = Acc(0.0), acc10 = Acc(0.0), acc11 = Acc(0.0);

    for (uint k0 = 0; k0 < K; k0 += CBK)
    {
        // x tile: 64 tokens x 32 = 256 threads x 8 values.
        {
            uint r = tid / 4, q = tid % 4;
            uint t = t0 + r;
            float4 a = 0, b = 0;
            if (t < p.T)
            {
                float4* xp = (float4*)(p.x + uint64_t(t) * X_STRIDE + k0 + q * 8);
                a = xp[0]; b = xp[1];
            }
            uint o = r * SAS + q * 8;
            cA[o + 0] = half(a.x); cA[o + 1] = half(a.y); cA[o + 2] = half(a.z); cA[o + 3] = half(a.w);
            cA[o + 4] = half(b.x); cA[o + 5] = half(b.y); cA[o + 6] = half(b.z); cA[o + 7] = half(b.w);
        }
        // W tile: 128 rows x 32 = 512 groups of 8, two per thread.
        [ForceUnroll]
        for (uint it = 0; it < 2; it++)
        {
            uint idx = tid + it * 256;
            uint r = idx / 4, q = idx % 4;
            float w[8];
            if (N % CBN == 0 || n0 + r < N) load8(p.w, ROW_BYTES, n0 + r, k0 / 8 + q, w);
            else { [ForceUnroll] for (int i = 0; i < 8; i++) w[i] = 0; }
            uint o = r * SAS + q * 8;
            [ForceUnroll] for (int i = 0; i < 8; i++) cB[o + i] = half(w[i]);
        }
        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (uint kk = 0; kk < CBK; kk += 16)
        {
            MatA a0 = MatA.Load<CoopMatMatrixLayout.RowMajor>(cA, wm * SAS + kk, SAS);
            MatA a1 = MatA.Load<CoopMatMatrixLayout.RowMajor>(cA, (wm + 16) * SAS + kk, SAS);
            MatB b0 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(cB, wn * SAS + kk, SAS);
            MatB b1 = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(cB, (wn + 16) * SAS + kk, SAS);
            acc00 = coopMatMulAdd<float, false>(a0, b0, acc00);
            acc01 = coopMatMulAdd<float, false>(a0, b1, acc01);
            acc10 = coopMatMulAdd<float, false>(a1, b0, acc10);
            acc11 = coopMatMulAdd<float, false>(a1, b1, acc11);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (t0 + CBM <= p.T && (N % CBN == 0 || n0 + CBN <= N))
    {
        float* y = p.y + uint64_t(t0 + wm) * Y_STRIDE + n0 + wn;
        acc00.Store<CoopMatMatrixLayout.RowMajor>(y, 0u, Y_STRIDE);
        acc01.Store<CoopMatMatrixLayout.RowMajor>(y + 16, 0u, Y_STRIDE);
        acc10.Store<CoopMatMatrixLayout.RowMajor>(y + 16 * Y_STRIDE, 0u, Y_STRIDE);
        acc11.Store<CoopMatMatrixLayout.RowMajor>(y + 16 * Y_STRIDE + 16, 0u, Y_STRIDE);
        return;
    }
    // Partial token tile: stage each accumulator through shared memory and store guarded.
    [ForceUnroll]
    for (uint f = 0; f < 4; f++)
    {
        Acc c = f == 0 ? acc00 : f == 1 ? acc01 : f == 2 ? acc10 : acc11;
        c.Store<CoopMatMatrixLayout.RowMajor>(cC, sg * 256, 16u);
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint e = 0; e < 8; e++)
        {
            uint idx = tid + e * 256;
            uint s = idx / 256, w = idx % 256;
            uint t = t0 + (s / 4) * 32 + (f / 2) * 16 + w / 16;
            uint n = n0 + (s % 4) * 32 + (f % 2) * 16 + w % 16;
            if (t < p.T && n < N) p.y[uint64_t(t) * Y_STRIDE + n] = cC[idx];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
#endif

#ifdef KERNEL_GEMM_CM2
// ---------------------------------------------------------------------------------------------
// Large tensor-core GEMM. Workgroup = WARPS_M x WARPS_N subgroups, each owning FRAG_M x FRAG_N 16x16
// accumulators, i.e. a (WARPS_M*FRAG_M*16) tokens x (WARPS_N*FRAG_N*16) rows tile. Shared tiles are
// double buffered; the next K block is prefetched into registers while the current one is multiplied,
// so there is one barrier per K step. Requires K % 32 == 0.
using namespace linalg;
#ifndef WARPS_M
#define WARPS_M 2
#endif
#ifndef WARPS_N
#define WARPS_N 2
#endif
#ifndef FRAG_M
#define FRAG_M 4
#endif
#ifndef FRAG_N
#define FRAG_N 4
#endif
#define GTHREADS (WARPS_M * WARPS_N * 32)
#define GBM (WARPS_M * FRAG_M * 16)
#define GBN (WARPS_N * FRAG_N * 16)
// GBK = 32: double-buffered shared tiles, one barrier per step. GBK = 64: single buffer (fits 48 KB) with twice the
// work between register prefetches, which hides more global-load latency; two barriers per step.
#ifndef GBK
#define GBK 32
#endif
#if GBK == 32
#define GNBUF 2
#else
#define GNBUF 1
#endif
#define GQ (GBK / 8)
#define GAS (GBK + 8)
#define GA_PER_T (GBM * GBK / 8 / GTHREADS)
#define GB_PER_T (GBN * GBK / 16 / GTHREADS)   // weights are fetched 16 per item
typealias MatA2 = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB2 = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias Acc2 = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

// Tiles hold packed halves, 8 per uint4, so staging is one 16-byte store per 8 values (offsets/strides in uint4s).
groupshared uint4 gA[GNBUF * GBM * GAS / 8];
groupshared uint4 gB[GNBUF * GBN * GAS / 8];

uint q8x4_h2(uint u, half2 d, bool hi)
{
    // 2 of the 4 int8 in u (bytes 0,1 or 2,3) -> half2 of q * d
    u ^= 0x80808080u;
    uint v = hi ? (((u >> 16) & 0xFFu) | ((u >> 8) & 0xFF0000u)) : ((u & 0xFFu) | ((u << 8) & 0xFF0000u));
    return bit_cast<uint>((bit_cast<half2>(v | 0x64006400u) - half2(1152.0h, 1152.0h)) * d);
}

// 2 of the 4 E2M1 codes in u (bytes 0,1 or 2,3) -> half2 of value * s1 * s2.
uint fp4_h2(uint u, bool hi, half2 s1, half2 s2)
{
    uint v = hi ? (((u >> 16) & 0xFFu) | ((u >> 8) & 0xFF0000u)) : ((u & 0xFFu) | ((u << 8) & 0xFF0000u));
    return bit_cast<uint>(bit_cast<half2>(((v & 0x00070007u) << 9) | ((v & 0x00080008u) << 12)) * s1 * s2);
}

// 8 consecutive weights of row `row` from K offset 8g as packed halves.
uint4 load8h(uint8_t* w, uint row, uint g);

// 16 consecutive weights of row `row` from K offset 16g16 as packed halves (lo: first 8).
void load16h(uint8_t* w, uint row, uint g16, out uint4 lo, out uint4 hi)
{
#if WFMT == FMT_Q8_0R
    uint8_t* r = rowStart(w, ROW_BYTES, row);
    uint d16 = uint(*(uint16_t*)(r + (g16 >> 1) * 2));
    half2 d = bit_cast<half2>(d16 * 0x10001u);
    uint4 q = *(uint4*)(r + ROW_BYTES / 17 + uint64_t(g16) * 16);
    lo = uint4(q8x4_h2(q.x, d, false), q8x4_h2(q.x, d, true), q8x4_h2(q.y, d, false), q8x4_h2(q.y, d, true));
    hi = uint4(q8x4_h2(q.z, d, false), q8x4_h2(q.z, d, true), q8x4_h2(q.w, d, false), q8x4_h2(q.w, d, true));
#elif WFMT == FMT_Q4_K || WFMT == FMT_Q5_K
    // One 16-byte load of the 16 codes (144 / 176-byte blocks: 16-byte aligned; the offset is summed before it is added
    // to the pointer, as pointer minus integer miscompiles), then packed-half math: (1024 + q) - 1024
    // is exact, q * scale - min rounds about like the f32 value converted to half (plus the rounding of scale and min).
#if WFMT == FMT_Q4_K
#define KQ_BLOCK 144
#else
#define KQ_BLOCK 176
#endif
    uint8_t* blk = rowStart(w, ROW_BYTES, row) + uint64_t(g16 >> 4) * KQ_BLOCK;
    uint idx = (g16 & 15) * 16, j = idx >> 6, upper = (idx >> 5) & 1, l0 = idx & 31;
    uint4 q = (*(uint4*)(blk + (KQ_BLOCK - 128 + 32 * j + l0)) >> (4 * upper)) & 0x0F0F0F0Fu;
#if WFMT == FMT_Q5_K
    q |= ((*(uint4*)(blk + 16 + l0) >> (2 * j + upper)) & 0x01010101u) << 4;
#endif
#undef KQ_BLOCK
    float sc, mn;
    scale_min_k4(blk + 4, 2 * j + upper, sc, mn);
    uint dm = *(uint*)blk;
    half2 a = half2(half(f16tof32(dm & 0xFFFF) * sc));
    half2 nb = half2(half(-f16tof32(dm >> 16) * mn));
    lo = uint4(codes_h2(q.x, false, 1024.0h, a, nb), codes_h2(q.x, true, 1024.0h, a, nb), codes_h2(q.y, false, 1024.0h, a, nb), codes_h2(q.y, true, 1024.0h, a, nb));
    hi = uint4(codes_h2(q.z, false, 1024.0h, a, nb), codes_h2(q.z, true, 1024.0h, a, nb), codes_h2(q.w, false, 1024.0h, a, nb), codes_h2(q.w, true, 1024.0h, a, nb));
#elif WFMT == FMT_Q6_K
    // 6-bit codes from 16 low-nibble and 16 high-bit bytes (210-byte blocks: 2-byte aligned, so u16 loads);
    // (1024 + q) - 1056 = q - 32 is exact, one rounding for the scaled value besides the scale's own.
    uint8_t* blk = rowStart(w, ROW_BYTES, row) + uint64_t(g16 >> 4) * 210;
    uint idx = (g16 & 15) * 16, n = idx >> 7, qd = (idx & 127) >> 5, l0 = idx & 31;
    uint4 ql = load16_u16(blk + n * 64 + (qd & 1) * 32 + l0);
    uint4 qh = load16_u16(blk + 128 + n * 32 + l0);
    uint4 q = ((ql >> ((qd >> 1) * 4)) & 0x0F0F0F0Fu) | (((qh >> (2 * qd)) & 0x03030303u) << 4);
    float d = h2f(*(uint16_t*)(blk + 208));
    half2 s = half2(half(d * float(int(uint(blk[192 + n * 8 + (l0 >> 4) + 2 * qd]) << 24) >> 24)));
    half2 z = half2(0.0h, 0.0h);
    lo = uint4(codes_h2(q.x, false, 1056.0h, s, z), codes_h2(q.x, true, 1056.0h, s, z), codes_h2(q.y, false, 1056.0h, s, z), codes_h2(q.y, true, 1056.0h, s, z));
    hi = uint4(codes_h2(q.z, false, 1056.0h, s, z), codes_h2(q.z, true, 1056.0h, s, z), codes_h2(q.w, false, 1056.0h, s, z), codes_h2(q.w, true, 1056.0h, s, z));
#elif WFMT == FMT_Q8_0
    // 34-byte blocks { f16 d; i8 qs[32] }, 2-byte aligned: word loads, then the int8 -> half trick of Q8_0R.
    uint64_t at = uint64_t(rowStart(w, ROW_BYTES, row)) + uint64_t(g16 >> 1) * 34;
    half2 d = bit_cast<half2>(uint(*(uint16_t*)at) * 0x10001u);
    uint4 q = load16_any(at + (2 + 16 * (g16 & 1)));
    lo = uint4(q8x4_h2(q.x, d, false), q8x4_h2(q.x, d, true), q8x4_h2(q.y, d, false), q8x4_h2(q.y, d, true));
    hi = uint4(q8x4_h2(q.z, d, false), q8x4_h2(q.z, d, true), q8x4_h2(q.w, d, false), q8x4_h2(q.w, d, true));
#elif WFMT == FMT_Q4_0 || WFMT == FMT_Q5_0
    // Q4_0 { f16 d; u8 qs[16] } (18 bytes), Q5_0 { f16 d; u32 qh; u8 qs[16] } (22 bytes): values 0..15 are the low nibbles
    // of qs, 16..31 the high ones (Q5_0: bit 4 of value i is bit i of qh). (1024 + q) - offset is exact, * d rounds once.
#if WFMT == FMT_Q4_0
    uint64_t at = uint64_t(rowStart(w, ROW_BYTES, row)) + uint64_t(g16 >> 1) * 18;
    uint4 q = (load16_any(at + 2) >> (4 * (g16 & 1))) & 0x0F0F0F0Fu;
    half offset = 1032.0h;
#else
    uint64_t at = uint64_t(rowStart(w, ROW_BYTES, row)) + uint64_t(g16 >> 1) * 22;
    uint4 q = (load16_any(at + 6) >> (4 * (g16 & 1))) & 0x0F0F0F0Fu;
    uint qh = uint(*(uint16_t*)(at + (2 + 2 * (g16 & 1))));
    // Bit j of each nibble group goes to bit 4 of byte j (the four shifted copies of the nibble do not overlap).
    q |= ((uint4(qh, qh >> 4, qh >> 8, qh >> 12) & 0xFu) * 0x00204081u & 0x01010101u) << 4;
    half offset = 1040.0h;
#endif
    half2 d = bit_cast<half2>(uint(*(uint16_t*)at) * 0x10001u);
    half2 z = half2(0.0h, 0.0h);
    lo = uint4(codes_h2(q.x, false, offset, d, z), codes_h2(q.x, true, offset, d, z), codes_h2(q.y, false, offset, d, z), codes_h2(q.y, true, offset, d, z));
    hi = uint4(codes_h2(q.z, false, offset, d, z), codes_h2(q.z, true, offset, d, z), codes_h2(q.w, false, offset, d, z), codes_h2(q.w, true, offset, d, z));
#elif WFMT == FMT_MXFP4
    // 16 E2M1 codes (the low or high nibbles of one 17-byte block). A code's
    // 3 magnitude bits at half bits 9..11 and its sign at bit 15 make the half code * 2^-14 (subnormals included), so
    // one multiply by 2^14 * the block scale (a power of two, split in two factors when it exceeds half's range) rounds once.
    uint64_t at = uint64_t(rowStart(w, ROW_BYTES, row)) + uint64_t(g16 >> 1) * 17;
    uint4 q = (load16_any(at + 1) >> (4 * (g16 & 1))) & 0x0F0F0F0Fu;
    float s = e8m0_half(uint(*(uint8_t*)at)) * 32768.0;   // e8m0_half pairs with the doubled codes 0..12
    half2 s1 = half2(half(min(s, 32768.0)));
    half2 s2 = half2(half(max(s / 32768.0, 1.0)));
    lo = uint4(fp4_h2(q.x, false, s1, s2), fp4_h2(q.x, true, s1, s2), fp4_h2(q.y, false, s1, s2), fp4_h2(q.y, true, s1, s2));
    hi = uint4(fp4_h2(q.z, false, s1, s2), fp4_h2(q.z, true, s1, s2), fp4_h2(q.w, false, s1, s2), fp4_h2(q.w, true, s1, s2));
#elif defined(WFMT_FP8)
    load16h_fp8(w, ROW_BYTES, row, g16, lo, hi);
#else
    lo = load8h(w, row, g16 * 2);
    hi = load8h(w, row, g16 * 2 + 1);
#endif
}

uint4 load8h(uint8_t* w, uint row, uint g)
{
#if WFMT == FMT_Q8_0R
    // int8 -> f16 via the 0x6400 exponent trick (1024 + (q + 128)), then (h - 1152) * d: both steps exact but the
    // last, which rounds once, exactly like converting the f32 product d * q to half.
    uint8_t* r = rowStart(w, ROW_BYTES, row);
    uint d16 = uint(*(uint16_t*)(r + (g >> 2) * 2));
    half2 d = bit_cast<half2>(d16 * 0x10001u);
    uint2 q = *(uint2*)(r + ROW_BYTES / 17 + uint64_t(g) * 8);
    half2 off = half2(1152.0h, 1152.0h);
    uint4 o;
    uint u = q.x ^ 0x80808080u;
    o.x = bit_cast<uint>((bit_cast<half2>((u & 0xFFu) | ((u << 8) & 0xFF0000u) | 0x64006400u) - off) * d);
    o.y = bit_cast<uint>((bit_cast<half2>(((u >> 16) & 0xFFu) | ((u >> 8) & 0xFF0000u) | 0x64006400u) - off) * d);
    u = q.y ^ 0x80808080u;
    o.z = bit_cast<uint>((bit_cast<half2>((u & 0xFFu) | ((u << 8) & 0xFF0000u) | 0x64006400u) - off) * d);
    o.w = bit_cast<uint>((bit_cast<half2>(((u >> 16) & 0xFFu) | ((u >> 8) & 0xFF0000u) | 0x64006400u) - off) * d);
    return o;
#elif defined(WFMT_FP8)
    return load8h_fp8(w, ROW_BYTES, row, g);
#else
    float v[8];
    load8(w, ROW_BYTES, row, g, v);
    return uint4(f32tof16(v[0]) | (f32tof16(v[1]) << 16), f32tof16(v[2]) | (f32tof16(v[3]) << 16),
                 f32tof16(v[4]) | (f32tof16(v[5]) << 16), f32tof16(v[6]) | (f32tof16(v[7]) << 16));
#endif
}
groupshared float gC[WARPS_M * WARPS_N * 256];
#if MOE
groupshared uint gItem[GBM];
#endif

[shader("compute")]
[numthreads(GTHREADS, 1, 1)]
void gemm_cm2(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    GRID_BIND(p);
    uint tid = ltid.x;
    uint sg = tid / 32, lane = tid % 32;
    uint n0 = gid.x * GBN;
#if MOE
    // The tile's expert and items; t indexes the tile's rows (0 .. rows - 1) instead of tokens.
    uint tile = gid.y;
    if (tile >= p.scratch[0]) return;
    uint expert = p.scratch[1 + 3 * tile], rowOff = p.scratch[2 + 3 * tile];
    uint tileRows = p.scratch[3 + 3 * tile];
    uint8_t* wbase = rowStart(p.w, EXPERT_BYTES, expert);
    for (uint r = tid; r < GBM; r += GTHREADS) gItem[r] = r < tileRows ? p.scratch[1 + 3 * MAXTILES + rowOff + r] : 0xFFFFFFFFu;
    GroupMemoryBarrierWithGroupSync();
    uint t0 = 0;
    #define ROW_OF(t) (PER_SLOT ? gItem[t] : gItem[t] / S)
    #define ROWS_LIVE tileRows
#else
    uint t0 = gid.y * GBM;
    uint8_t* wbase = p.w;
    #define ROW_OF(t) (t)
    #define ROWS_LIVE p.T
#endif
    uint wm = (sg / WARPS_N) * (FRAG_M * 16);
    uint wn = (sg % WARPS_N) * (FRAG_N * 16);
#ifndef SPLITK
#define SPLITK 1
#endif
    // SPLITK > 1: gid.z owns a K range and writes partial sums to slice gid.z of p.y ([SPLITK][T][N]).
#if SPLITK > 1
    const uint KS = ((K / GBK + SPLITK - 1) / SPLITK) * GBK;
    uint kBeg = gid.z * KS, kEnd = min(uint(K), kBeg + KS);
    float* yb = p.y + uint64_t(gid.z) * p.T * N;
#else
    uint kBeg = 0, kEnd = K;
    float* yb = p.y;
#endif

    Acc2 acc[FRAG_M][FRAG_N];
    [ForceUnroll] for (int i = 0; i < FRAG_M; i++) [ForceUnroll] for (int j = 0; j < FRAG_N; j++) acc[i][j] = Acc2(0.0);

    uint4 xs[GA_PER_T];
    uint4 ws[GB_PER_T][2];

    // X_F16: activations were converted to dense f16 rows beforehand (to_f16), halving tile load traffic.
#if X_F16
    #define GEMM_LOAD_X(t, k, dst)                                                                      \
        { dst = 0; if ((t) < ROWS_LIVE) dst = ((uint4*)((uint16_t*)p.x + uint64_t(ROW_OF(t)) * X_STRIDE + (k)))[0]; }
#else
    #define GEMM_LOAD_X(t, k, dst)                                                                      \
        { float4 a = 0, b = 0;                                                                          \
          if ((t) < ROWS_LIVE) { float4* xp = (float4*)(p.x + uint64_t(ROW_OF(t)) * X_STRIDE + (k)); a = xp[0]; b = xp[1]; } \
          dst = uint4(f32tof16(a.x) | (f32tof16(a.y) << 16), f32tof16(a.z) | (f32tof16(a.w) << 16),       \
                      f32tof16(b.x) | (f32tof16(b.y) << 16), f32tof16(b.z) | (f32tof16(b.w) << 16)); }
#endif
    #define FETCH(k0)                                                                                   \
    [ForceUnroll] for (uint it = 0; it < GA_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / GQ, q = idx % GQ;                                       \
        uint t = t0 + r;                                                                                \
        GEMM_LOAD_X(t, (k0) + q * 8, xs[it])                                                            \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < GB_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / (GBK / 16), q = idx % (GBK / 16);                     \
        ws[it][0] = 0; ws[it][1] = 0;                                                                   \
        if (N % GBN == 0 || n0 + r < N) load16h(wbase, n0 + r, (k0) / 16 + q, ws[it][0], ws[it][1]);      \
    }
    #define STAGE(buf)                                                                                  \
    [ForceUnroll] for (uint it = 0; it < GA_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / GQ, q = idx % GQ;                                       \
        gA[((buf) * GBM * GAS + r * GAS + q * 8) / 8] = xs[it];                                         \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < GB_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / (GBK / 16), q = idx % (GBK / 16);                     \
        uint ob = ((buf) * GBN * GAS + r * GAS + q * 16) / 8;                                           \
        gB[ob] = ws[it][0]; gB[ob + 1] = ws[it][1];                                                     \
    }

    // A split-K slice may be empty (K steps not divisible by SPLITK): never prefetch at or past K.
    if (kBeg < kEnd)
    {
        FETCH(kBeg)
        STAGE(0u)
    }
    GroupMemoryBarrierWithGroupSync();

    uint cur = 0;
    for (uint k0 = kBeg; k0 < kEnd; k0 += GBK)
    {
        bool more = k0 + GBK < kEnd;
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
#if GNBUF == 2
        if (more) { STAGE(cur ^ 1) }
        GroupMemoryBarrierWithGroupSync();
        cur ^= 1;
#else
        GroupMemoryBarrierWithGroupSync();
        if (more) { STAGE(0u) }
        GroupMemoryBarrierWithGroupSync();
#endif
    }
    #undef FETCH
    #undef STAGE
    #undef GEMM_LOAD_X
    #undef ROW_OF
    #undef ROWS_LIVE

    if (!MOE && t0 + GBM <= p.T && (N % GBN == 0 || n0 + GBN <= N))
    {
        [ForceUnroll] for (int i = 0; i < FRAG_M; i++) [ForceUnroll] for (int j = 0; j < FRAG_N; j++)
            acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(yb + uint64_t(t0 + wm + i * 16) * Y_STRIDE + n0 + wn + j * 16, 0u, Y_STRIDE);
        return;
    }
    // Partial tile: each subgroup stages its accumulators through its own shared slice.
    [ForceUnroll]
    for (int i = 0; i < FRAG_M; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j++)
        {
            acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(gC, sg * 256, 16u);
            GroupMemoryBarrierWithGroupSync();
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
                uint w = lane + e * 32;
                uint t = t0 + wm + i * 16 + w / 16;
                uint n = n0 + wn + j * 16 + w % 16;
#if MOE
                if (t < tileRows && n < N) yb[uint64_t(gItem[t]) * N + n] = gC[sg * 256 + w];
#else
                if (t < p.T && n < N) yb[uint64_t(t) * Y_STRIDE + n] = gC[sg * 256 + w];
#endif
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
#endif

#ifdef KERNEL_SPLITK_REDUCE
// Sums the SPLITK partial results of gemm_cm2 (x = partials [SPLITK][T][N]) into y.
[shader("compute")]
[numthreads(256, 1, 1)]
void splitk_reduce(uint3 dtid : SV_DispatchThreadID, uniform Params p)
{
    uint64_t i = dtid.x, total = uint64_t(p.T) * N;
    if (i >= total) return;
    float s = 0;
    [ForceUnroll] for (uint z = 0; z < SPLITK; z++) s += p.x[z * total + i];
    p.y[(i / N) * Y_STRIDE + i % N] = s;
}
#endif

#ifdef KERNEL_GEMM_NV2
// ---------------------------------------------------------------------------------------------
// Large GEMM on workgroup-scope cooperative matrices (DeviceFeatures.MatrixWorkgroup, VK_NV_cooperative_matrix2).
// The device loads the operand tiles itself through tensor layouts: activations as f16 rows (X_F16, X_STRIDE),
// weights with a transposed view and, for block formats, a decode function. Out-of-range rows/columns are
// clamped to zero on load and dropped on store, so partial tiles need no special path.
// Workgroup: 256 invocations computing NV_BT tokens x NV_BN rows of y. SPLITK as in gemm_cm2.
// Weight formats: f16, q8_0, q8_0r. Requires K % NV_BK == 0.
using namespace linalg;
#ifndef NV_BT
#define NV_BT 128
#endif
#ifndef NV_BN
#define NV_BN 256
#endif
#ifndef NV_BK
#define NV_BK 64
#endif
typealias NvA = CoopMat<half, MemoryScope.Workgroup, NV_BT, NV_BK, CoopMatMatrixUse.MatrixA>;
typealias NvB = CoopMat<half, MemoryScope.Workgroup, NV_BK, NV_BN, CoopMatMatrixUse.MatrixB>;
typealias NvC = CoopMat<float, MemoryScope.Workgroup, NV_BT, NV_BN, CoopMatMatrixUse.MatrixAccumulator>;

#if WFMT == FMT_Q8_0R
// One repacked row is one block: [K/32 f16 scales][K int8].
struct NvWBlock { uint16_t d[K / 32]; int8_t q[K]; };
half nv_decode(NvWBlock* b, uint32_t blockCoord[2], uint32_t c[2])
{
    return bit_cast<half>(b.d[c[1] >> 5]) * half(b.q[c[1]]);
}
#define NV_BLOCK K
#elif WFMT == FMT_Q8_0
struct NvWBlock { uint16_t d; int8_t q[32]; };
half nv_decode(NvWBlock* b, uint32_t blockCoord[2], uint32_t c[2])
{
    return bit_cast<half>(b.d) * half(b.q[c[1]]);
}
#define NV_BLOCK 32
#endif

[shader("compute")]
[numthreads(256, 1, 1)]
void gemm_nv2(uint3 gid : SV_GroupID, uniform Params p)
{
    uint n0 = gid.x * NV_BN, t0 = gid.y * NV_BT;
#ifndef SPLITK
#define SPLITK 1
#endif
#if SPLITK > 1
    const uint KS = ((K / NV_BK + SPLITK - 1) / SPLITK) * NV_BK;
    uint kBeg = gid.z * KS, kEnd = min(uint(K), kBeg + KS);
    float* yb = p.y + uint64_t(gid.z) * p.T * N;
#else
    uint kBeg = 0, kEnd = K;
    float* yb = p.y;
#endif
    var lx = TensorLayout<2, CoopMatClampMode.Constant>().Dimension(p.T, K).Stride(X_STRIDE, 1);
#if WFMT == FMT_F16
    var lw = TensorLayout<2, CoopMatClampMode.Constant>().Dimension(N, K).Stride(K, 1);
#else
    // Strides count blocks: one block per row for q8_0r, K/32 per row for q8_0.
    var lw = TensorLayout<2, CoopMatClampMode.Constant>().Dimension(N, K).Stride(K / NV_BLOCK, 1).BlockSize(1, NV_BLOCK);
#endif
    var wT = TensorView<2, false, 1, 0>();   // W is [N, K]; the B operand is its transpose
    NvC acc = NvC(0.0);
    for (uint k0 = kBeg; k0 < kEnd; k0 += NV_BK)
    {
        NvA a = NvA.Load((half*)p.x, NvA(0.0), lx.Slice(t0, NV_BT, k0, NV_BK));
#if WFMT == FMT_F16
        NvB b = NvB.Load((half*)p.w, NvB(0.0), lw.Slice(n0, NV_BN, k0, NV_BK), wT);
#else
        NvB b = NvB.Load((NvWBlock*)p.w, NvB(0.0), lw.Slice(n0, NV_BN, k0, NV_BK), wT, nv_decode);
#endif
        acc = coopMatMulAdd<float, false>(a, b, acc);
    }
    var ly = TensorLayout<2, CoopMatClampMode.Constant>().Dimension(p.T, N).Stride(Y_STRIDE, 1);
    acc.Store(yb, ly.Slice(t0, NV_BT, n0, NV_BN));
}
#endif
