// y[t, n] = sum_k x[t, k] * W[n, k]      (x: f32 rows with stride X_STRIDE, W: WFMT rows of K values)
// Specialization macros: WFMT, K, N, X_STRIDE, Y_STRIDE, ROW_BYTES, SG
#include "common.slang"

// DUAL: two weight matrices of identical shape/format share x; the outputs are combined per element by
// `float epi2(float a, float b)` (defined by the including generated source, e.g. silu(a) * b) — matvec paths only.
#ifndef DUAL
#define DUAL 0
#endif

struct Params
{
    float* x;
    uint8_t* w;
#if DUAL
    uint8_t* w2;
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
    uint8_t* wr = p.w + ROW_BYTES * row;
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
#define BM 64
#define BN 64
#define BK 32
#define TM 4
#define TN 4

groupshared float sX[BK][BM + 4];
groupshared float sW[BK][BN + 4];

[shader("compute")]
[numthreads(256, 1, 1)]
void gemm(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    uint tid = ltid.x;
    uint n0 = gid.x * BN;
    uint t0 = gid.y * BM;
    uint tx = tid % (BN / TN);   // 16 threads along N
    uint ty = tid / (BN / TN);   // 16 threads along T

    float acc[TM][TN];
    [ForceUnroll] for (int i = 0; i < TM; i++) [ForceUnroll] for (int j = 0; j < TN; j++) acc[i][j] = 0;

    for (uint k0 = 0; k0 < K; k0 += BK)
    {
        // W tile: BN rows x BK values = 64 rows * 4 groups of 8 = 256 loads (one per thread).
        {
            uint r = tid / 4, gq = tid % 4;
            float w[8];
            if (n0 + r < N && k0 / 8 + gq < K / 8) load8(p.w, ROW_BYTES, n0 + r, k0 / 8 + gq, w);
            else { [ForceUnroll] for (int i = 0; i < 8; i++) w[i] = 0; }
            [ForceUnroll] for (int i = 0; i < 8; i++) sW[gq * 8 + i][r] = w[i];
        }
        // X tile: BM tokens x BK values = 64 * 32 floats = 2048 = 256 threads * 8.
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
            if (n < N) p.y[uint64_t(t) * Y_STRIDE + n] = acc[i][j];
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
#define GBK 32
#define GAS (GBK + 8)
#define GA_PER_T (GBM * GBK / 8 / GTHREADS)
#define GB_PER_T (GBN * GBK / 8 / GTHREADS)
typealias MatA2 = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB2 = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias Acc2 = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

groupshared half gA[2 * GBM * GAS];
groupshared half gB[2 * GBN * GAS];
groupshared float gC[WARPS_M * WARPS_N * 256];

[shader("compute")]
[numthreads(GTHREADS, 1, 1)]
void gemm_cm2(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    uint tid = ltid.x;
    uint sg = tid / 32, lane = tid % 32;
    uint n0 = gid.x * GBN;
    uint t0 = gid.y * GBM;
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

    half xs[GA_PER_T][8];
    half ws[GB_PER_T][8];

    // X_F16: activations were converted to dense f16 rows beforehand (to_f16), halving tile load traffic.
#if X_F16
    #define GEMM_LOAD_X(t, k, dst)                                                                      \
        { uint4 u = 0;                                                                                  \
          if ((t) < p.T) u = ((uint4*)((uint16_t*)p.x + uint64_t(t) * X_STRIDE + (k)))[0];              \
          dst[0] = half(f16tof32(u.x & 0xFFFF)); dst[1] = half(f16tof32(u.x >> 16));                    \
          dst[2] = half(f16tof32(u.y & 0xFFFF)); dst[3] = half(f16tof32(u.y >> 16));                    \
          dst[4] = half(f16tof32(u.z & 0xFFFF)); dst[5] = half(f16tof32(u.z >> 16));                    \
          dst[6] = half(f16tof32(u.w & 0xFFFF)); dst[7] = half(f16tof32(u.w >> 16)); }
#else
    #define GEMM_LOAD_X(t, k, dst)                                                                      \
        { float4 a = 0, b = 0;                                                                          \
          if ((t) < p.T) { float4* xp = (float4*)(p.x + uint64_t(t) * X_STRIDE + (k)); a = xp[0]; b = xp[1]; } \
          dst[0] = half(a.x); dst[1] = half(a.y); dst[2] = half(a.z); dst[3] = half(a.w);               \
          dst[4] = half(b.x); dst[5] = half(b.y); dst[6] = half(b.z); dst[7] = half(b.w); }
#endif
    #define FETCH(k0)                                                                                   \
    [ForceUnroll] for (uint it = 0; it < GA_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / 4, q = idx % 4;                                       \
        uint t = t0 + r;                                                                                \
        GEMM_LOAD_X(t, (k0) + q * 8, xs[it])                                                            \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < GB_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / 4, q = idx % 4;                                       \
        float w[8];                                                                                     \
        if (N % GBN == 0 || n0 + r < N) load8(p.w, ROW_BYTES, n0 + r, (k0) / 8 + q, w);                 \
        else { [ForceUnroll] for (int i = 0; i < 8; i++) w[i] = 0; }                                    \
        [ForceUnroll] for (int i = 0; i < 8; i++) ws[it][i] = half(w[i]);                               \
    }
    #define STAGE(buf)                                                                                  \
    [ForceUnroll] for (uint it = 0; it < GA_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / 4, q = idx % 4;                                       \
        uint oa = (buf) * GBM * GAS + r * GAS + q * 8;                                                  \
        [ForceUnroll] for (int i = 0; i < 8; i++) gA[oa + i] = xs[it][i];                               \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < GB_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / 4, q = idx % 4;                                       \
        uint ob = (buf) * GBN * GAS + r * GAS + q * 8;                                                  \
        [ForceUnroll] for (int i = 0; i < 8; i++) gB[ob + i] = ws[it][i];                               \
    }

    FETCH(kBeg)
    STAGE(0u)
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
            [ForceUnroll] for (int j = 0; j < FRAG_N; j++) b[j] = MatB2.Load<CoopMatMatrixLayout.ColumnMajor>(gB, bb + (wn + j * 16) * GAS + kk, GAS);
            [ForceUnroll]
            for (int i = 0; i < FRAG_M; i++)
            {
                MatA2 a = MatA2.Load<CoopMatMatrixLayout.RowMajor>(gA, ba + (wm + i * 16) * GAS + kk, GAS);
                [ForceUnroll] for (int j = 0; j < FRAG_N; j++) acc[i][j] = coopMatMulAdd<float, false>(a, b[j], acc[i][j]);
            }
        }
        if (more) { STAGE(cur ^ 1) }
        GroupMemoryBarrierWithGroupSync();
        cur ^= 1;
    }
    #undef FETCH
    #undef STAGE
    #undef GEMM_LOAD_X

    if (t0 + GBM <= p.T && (N % GBN == 0 || n0 + GBN <= N))
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
                if (t < p.T && n < N) yb[uint64_t(t) * Y_STRIDE + n] = gC[sg * 256 + w];
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
