// FP8 tensor-core GEMM for prefill-sized batches (DeviceFeatures.MatrixFp8): y[t, n] = sum_k x[t, k] * W[n, k] with
// E4M3 activations and E4M3 weights on 16x16x32 f8 x f8 -> f32 cooperative matrices.
//
// quantize_fp8 turns each activation row into E4M3 codes with one f32 scale per 128 values (amax / 448), stored as
// [K codes][K / 128 f32 scales] per row. The weights are the block-scaled f8e4m3_b128 / f8e4m3_b32 formats
// ({ f32 d; u8 q[B] } along K). Each weight block's MMAs accumulate in a fresh f32 accumulator, which is then scaled once:
//     acc[t, n] += d_x[t, k / 128] * d_w[n, k / B] * C[t, n]
// so a 128-value weight block costs one multiply-add per output element per four MMAs (the int8 GEMM in mmq.slang needs
// three to four per MMA). The accumulator element layout is probed as in mmq.slang (PAIR: NVIDIA's mma fragment pattern).
//
// Workgroup = F8_WM x F8_WN subgroups, each owning F8_FM x F8_FN 16x16 output fragments; K is walked in steps of 128 values
// staged in shared memory, the next step prefetched into registers during the MMAs. SPLITK > 1: workgroup z sums its share
// of K into slice z of y ([SPLITK][T][N], reduced by splitk_reduce in matmul.slang).
//
// Specialization macros: KERNEL_GEMM_FP8 / KERNEL_QUANTIZE_FP8, WBLOCK (32 or 128), K, N, Y_STRIDE, ROW_BYTES, F8_WM, F8_WN,
// F8_FM, F8_FN, SPLITK, F8_GENERIC_LAYOUT (1: per-element scale loads), X_STRIDE (quantize_fp8). Requires K % 128 == 0 and
// 32-lane subgroups.
#include "common.slang"
using namespace linalg;

// Quantized activation row: K codes, then K / 128 f32 scales.
#define F8ROW (K + K / 32)

#ifdef KERNEL_QUANTIZE_FP8
struct QuantParams
{
    float* x;
    uint8_t* q;
    uint T;
};

// E4M3 code of |v| <= 448, rounded to nearest even (subnormals below 2^-6 included).
uint e4m3_code(float v)
{
    uint sign = v < 0 ? 0x80u : 0u;
    float a = abs(v);
    uint code;
    if (a < 0.015625)
    {
        // Subnormals and zero: multiples of 2^-9 (a rounding up to 8 * 2^-9 is the smallest normal, code 8).
        code = uint(round(a * 512.0));
    }
    else
    {
        uint bits = asuint(a);
        bits += 0x7FFFFu + ((bits >> 20) & 1u);
        code = min((bits >> 20) - (120u << 3), 0x7Eu);
    }
    return sign | code;
}

// One thread per 4 values; a subgroup covers one 128-value group (32-lane subgroups).
[shader("compute")]
[numthreads(256, 1, 1)]
void quantize_fp8(uint3 dtid : SV_DispatchThreadID, uniform QuantParams p)
{
    uint i = dtid.x;
    uint t = i / (K / 4);
    uint k4 = i % (K / 4);
    bool live = t < p.T;
    float4 v = 0;
    if (live) v = *(float4*)(p.x + uint64_t(t) * X_STRIDE + k4 * 4);
    float amax = WaveActiveMax(max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w))));
    float d = amax / 448.0;
    float id = d > 0 ? 1.0 / d : 0.0;
    if (!live) return;
    uint8_t* row = p.q + uint64_t(t) * F8ROW;
    *(uint*)(row + k4 * 4) = e4m3_code(v.x * id) | (e4m3_code(v.y * id) << 8) | (e4m3_code(v.z * id) << 16) | (e4m3_code(v.w * id) << 24);
    if ((k4 & 31) == 0) *(float*)(row + K + (k4 / 32) * 4) = d;
}
#endif

#ifdef KERNEL_GEMM_FP8
#ifndef F8_WM
#define F8_WM 2
#endif
#ifndef F8_WN
#define F8_WN 4
#endif
#ifndef F8_FM
#define F8_FM 4
#endif
#ifndef F8_FN
#define F8_FN 2
#endif
#ifndef SPLITK
#define SPLITK 1
#endif
#ifndef F8_GENERIC_LAYOUT
#define F8_GENERIC_LAYOUT 0
#endif

#define F8_THREADS (F8_WM * F8_WN * 32)
#define F8_BM (F8_WM * F8_FM * 16)
#define F8_BN (F8_WN * F8_FN * 16)
#define F8_BK 128
// Weight blocks per K step and MMAs per weight block.
#define F8_WBLOCKS (F8_BK / WBLOCK)
#define F8_MMAS (WBLOCK / 32)
// Shared rows: 128 codes + 16 bytes of padding, in words.
#define F8_RS (F8_BK / 4 + 4)
// Prefetch items per thread: activation uint4s, weight words, weight block scales.
#define F8_A_ITEMS ((F8_BM * F8_BK / 16 + F8_THREADS - 1) / F8_THREADS)
#define F8_B_ITEMS ((F8_BN * F8_BK / 4 + F8_THREADS - 1) / F8_THREADS)
#define F8_BS_ITEMS ((F8_BN * F8_WBLOCKS + F8_THREADS - 1) / F8_THREADS)

struct Params
{
    uint8_t* x;
    uint8_t* w;
    float* y;
    uint T;
};

typealias MatA8 = CoopMat<FloatE4M3, MemoryScope.Subgroup, 16, 32, CoopMatMatrixUse.MatrixA>;
typealias MatB8 = CoopMat<FloatE4M3, MemoryScope.Subgroup, 32, 16, CoopMatMatrixUse.MatrixB>;
typealias AccF = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

groupshared uint sA[F8_BM * F8_RS];
groupshared uint sB[F8_BN * F8_RS];
// Scales of the staged step: one per activation row; F8_WBLOCKS per weight row, block-major.
groupshared float sAS[F8_BM];
groupshared float sBS[F8_WBLOCKS * F8_BN];

// The tile loop and the stores. PAIR: the accumulator layout is the pair layout (element e is row r0 + 8 * (e >> 2),
// column c0 + (e & 1) + 8 * ((e >> 1) & 1)); otherwise every element's row and column come from `index` (row * 16 + column).
void fp8_tile<let PAIR : bool>(uint3 gid, uint tid, Params p, int index[8])
{
    uint sg = tid / 32;
    uint n0 = gid.x * F8_BN;
    uint t0 = gid.y * F8_BM;
    uint wm = (sg / F8_WN) * (F8_FM * 16);
    uint wn = (sg % F8_WN) * (F8_FN * 16);
    uint r0 = uint(index[0]) >> 4, c0 = uint(index[0]) & 15;
#if SPLITK > 1
    const uint KS = ((K / F8_BK + SPLITK - 1) / SPLITK) * F8_BK;
    uint kBeg = gid.z * KS, kEnd = min(uint(K), kBeg + KS);
    float* yb = p.y + uint64_t(gid.z) * p.T * N;
#else
    uint kBeg = 0, kEnd = K;
    float* yb = p.y;
#endif

    float acc[F8_FM][F8_FN][8];
    [ForceUnroll] for (int i = 0; i < F8_FM; i++) [ForceUnroll] for (int j = 0; j < F8_FN; j++) [ForceUnroll] for (int e = 0; e < 8; e++) acc[i][j][e] = 0;

    uint4 xa[F8_A_ITEMS];
    float xs;
    uint wq[F8_B_ITEMS];
    float wsc[F8_BS_ITEMS];

    // Rows past T or N are clamped for loading (their results are never stored).
    #define FETCH(k0)                                                                                   \
    [ForceUnroll] for (uint it = 0; it < F8_A_ITEMS; it++)                                              \
    {                                                                                                   \
        uint idx = tid + it * F8_THREADS, r = idx / (F8_BK / 16), q = idx % (F8_BK / 16);               \
        uint t = min(t0 + r, p.T - 1);                                                                  \
        if (idx < F8_BM * F8_BK / 16) xa[it] = *(uint4*)byteAt(p.x, uint64_t(t) * F8ROW + (k0) + q * 16); \
    }                                                                                                   \
    if (tid < F8_BM) xs = *(float*)byteAt(p.x, uint64_t(min(t0 + tid, p.T - 1)) * F8ROW + K + (k0) / 32); \
    [ForceUnroll] for (uint it = 0; it < F8_B_ITEMS; it++)                                              \
    {                                                                                                   \
        uint idx = tid + it * F8_THREADS, r = idx / (F8_BK / 4), word = idx % (F8_BK / 4);              \
        uint8_t* row = rowStart(p.w, ROW_BYTES, min(n0 + r, uint(N) - 1));                              \
        uint blk = ((k0) + word * 4) / WBLOCK;                                                          \
        if (idx < F8_BN * F8_BK / 4) wq[it] = *(uint*)(row + (uint64_t(blk) * (WBLOCK + 4) + 4 + ((k0) + word * 4) % WBLOCK)); \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < F8_BS_ITEMS; it++)                                             \
    {                                                                                                   \
        uint idx = tid + it * F8_THREADS, r = idx % F8_BN, bl = idx / F8_BN;                            \
        uint8_t* row = rowStart(p.w, ROW_BYTES, min(n0 + r, uint(N) - 1));                              \
        if (idx < F8_BN * F8_WBLOCKS) wsc[it] = *(float*)(row + uint64_t((k0) / WBLOCK + bl) * (WBLOCK + 4)); \
    }
    #define STAGE()                                                                                     \
    [ForceUnroll] for (uint it = 0; it < F8_A_ITEMS; it++)                                              \
    {                                                                                                   \
        uint idx = tid + it * F8_THREADS, r = idx / (F8_BK / 16), q = idx % (F8_BK / 16);               \
        if (idx < F8_BM * F8_BK / 16)                                                                   \
        {                                                                                               \
            uint o = r * F8_RS + q * 4;                                                                 \
            sA[o] = xa[it].x; sA[o + 1] = xa[it].y; sA[o + 2] = xa[it].z; sA[o + 3] = xa[it].w;         \
        }                                                                                               \
    }                                                                                                   \
    if (tid < F8_BM) sAS[tid] = xs;                                                                     \
    [ForceUnroll] for (uint it = 0; it < F8_B_ITEMS; it++)                                              \
    {                                                                                                   \
        uint idx = tid + it * F8_THREADS, r = idx / (F8_BK / 4), word = idx % (F8_BK / 4);              \
        if (idx < F8_BN * F8_BK / 4) sB[r * F8_RS + word] = wq[it];                                     \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < F8_BS_ITEMS; it++)                                             \
    {                                                                                                   \
        uint idx = tid + it * F8_THREADS;                                                               \
        if (idx < F8_BN * F8_WBLOCKS) sBS[idx] = wsc[it];                                               \
    }

    // A split-K slice may be empty (K steps not divisible by SPLITK): never prefetch at or past K.
    uint steps = (kEnd - kBeg) / F8_BK;
    if (steps > 0)
    {
        FETCH(kBeg)
        STAGE()
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 0; s < steps; s++)
    {
        bool more = s + 1 < steps;
        if (more)
        {
            FETCH(kBeg + (s + 1) * F8_BK)
        }

        float xr[F8_FM][2];
        if (PAIR)
        {
            [ForceUnroll]
            for (int i = 0; i < F8_FM; i++)
            {
                xr[i][0] = sAS[wm + i * 16 + r0];
                xr[i][1] = sAS[wm + i * 16 + r0 + 8];
            }
        }

        [MaxIters(F8_WBLOCKS)] [loop]
        for (uint wb = 0; wb < F8_WBLOCKS; wb++)
        {
            AccF c[F8_FM][F8_FN];
            [ForceUnroll] for (int i = 0; i < F8_FM; i++) [ForceUnroll] for (int j = 0; j < F8_FN; j++) c[i][j] = AccF(0.0);
            [ForceUnroll]
            for (uint m = 0; m < F8_MMAS; m++)
            {
                uint kw = (wb * WBLOCK + m * 32) / 4;
                MatB8 b[F8_FN];
                [ForceUnroll]
                for (int j = 0; j < F8_FN; j++)
                {
                    b[j] = MatB8.Load<CoopMatMatrixLayout.ColumnMajor>(sB, (wn + j * 16) * F8_RS + kw, F8_RS);
                }
                [ForceUnroll]
                for (int i = 0; i < F8_FM; i++)
                {
                    MatA8 a = MatA8.Load<CoopMatMatrixLayout.RowMajor>(sA, (wm + i * 16) * F8_RS + kw, F8_RS);
                    [ForceUnroll]
                    for (int j = 0; j < F8_FN; j++)
                    {
                        c[i][j] = coopMatMulAdd<float, false>(a, b[j], c[i][j]);
                    }
                }
            }

            uint bs = wb * F8_BN;
            [ForceUnroll]
            for (int j = 0; j < F8_FN; j++)
            {
                float ws[4];
                if (PAIR)
                {
                    uint base = bs + wn + j * 16 + c0;
                    ws[0] = sBS[base]; ws[1] = sBS[base + 1]; ws[2] = sBS[base + 8]; ws[3] = sBS[base + 9];
                }
                [ForceUnroll]
                for (int i = 0; i < F8_FM; i++)
                {
                    [ForceUnroll]
                    for (uint e = 0; e < 8; e++)
                    {
                        if (PAIR)
                        {
                            acc[i][j][e] += c[i][j][e] * (xr[i][e >> 2] * ws[e & 3]);
                        }
                        else
                        {
                            uint r = uint(index[e]) >> 4, col = uint(index[e]) & 15;
                            acc[i][j][e] += c[i][j][e] * (sAS[wm + i * 16 + r] * sBS[bs + wn + j * 16 + col]);
                        }
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (more)
        {
            STAGE()
            GroupMemoryBarrierWithGroupSync();
        }
    }
    #undef FETCH
    #undef STAGE

    [ForceUnroll]
    for (int i = 0; i < F8_FM; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < F8_FN; j++)
        {
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
                uint t = t0 + wm + i * 16 + (uint(index[e]) >> 4);
                uint n = n0 + wn + j * 16 + (uint(index[e]) & 15);
                if (t < p.T && n < N) yb[uint64_t(t) * Y_STRIDE + n] = acc[i][j][e];
            }
        }
    }
}

[shader("compute")]
[numthreads(F8_THREADS, 1, 1)]
void gemm_fp8(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    uint tid = ltid.x;
    // Element layout of the f32 accumulator: load the matrix whose element (r, c) is r * 16 + c.
    for (uint i = tid; i < 256; i += F8_THREADS) sB[i] = asuint(float(i));
    GroupMemoryBarrierWithGroupSync();
    AccF probe = AccF.Load<CoopMatMatrixLayout.RowMajor>(sB, 0, 16);
    int index[8];
    [ForceUnroll] for (uint e = 0; e < 8; e++) index[e] = int(probe[e]);
    // The tile loop overwrites the probe in its first step.
    GroupMemoryBarrierWithGroupSync();
    uint r0 = uint(index[0]) >> 4, c0 = uint(index[0]) & 15;
    bool pairLane = (c0 & 1) == 0 && r0 < 8 && c0 < 8;
    [ForceUnroll]
    for (uint e = 0; e < 8; e++)
        pairLane = pairLane && uint(index[e]) == (r0 + 8 * (e >> 2)) * 16 + c0 + (e & 1) + 8 * ((e >> 1) & 1);
    // Uniform over the workgroup: every subgroup sees the same layout.
    if (!F8_GENERIC_LAYOUT && WaveActiveAllTrue(pairLane))
        fp8_tile<true>(gid, tid, p, index);
    else
        fp8_tile<false>(gid, tid, p, index);
}
#endif
