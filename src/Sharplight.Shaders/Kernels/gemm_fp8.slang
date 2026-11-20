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
// staged in shared memory, the next step prefetched into registers during the MMAs. (Measured on Blackwell at 512 rows: the
// MMAs and shared-memory loads alone run at ~520 TFLOPS; the global fetch of both operands brings that down to ~280, and
// neither double-buffered shared memory nor 64-row tiles hide it.) SPLITK > 1: workgroup z sums its share
// of K into slice z of y ([SPLITK][T][N], reduced by splitk_reduce in matmul.slang).
//
// DUAL (SPLITK = 1, F8_FN even): two weight matrices of the same shape (w, w2; e.g. gate and up) share the activations. The
// B tile's 16-row fragments alternate between them (fragment 2i: rows of w, 2i + 1: the same rows of w2), so each thread
// holds both results of an output element and stores `epi2(a, b)` (defined by the including generated source, e.g.
// silu(a) * b); a workgroup covers F8_BN / 2 outputs.
//
// Specialization macros: KERNEL_GEMM_FP8 / KERNEL_QUANTIZE_FP8, WBLOCK (32 or 128), K, N, Y_STRIDE, ROW_BYTES, F8_WM, F8_WN,
// F8_FM, F8_FN, SPLITK, DUAL, F8_SPLIT, F8_GROUPED, F8_GENERIC_LAYOUT (1: per-element scale loads), X_STRIDE and DENSE_XCAP
// (quantize_fp8). Requires K % 128 == 0 and 32-lane subgroups.
#include "common.slang"
#include "splitk.slang"
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
    float d = e4m3_group_scale(v);
    if (!live) return;
    uint8_t* row = p.q + uint64_t(t) * F8ROW;
    *(uint*)(row + k4 * 4) = e4m3_pack4(v, d);
    if ((k4 & 31) == 0) *(float*)(row + K + (k4 / 32) * 4) = d;
#ifdef DENSE_XCAP
    // The dense copy of the scales after DENSE_XCAP rows: [K / 128][DENSE_XCAP] f32 (gemm_fp8_native.cu).
    if ((k4 & 31) == 0) *(float*)(p.q + uint64_t(DENSE_XCAP) * F8ROW + (uint64_t(k4 / 32) * DENSE_XCAP + t) * 4) = d;
#endif
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
#ifndef DUAL
#define DUAL 0
#endif
// F8_SPLIT: weight rows in the aligned split layout [K codes][K / WBLOCK f32 scales] (WeightRepack.Fp8B32Split /
// Fp8B128Split), fetched with 16-byte loads; otherwise blocks of { f32 d; u8 q[WBLOCK] }, fetched a word at a time.
#ifndef F8_SPLIT
#define F8_SPLIT 0
#endif
// F8_GROUPED (with F8_SPLIT): the split rows in groups of 128 rows, [128 rows of K codes][K / 128 blocks of the 128 rows' f32
// scales] (WeightRepack.Fp8B128Grouped).
#ifndef F8_GROUPED
#define F8_GROUPED 0
#endif
#if DUAL && ((SPLITK != 1 && !SPLITK_FUSED) || F8_FN % 2 != 0)
#error "the dual fp8 GEMM needs SPLITK = 1 (or the fused split-K reduction) and an even F8_FN"
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
#if F8_SPLIT
#define F8_B_ITEMS ((F8_BN * F8_BK / 16 + F8_THREADS - 1) / F8_THREADS)
#else
#define F8_B_ITEMS ((F8_BN * F8_BK / 4 + F8_THREADS - 1) / F8_THREADS)
#endif
#define F8_BS_ITEMS ((F8_BN * F8_WBLOCKS + F8_THREADS - 1) / F8_THREADS)

struct Params
{
    uint8_t* x;
    uint8_t* w;
#if DUAL
    uint8_t* w2;
#endif
    float* y;
#if SPLITK_FUSED
    float* yout;      // SPLITK_FUSED (splitk.slang): the reduced output (y holds the partial slices)
    uint* arrivals;   // per output tile: the split workgroups arrived so far (zero between launches)
#endif
#if MULTI > 1
    uint8_t* mw1;     // MULTI (splitk.slang): the second and third weight matrices, their outputs and (MULTI_BIAS) the bias rows
#if MULTI > 2
    uint8_t* mw2;
#endif
    float* yout1;
#if MULTI > 2
    float* yout2;
#endif
#if MULTI_BIAS
    float* mb0;
    float* mb1;
#if MULTI > 2
    float* mb2;
#endif
#endif
#endif
    uint T;
};

// Weight matrix and output row of B tile row r (n0: the workgroup's first output row).
#if DUAL
#define F8_BN_OUT (F8_BN / 2)
#define F8_WPTR(r) ((((r) / 16) & 1) != 0 ? p.w2 : p.w)
#define F8_WROW(n0, r) ((n0) + ((r) / 32) * 16 + (r) % 16)
#elif MULTI > 1
#define F8_BN_OUT F8_BN
#define F8_WPTR(r) wbaseM
#define F8_WROW(n0, r) ((n0) + (r))
#else
#define F8_BN_OUT F8_BN
#define F8_WPTR(r) p.w
#define F8_WROW(n0, r) ((n0) + (r))
#endif
#if SPLITK_FUSED
#if SPLITK < 2 || (DUAL && MULTI > 1)
#error "the fused split-K reduction of gemm_fp8 needs SPLITK > 1 (and DUAL without MULTI)"
#endif
// DUAL: the partial slices hold both products of each output, [SPLITK][T][2N] (w's at column n, w2's at N + n).
#define SPLITK_STORE2(p, g, u, t, n) { p.yout[uint64_t(t) * Y_STRIDE + (n)] = epi2(g, u); }
#if MULTI > 1
#define SPLITK_STORE(p, s, t, n) MULTI_STORE(p, s, t, n)
#else
#define SPLITK_STORE(p, s, t, n) { p.yout[uint64_t(t) * Y_STRIDE + (n)] = (s); }
#endif
#define SPLITK_ROW_SCALE(p, t) 1.0
#endif

#if SL_TARGET_CUDA
#include "cuda_mma.slang"
typealias MatA8 = CuA8f;
typealias MatB8 = CuB8f;
typealias AccF = CuAccF;
#else
typealias MatA8 = CoopMat<FloatE4M3, MemoryScope.Subgroup, 16, 32, CoopMatMatrixUse.MatrixA>;
typealias MatB8 = CoopMat<FloatE4M3, MemoryScope.Subgroup, 32, 16, CoopMatMatrixUse.MatrixB>;
typealias AccF = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
#endif

groupshared uint sA[F8_BM * F8_RS];
groupshared uint sB[F8_BN * F8_RS];
// Scales of the staged step: one per activation row; F8_WBLOCKS per weight row, block-major.
groupshared float sAS[F8_BM];
groupshared float sBS[F8_WBLOCKS * F8_BN];

// The MMAs and block scaling of the staged K step (see fp8_tile).
void fp8_step<let PAIR : bool>(inout float acc[F8_FM][F8_FN][8], uint wm, uint wn, uint r0, uint c0, int index[8])
{
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

    // The block products are added with an explicit fused multiply-add, so the dual kernel (DUAL) and the single one round
    // alike whatever the compiler would contract in each. The block accumulator holds one row or column of fragments rather
    // than F8_FM x F8_FN, which frees registers (10-13 % faster from 256 rows) at the cost of reloading B or A fragments.
    [MaxIters(F8_WBLOCKS)] [loop]
    for (uint wb = 0; wb < F8_WBLOCKS; wb++)
    {
        uint bs = wb * F8_BN;
        if (PAIR)
        {
            // Pair layout: one output row fragment at a time (the A fragments are loaded once, the B fragments per row fragment;
            // [4096 → 12288] at 2048 rows 194 → 197 TFLOPS against column fragments). Generic layout: one column fragment at a time.
            float ws[F8_FN][4];
            [ForceUnroll]
            for (int j = 0; j < F8_FN; j++)
            {
                uint base = bs + wn + j * 16 + c0;
                ws[j][0] = sBS[base]; ws[j][1] = sBS[base + 1]; ws[j][2] = sBS[base + 8]; ws[j][3] = sBS[base + 9];
            }
#if SL_TARGET_CUDA
            // CUDA: the B fragments of the weight block are loaded once for all row fragments. The shared-memory fragment loads are
            // volatile there (ldmatrix in inline PTX), so the compiler cannot merge the reloads of the form below as the Vulkan
            // driver does (33 % slower fp8 GEMMs).
            MatB8 bw[F8_MMAS][F8_FN];
            [ForceUnroll]
            for (uint m = 0; m < F8_MMAS; m++)
            {
                [ForceUnroll]
                for (int j = 0; j < F8_FN; j++)
                {
                    bw[m][j] = MatB8.Load<CoopMatMatrixLayout.ColumnMajor>(sB, (wn + j * 16) * F8_RS + (wb * WBLOCK + m * 32) / 4, F8_RS);
                }
            }
#endif
            [ForceUnroll]
            for (int i = 0; i < F8_FM; i++)
            {
                AccF c[F8_FN];
                [ForceUnroll] for (int j = 0; j < F8_FN; j++) c[j] = AccF(0.0);
                [ForceUnroll]
                for (uint m = 0; m < F8_MMAS; m++)
                {
                    uint kw = (wb * WBLOCK + m * 32) / 4;
                    MatA8 a = MatA8.Load<CoopMatMatrixLayout.RowMajor>(sA, (wm + i * 16) * F8_RS + kw, F8_RS);
                    [ForceUnroll]
                    for (int j = 0; j < F8_FN; j++)
                    {
#if SL_TARGET_CUDA
                        c[j] = coopMatMulAdd<float, false>(a, bw[m][j], c[j]);
#else
                        MatB8 b = MatB8.Load<CoopMatMatrixLayout.ColumnMajor>(sB, (wn + j * 16) * F8_RS + kw, F8_RS);
                        c[j] = coopMatMulAdd<float, false>(a, b, c[j]);
#endif
                    }
                }
                [ForceUnroll]
                for (int j = 0; j < F8_FN; j++)
                    [ForceUnroll]
                    for (uint e = 0; e < 8; e++)
                        acc[i][j][e] = fma(c[j][e], xr[i][e >> 2] * ws[j][e & 3], acc[i][j][e]);
            }
            continue;
        }
        [ForceUnroll]
        for (int j = 0; j < F8_FN; j++)
        {
            AccF c[F8_FM];
            [ForceUnroll] for (int i = 0; i < F8_FM; i++) c[i] = AccF(0.0);
            [ForceUnroll]
            for (uint m = 0; m < F8_MMAS; m++)
            {
                uint kw = (wb * WBLOCK + m * 32) / 4;
                MatB8 b = MatB8.Load<CoopMatMatrixLayout.ColumnMajor>(sB, (wn + j * 16) * F8_RS + kw, F8_RS);
                [ForceUnroll]
                for (int i = 0; i < F8_FM; i++)
                {
                    MatA8 a = MatA8.Load<CoopMatMatrixLayout.RowMajor>(sA, (wm + i * 16) * F8_RS + kw, F8_RS);
                    c[i] = coopMatMulAdd<float, false>(a, b, c[i]);
                }
            }

            [ForceUnroll]
            for (int i = 0; i < F8_FM; i++)
            {
                [ForceUnroll]
                for (uint e = 0; e < 8; e++)
                {
                    uint r = uint(index[e]) >> 4, col = uint(index[e]) & 15;
                    acc[i][j][e] = fma(c[i][e], sAS[wm + i * 16 + r] * sBS[bs + wn + j * 16 + col], acc[i][j][e]);
                }
            }
        }
    }
}

// The tile loop and the stores. PAIR: the accumulator layout is the pair layout (element e is row r0 + 8 * (e >> 2),
// column c0 + (e & 1) + 8 * ((e >> 1) & 1)); otherwise every element's row and column come from `index` (row * 16 + column).
void fp8_tile<let PAIR : bool>(uint3 gid, uint tid, Params p, int index[8])
{
    uint sg = tid / 32;
    // Grid x = token tile, y = weight tile: consecutive workgroups share a weight tile, so each weight byte comes from memory
    // about once even when the matrix (or a dual pair) is larger than the L2 cache.
    uint n0 = gid.y * F8_BN_OUT;
    uint t0 = gid.x * F8_BM;
    uint wm = (sg / F8_WN) * (F8_FM * 16);
    uint wn = (sg % F8_WN) * (F8_FN * 16);
    uint r0 = uint(index[0]) >> 4, c0 = uint(index[0]) & 15;
#if MULTI > 1
    uint8_t* wbaseM = MULTI_WBASE(p, n0);
#endif
#if SPLITK > 1
    const uint KS = ((K / F8_BK + SPLITK - 1) / SPLITK) * F8_BK;
    uint kBeg = gid.z * KS, kEnd = min(uint(K), kBeg + KS);
    float* yb = p.y + uint64_t(gid.z) * p.T * N * (DUAL ? 2 : 1);
#else
    uint kBeg = 0, kEnd = K;
    float* yb = p.y;
#endif

    float acc[F8_FM][F8_FN][8];
    [ForceUnroll] for (int i = 0; i < F8_FM; i++) [ForceUnroll] for (int j = 0; j < F8_FN; j++) [ForceUnroll] for (int e = 0; e < 8; e++) acc[i][j][e] = 0;

    uint4 xa[F8_A_ITEMS];
    float xs;
#if F8_SPLIT
    uint4 wq[F8_B_ITEMS];
#else
    uint wq[F8_B_ITEMS];
#endif
    float wsc[F8_BS_ITEMS];

#if F8_SPLIT
#if F8_GROUPED
    // Codes of row n at its group + (n % 128) K; scale of block b at F8_WSCALES(n) + 512 b (its group + 128 K + 4 (n % 128)).
    #define F8_WCODES(base, n) ((uint8_t*)(uint64_t(base) + ROW_BYTES * ((n) & ~127u) + uint64_t(K) * ((n) & 127u)))
    #define F8_WSCALES(base, n) ((uint8_t*)(uint64_t(base) + ROW_BYTES * ((n) & ~127u) + uint64_t(K) * 128 + ((n) & 127u) * 4))
    #define F8_SCALE_AT(b) (uint64_t(b) * 512)
#else
    // Codes: 16 per item at row + k; scale of block b at row + K + 4 b.
    #define F8_WCODES(base, n) rowStart(base, ROW_BYTES, n)
    #define F8_WSCALES(base, n) rowStart(base, ROW_BYTES, n)
    #define F8_SCALE_AT(b) (uint64_t(K) + uint64_t(b) * 4)
#endif
    #define F8_FETCH_W(k0)                                                                              \
    [ForceUnroll] for (uint it = 0; it < F8_B_ITEMS; it++)                                              \
    {                                                                                                   \
        uint idx = tid + it * F8_THREADS, r = idx / (F8_BK / 16), q = idx % (F8_BK / 16);               \
        uint8_t* row = F8_WCODES(F8_WPTR(r), min(F8_WROW(n0, r), uint(N) - 1));                         \
        if (idx < F8_BN * F8_BK / 16) wq[it] = LOAD_U4_ALIGNED(row + (k0) + q * 16, ALIGN_OF(ROW_BYTES)); \
    }
    #define F8_STAGE_W()                                                                                \
    [ForceUnroll] for (uint it = 0; it < F8_B_ITEMS; it++)                                              \
    {                                                                                                   \
        uint idx = tid + it * F8_THREADS, r = idx / (F8_BK / 16), q = idx % (F8_BK / 16);               \
        if (idx < F8_BN * F8_BK / 16)                                                                   \
        {                                                                                               \
            uint o = r * F8_RS + q * 4;                                                                 \
            sB[o] = wq[it].x; sB[o + 1] = wq[it].y; sB[o + 2] = wq[it].z; sB[o + 3] = wq[it].w;         \
        }                                                                                               \
    }
#else
    #define F8_WSCALES(base, n) rowStart(base, ROW_BYTES, n)
    #define F8_SCALE_AT(b) (uint64_t(b) * (WBLOCK + 4))
    #define F8_FETCH_W(k0)                                                                              \
    [ForceUnroll] for (uint it = 0; it < F8_B_ITEMS; it++)                                              \
    {                                                                                                   \
        uint idx = tid + it * F8_THREADS, r = idx / (F8_BK / 4), word = idx % (F8_BK / 4);              \
        uint8_t* row = rowStart(F8_WPTR(r), ROW_BYTES, min(F8_WROW(n0, r), uint(N) - 1));               \
        uint blk = ((k0) + word * 4) / WBLOCK;                                                          \
        if (idx < F8_BN * F8_BK / 4) wq[it] = *(uint*)(row + (uint64_t(blk) * (WBLOCK + 4) + 4 + ((k0) + word * 4) % WBLOCK)); \
    }
    #define F8_STAGE_W()                                                                                \
    [ForceUnroll] for (uint it = 0; it < F8_B_ITEMS; it++)                                              \
    {                                                                                                   \
        uint idx = tid + it * F8_THREADS, r = idx / (F8_BK / 4), word = idx % (F8_BK / 4);              \
        if (idx < F8_BN * F8_BK / 4) sB[r * F8_RS + word] = wq[it];                                     \
    }
#endif
    // Rows past T or N are clamped for loading (their results are never stored).
    #define FETCH(k0)                                                                                   \
    [ForceUnroll] for (uint it = 0; it < F8_A_ITEMS; it++)                                              \
    {                                                                                                   \
        uint idx = tid + it * F8_THREADS, r = idx / (F8_BK / 16), q = idx % (F8_BK / 16);               \
        uint t = min(t0 + r, p.T - 1);                                                                  \
        if (idx < F8_BM * F8_BK / 16) xa[it] = LOAD_U4_ALIGNED(byteAt(p.x, uint64_t(t) * F8ROW + (k0) + q * 16), ALIGN_OF(F8ROW)); \
    }                                                                                                   \
    if (tid < F8_BM) xs = *(float*)byteAt(p.x, uint64_t(min(t0 + tid, p.T - 1)) * F8ROW + K + (k0) / 32); \
    F8_FETCH_W(k0)                                                                                      \
    [ForceUnroll] for (uint it = 0; it < F8_BS_ITEMS; it++)                                             \
    {                                                                                                   \
        uint idx = tid + it * F8_THREADS, r = idx % F8_BN, bl = idx / F8_BN;                            \
        uint8_t* row = F8_WSCALES(F8_WPTR(r), min(F8_WROW(n0, r), uint(N) - 1));                                    \
        if (idx < F8_BN * F8_WBLOCKS) wsc[it] = *(float*)(row + F8_SCALE_AT((k0) / WBLOCK + bl));       \
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
    F8_STAGE_W()                                                                                        \
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
        fp8_step<PAIR>(acc, wm, wn, r0, c0, index);
        GroupMemoryBarrierWithGroupSync();
        if (more)
        {
            STAGE()
            GroupMemoryBarrierWithGroupSync();
        }
    }
    #undef FETCH
    #undef STAGE
    #undef F8_FETCH_W
    #undef F8_STAGE_W
    #undef F8_SCALE_AT
    #undef F8_WSCALES
#if F8_SPLIT
    #undef F8_WCODES
#endif

    [ForceUnroll]
    for (int i = 0; i < F8_FM; i++)
    {
#if DUAL
        [ForceUnroll]
        for (int j = 0; j < F8_FN; j += 2)
        {
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
                uint t = t0 + wm + i * 16 + (uint(index[e]) >> 4);
                uint n = n0 + wn / 2 + (j / 2) * 16 + (uint(index[e]) & 15);
#if SPLITK_FUSED
                if (t < p.T && n < N)
                {
                    yb[uint64_t(t) * 2 * N + n] = acc[i][j][e];
                    yb[uint64_t(t) * 2 * N + N + n] = acc[i][j + 1][e];
                }
#else
                if (t < p.T && n < N) yb[uint64_t(t) * Y_STRIDE + n] = epi2(acc[i][j][e], acc[i][j + 1][e]);
#endif
            }
        }
#else
        [ForceUnroll]
        for (int j = 0; j < F8_FN; j++)
        {
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
                uint t = t0 + wm + i * 16 + (uint(index[e]) >> 4);
                uint n = n0 + wn + j * 16 + (uint(index[e]) & 15);
#if SPLITK_FUSED
                if (t < p.T && n < N) yb[uint64_t(t) * N + n] = acc[i][j][e];
#else
                if (t < p.T && n < N) yb[uint64_t(t) * Y_STRIDE + n] = acc[i][j][e];
#endif
            }
        }
#endif
    }
#if SPLITK_FUSED && DUAL
    // Grid x = token tile, y = weight tile; the last workgroup of the tile reduces both products and applies epi2.
    SPLITK_FINISH_DUAL(p, tid, F8_THREADS, F8_BM, F8_BN_OUT, gid.x * ((N + F8_BN_OUT - 1) / F8_BN_OUT) + gid.y, t0, n0)
#elif SPLITK_FUSED
    // Grid x = token tile, y = weight tile; the last workgroup of the tile reduces it.
    SPLITK_FINISH(p, tid, F8_THREADS, F8_BM, F8_BN, gid.x * ((N + F8_BN - 1) / F8_BN) + gid.y, t0, n0)
#endif
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
