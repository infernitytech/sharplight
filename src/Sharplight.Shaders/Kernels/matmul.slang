// y[t, n] = sum_k x[t, k] * W[n, k]      (x: f32 rows with stride X_STRIDE, W: WFMT rows of K values)
// Specialization macros: WFMT, K, N, X_STRIDE, Y_STRIDE, ROW_BYTES, SG
#include "common.slang"
#include "splitk.slang"

// DUAL: two weight matrices of identical shape/format share x; the outputs are combined per element by
// `float epi2(float a, float b)` (defined by the including generated source, e.g. silu(a) * b): the matvecs and gemm_cm2 (X_F16, no K
// split). gemm_cm2's B tile alternates 16-row fragments of w and w2 (fragment 2i: rows of w, 2i + 1: the
// same rows of w2), so a workgroup covers GBN / 2 outputs and each subgroup holds both products of its outputs. With MOE (and
// X_SORTED), w and w2 are the expert matrices of a gate / up pair and the store applies MOE_EPI(m0, m1, e, n) instead, which may read
// per-expert rows of the tables declared by MOE_EPI_FIELDS (e.g. the experts' biases).
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
// MOE_BIAS (with MOE, single matrix; gemm_cm2 and gemm_sk): y gets product + moe_bias[expert, n], the routed experts' bias rows.
#ifndef MOE_BIAS
#define MOE_BIAS 0
#endif
#if MOE_BIAS
#define MOE_BIAS_OF(e, n) p.moe_bias[uint64_t(e) * N + (n)]
#else
#define MOE_BIAS_OF(e, n) 0.0
#endif
// MOE_EPI1(m, e, n, t, s) (with MOE, single matrix; defined by the including generated source): an elementwise group applied to
// the product m of expert e, output column n, token t and slot s in the store (it may read per-expert rows and per-slot scalars of
// the tables declared by MOE_EPI_FIELDS), e.g. the down projection's bias and router weight before the slot sum.
#ifdef MOE_EPI1
#define MOE_STORE1(v, e, n, item) MOE_EPI1(v, e, n, (item) / S, (item) % S)
#elif MOE_BIAS
#define MOE_STORE1(v, e, n, item) ((v) + MOE_BIAS_OF(e, n))
#else
#define MOE_STORE1(v, e, n, item) (v)
#endif
// MOE_TILE_MAJOR (with MOE): the grid's x runs over the expert tiles and y over the column tiles (default: the other way round).
#ifndef MOE_TILE_MAJOR
#define MOE_TILE_MAJOR 0
#endif
// MM_EPI (gemm_cm2 with GEMM_SPLIT and SPLITK 1, splitk_reduce): the including generated source defines
// `float mm_epi(float m0, float a0, float a1, float a2)`, an elementwise group applied to each result m0 in the store, which
// reads up to three extra tensors (Params e0 .. e2, access kinds MM_EPI_K0 .. MM_EPI_K2, see mm_epi_arg); y is the group's
// output, dense with Y_STRIDE = N.
#ifndef MM_EPI
#define MM_EPI 0
#endif
// X_SORTED (with MOE and X_F16): x holds the routed rows in expert-sorted order (moe_gather_x16 in moe.slang, X_STRIDE = K), so
// the rows of a tile are the dense rows rowOffset .. rowOffset + rows - 1. With X_RS, x is row-ranged (common.slang,
// f16_range_scale; scale word of sorted row o at byte offset X_RS + 4 o) and the store multiplies each row by its inverse scale.
#ifndef X_SORTED
#define X_SORTED 0
#endif
#if MOE && X_SORTED && defined(X_RS)
#define MOE_RANGED 1
#else
#define MOE_RANGED 0
#endif
// DUAL_Y16 (dual gemm_cm2): also stores y rounded to f16 into y16, the f16 copy the next GEMM reads (the to_f16 conversion,
// round to nearest even, so the copy is identical to a separate conversion). With DUAL_Y16_RANGE the copy is row-ranged
// (common.slang, f16_range_scale): values from 2^15 on raise the row's flag in y16_flags to the bits of their magnitude (atomic
// max, per subgroup tile), and the f16_range_fixup launch after this one converts those rows again with their range scale and
// writes every scale word.
#ifndef DUAL_Y16
#define DUAL_Y16 0
#endif
#ifndef DUAL_Y16_RANGE
#define DUAL_Y16_RANGE 0
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
#if MOE && MOE_BIAS
    float* moe_bias;   // [E, N]: the bias row of each tile's expert is added in the store
#endif
#if MOE && defined(MOE_EPI_FIELDS)
    MOE_EPI_FIELDS   // tables the expert epilogue reads (e.g. the experts' biases, the router weights)
#endif
#if DUAL_Y16
    uint8_t* y16;   // the f16 copy of y (dense rows of N halves)
#endif
#if DUAL_Y16_RANGE
    uint* y16_flags;   // per row: 0, or the bits of the largest magnitude from 2^15 on (f16_range_fixup clears them)
#endif
#if GEMM_SPLIT
    float* ws;   // per weight row: the power-of-two scale applied before splitting (split_scale)
#endif
#if BIAS
    float* bias;
#endif
#if MM_EPI && MM_EPI_ARGS > 0
    float* e0;   // the epilogue's tensors (MM_EPI_ARGS of them)
#endif
#if MM_EPI && MM_EPI_ARGS > 1
    float* e1;
#endif
#if MM_EPI && MM_EPI_ARGS > 2
    float* e2;
#endif
#if SPLITK_FUSED || MULTI > 1
    float* yout;      // gemm_cm2 with SPLITK_FUSED (splitk.slang): the reduced output (y holds the partial slices)
#endif
#if SPLITK_FUSED
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

#include "mm_epi.slang"

// MV_FMA_CHAIN (row-invariant compiles): the products of one 8-weight group in a fixed order of fused multiply-adds, so
// that every compiled variant (any MV_COLS) rounds a row's sum the same way. Otherwise the plain sum, which the driver
// contracts per variant (on the NVIDIA driver mostly, but not always, into this same chain). The dual matvecs keep the
// plain sum, which the driver contracts alike in all of their variants.
float dot8(float w[8], float4 a, float4 b)
{
#if DUAL || !defined(MV_FMA_CHAIN)
    return w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
#else
    float s = w[0] * a.x;
    s = fma(w[1], a.y, s);
    s = fma(w[2], a.z, s);
    s = fma(w[3], a.w, s);
    s = fma(w[4], b.x, s);
    s = fma(w[5], b.y, s);
    s = fma(w[6], b.z, s);
    return fma(w[7], b.w, s);
#endif
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
#if DUAL && !MOE   // (expert-grouped dual kernels store MOE_EPI instead)
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
// MV_SK_ROWS: output rows per workgroup. Each thread applies every activation value it loads to that many weight rows, so x is
// read from cache once per MV_SK_ROWS rows (in f32 it is up to four times the bytes of the weights it multiplies), and every
// thread keeps MV_SK_ROWS weight loads in flight. A row's sum is the same at every MV_SK_ROWS: thread tid adds the groups tid,
// tid + 256, ... in order, then the subgroup and workgroup sums run in the same order.
#ifndef MV_SK_ROWS
#define MV_SK_ROWS 1
#endif
groupshared float skPart[MV_SK_ROWS][MV_COLS][MV_SK_SG];
#if DUAL
groupshared float skPart2[MV_SK_ROWS][MV_COLS][MV_SK_SG];
#endif

[shader("compute")]
[numthreads(SG * MV_SK_SG, 1, 1)]
void matvec_sk(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    GRID_BIND(p);
    uint tid = ltid.x, lane = tid % SG, sg = tid / SG;
    uint row0 = gid.x * MV_SK_ROWS;
    float acc[MV_SK_ROWS][MV_COLS];
    [ForceUnroll] for (int r = 0; r < MV_SK_ROWS; r++) [ForceUnroll] for (int c = 0; c < MV_COLS; c++) acc[r][c] = 0;
#if DUAL
    float acc2[MV_SK_ROWS][MV_COLS];
    [ForceUnroll] for (int r = 0; r < MV_SK_ROWS; r++) [ForceUnroll] for (int c = 0; c < MV_COLS; c++) acc2[r][c] = 0;
#endif
    [unroll(MV_UNROLL)]
    for (uint g = tid; g < K / 8; g += SG * MV_SK_SG)
    {
        float w[MV_SK_ROWS][8];
#if DUAL
        float w2[MV_SK_ROWS][8];
#endif
        [ForceUnroll]
        for (int r = 0; r < MV_SK_ROWS; r++)
        {
            // Rows past N (N % MV_SK_ROWS != 0) read the last row; their sums are not stored.
            uint row = MV_SK_ROWS == 1 || N % MV_SK_ROWS == 0 ? row0 + r : min(row0 + r, uint(N) - 1);
            load8(p.w, ROW_BYTES, row, g, w[r]);
#if DUAL
            load8(p.w2, ROW_BYTES, row, g, w2[r]);
#endif
        }
        [ForceUnroll]
        for (int c = 0; c < MV_COLS; c++)
        {
            if (c < p.T)
            {
                float4* xp = (float4*)(p.x + uint64_t(c) * X_STRIDE + g * 8);
                float4 a = xp[0], b = xp[1];
                [ForceUnroll]
                for (int r = 0; r < MV_SK_ROWS; r++)
                {
                    acc[r][c] += dot8(w[r], a, b);
#if DUAL
                    acc2[r][c] += dot8(w2[r], a, b);
#endif
                }
            }
        }
    }
    [ForceUnroll]
    for (int r = 0; r < MV_SK_ROWS; r++)
    {
        [ForceUnroll]
        for (int c = 0; c < MV_COLS; c++)
        {
            float s = WaveActiveSum(acc[r][c]);
            if (lane == 0) skPart[r][c][sg] = s;
#if DUAL
            float s2 = WaveActiveSum(acc2[r][c]);
            if (lane == 0) skPart2[r][c][sg] = s2;
#endif
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid < MV_SK_ROWS * MV_COLS)
    {
        uint r = tid / MV_COLS, c = tid % MV_COLS, row = row0 + r;
        if (c < p.T && row < N)
        {
            float s = 0;
            [ForceUnroll] for (int i = 0; i < MV_SK_SG; i++) s += skPart[r][c][i];
#if DUAL && !MOE
            float s2 = 0;
            [ForceUnroll] for (int i = 0; i < MV_SK_SG; i++) s2 += skPart2[r][c][i];
            s = epi2(s, s2);
#endif
#if BIAS
            s = s + p.bias[row];
#endif
            p.y[uint64_t(c) * Y_STRIDE + row] = s;
        }
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

#ifdef KERNEL_GEMM_WIDE
// ---------------------------------------------------------------------------------------------
// Wide SIMT GEMM for large f32 grids: 128 tokens x 128 weight rows per workgroup of 256 threads, 8 x 8 outputs per thread
// (two 4-wide strips per axis, 64 apart, so every shared read is a conflict-free float4), K in steps of WBK = 16 through
// two shared buffers: the next step's global loads are in registers while this step's products run, one barrier per step.
// GEMM_SPLITK > 1 as in gemm. Needs K % 8 == 0 (load8 groups).
#define WBM 128
#define WBN 128
#define WBK 16

groupshared float4 wX[2][WBK][WBM / 4];
groupshared float4 wW[2][WBK][WBN / 4];

#if WFMT == FMT_F32
// f32 weights: thread loads float4 j of the step for tile rows tid / 4 + 64 j, K quad tid % 4: four threads read one row's
// 64 contiguous bytes, so a warp touches 8 rows per load instead of 32.
void wide_fetch(Params p, uint t0, uint n0, uint k0, uint tid, out float xv[8], out float wv[8])
{
    uint q = tid % 4;
    [ForceUnroll]
    for (uint j = 0; j < 2; j++)
    {
        uint r = tid / 4 + 64 * j;
        float4 a = 0, b = 0;
        if (t0 + r < p.T && k0 + q * 4 < K) a = *(float4*)(p.x + uint64_t(t0 + r) * X_STRIDE + k0 + q * 4);
        if (n0 + r < N && k0 + q * 4 < K) b = *(float4*)((float*)p.w + uint64_t(n0 + r) * K + k0 + q * 4);
        xv[j * 4 + 0] = a.x; xv[j * 4 + 1] = a.y; xv[j * 4 + 2] = a.z; xv[j * 4 + 3] = a.w;
        wv[j * 4 + 0] = b.x; wv[j * 4 + 1] = b.y; wv[j * 4 + 2] = b.z; wv[j * 4 + 3] = b.w;
    }
}

void wide_stage(uint buf, uint tid, float xv[8], float wv[8])
{
    uint q = tid % 4;
    [ForceUnroll]
    for (uint j = 0; j < 2; j++)
    {
        uint r = tid / 4 + 64 * j;
        [ForceUnroll] for (uint c = 0; c < 4; c++) { wX[buf][q * 4 + c][r >> 2][r & 3] = xv[j * 4 + c]; wW[buf][q * 4 + c][r >> 2][r & 3] = wv[j * 4 + c]; }
    }
}
#else
// Other weight formats: thread decodes 8 values (load8 group tid / 128) of tile row tid % 128.
void wide_fetch(Params p, uint t0, uint n0, uint k0, uint tid, out float xv[8], out float wv[8])
{
    uint r = tid % 128, gq = tid / 128;
    uint t = t0 + r;
    [ForceUnroll] for (int i = 0; i < 8; i++) { xv[i] = 0; wv[i] = 0; }
    if (t < p.T && k0 / 8 + gq < K / 8)
    {
        float4* xp = (float4*)(p.x + uint64_t(t) * X_STRIDE + k0 + gq * 8);
        float4 a = xp[0], b = xp[1];
        xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w; xv[4] = b.x; xv[5] = b.y; xv[6] = b.z; xv[7] = b.w;
    }
    if (n0 + r < N && k0 / 8 + gq < K / 8) load8(p.w, ROW_BYTES, n0 + r, k0 / 8 + gq, wv);
}

void wide_stage(uint buf, uint tid, float xv[8], float wv[8])
{
    uint r = tid % 128, gq = tid / 128;
    [ForceUnroll] for (int i = 0; i < 8; i++) { wX[buf][gq * 8 + i][r >> 2][r & 3] = xv[i]; wW[buf][gq * 8 + i][r >> 2][r & 3] = wv[i]; }
}
#endif

[shader("compute")]
[numthreads(256, 1, 1)]
void gemm_wide(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    GRID_BIND(p);
    uint tid = ltid.x;
    uint n0 = gid.x * WBN;
    uint t0 = gid.y * WBM;
    uint tx = tid % 16, ty = tid / 16;
#ifndef GEMM_SPLITK
#define GEMM_SPLITK 1
#endif
#if GEMM_SPLITK > 1
    const uint kShare = ((K + WBK - 1) / WBK + GEMM_SPLITK - 1) / GEMM_SPLITK * WBK;
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
    float acc[8][8];
    [ForceUnroll] for (int i = 0; i < 8; i++) [ForceUnroll] for (int j = 0; j < 8; j++) acc[i][j] = 0;

    float xv[8], wv[8];
    if (kBegin < kEnd)
    {
        wide_fetch(p, t0, n0, kBegin, tid, xv, wv);
        wide_stage(0, tid, xv, wv);
    }
    GroupMemoryBarrierWithGroupSync();
    // Two K steps per iteration so both shared buffers are addressed with constants.
#define WIDE_STEP(CUR)                                                                                       \
    {                                                                                                        \
        bool more = k0 + WBK < kEnd;                                                                         \
        if (more) { wide_fetch(p, t0, n0, k0 + WBK, tid, xv, wv); }                                          \
        [ForceUnroll]                                                                                        \
        for (int kk = 0; kk < WBK; kk++)                                                                     \
        {                                                                                                    \
            float4 a0 = wX[CUR][kk][ty], a1 = wX[CUR][kk][16 + ty];                                          \
            float4 b0 = wW[CUR][kk][tx], b1 = wW[CUR][kk][16 + tx];                                          \
            float a[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };                                 \
            float b[8] = { b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w };                                 \
            [ForceUnroll] for (int i = 0; i < 8; i++) [ForceUnroll] for (int j = 0; j < 8; j++) acc[i][j] += a[i] * b[j]; \
        }                                                                                                    \
        if (more) { wide_stage(1 - CUR, tid, xv, wv); }                                                      \
        GroupMemoryBarrierWithGroupSync();                                                                   \
    }
    for (uint k0 = kBegin; k0 < kEnd; k0 += WBK)
    {
        WIDE_STEP(0)
        k0 += WBK;
        if (k0 >= kEnd)
        {
            break;
        }
        WIDE_STEP(1)
    }
#undef WIDE_STEP

    [ForceUnroll]
    for (int i = 0; i < 8; i++)
    {
        uint t = t0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
        if (t >= p.T) continue;
        [ForceUnroll]
        for (int j = 0; j < 8; j++)
        {
            uint n = n0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);
            if (n < N) yb[uint64_t(t) * yStride + n] = acc[i][j];
        }
    }
}
#endif

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
#if SL_TARGET_CUDA
#include "cuda_mma.slang"
typealias MatA = CuA16;
typealias MatB = CuB16;
typealias Acc = CuAccF;
#else
typealias MatA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias Acc = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
#endif

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

#ifdef KERNEL_GEMM_SG
// ---------------------------------------------------------------------------------------------
// GEMM on 8x8 subgroup matrices (DeviceFeatures.MatrixF16Small: Metal simdgroup_matrix, f16 x f16 + f32). The tile and staging of
// gemm_cm: a workgroup of 8 subgroups computes 64 tokens x 128 rows, subgroup (sm, sn) owns a 32x32 block = 4x4 accumulators;
// x (f32) and W (any WFMT, dequantized via load8) are staged into shared memory as f16 per K step of 32. Requires K % 32 == 0;
// partial N / T tiles are handled (zero-filled loads, staged stores). GEMM_SPLITK > 1: workgroup z sums only its share of K into
// partials [GEMM_SPLITK][T][N] (see splitk_reduce).
using namespace linalg;
#ifndef GEMM_SPLITK
#define GEMM_SPLITK 1
#endif
#define SBM 64
#define SBN 128
#define SBK 32
#define SGS (SBK + 8)
typealias SgA = CoopMat<half, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixA>;
typealias SgB = CoopMat<half, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixB>;
typealias SgC = CoopMat<float, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixAccumulator>;

groupshared half sgA[SBM * SGS];
groupshared half sgB[SBN * SGS];
groupshared float sgC[8 * 64];

[shader("compute")]
[numthreads(256, 1, 1)]
void gemm_sg(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    GRID_BIND(p);
    uint tid = ltid.x;
    uint sg = tid / 32;
    uint lane = tid % 32;
    uint n0 = gid.x * SBN;
    uint t0 = gid.y * SBM;
    uint wm = (sg / 4) * 32;
    uint wn = (sg % 4) * 32;

    SgC acc[4][4];
    [ForceUnroll] for (int i = 0; i < 4; i++) [ForceUnroll] for (int j = 0; j < 4; j++) acc[i][j] = SgC(0.0);

#if GEMM_SPLITK > 1
    const uint kShare = ((K + SBK - 1) / SBK + GEMM_SPLITK - 1) / GEMM_SPLITK * SBK;
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
    for (uint k0 = kBegin; k0 < kEnd; k0 += SBK)
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
            uint o = r * SGS + q * 8;
            sgA[o + 0] = half(a.x); sgA[o + 1] = half(a.y); sgA[o + 2] = half(a.z); sgA[o + 3] = half(a.w);
            sgA[o + 4] = half(b.x); sgA[o + 5] = half(b.y); sgA[o + 6] = half(b.z); sgA[o + 7] = half(b.w);
        }
        // W tile: 128 rows x 32 = 512 groups of 8, two per thread.
        [ForceUnroll]
        for (uint it = 0; it < 2; it++)
        {
            uint idx = tid + it * 256;
            uint r = idx / 4, q = idx % 4;
            float w[8];
            if (N % SBN == 0 || n0 + r < N) load8(p.w, ROW_BYTES, n0 + r, k0 / 8 + q, w);
            else { [ForceUnroll] for (int i = 0; i < 8; i++) w[i] = 0; }
            uint o = r * SGS + q * 8;
            [ForceUnroll] for (int i = 0; i < 8; i++) sgB[o + i] = half(w[i]);
        }
        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (uint kk = 0; kk < SBK; kk += 8)
        {
            SgA a[4];
            SgB b[4];
            [ForceUnroll] for (int i = 0; i < 4; i++) a[i] = SgA.Load<CoopMatMatrixLayout.RowMajor>(sgA, (wm + i * 8) * SGS + kk, SGS);
            [ForceUnroll] for (int j = 0; j < 4; j++) b[j] = SgB.Load<CoopMatMatrixLayout.ColumnMajor>(sgB, (wn + j * 8) * SGS + kk, SGS);
            [ForceUnroll] for (int i = 0; i < 4; i++) [ForceUnroll] for (int j = 0; j < 4; j++) acc[i][j] = coopMatMulAdd<float, false>(a[i], b[j], acc[i][j]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (t0 + SBM <= p.T && (N % SBN == 0 || n0 + SBN <= N))
    {
        float* y = yb + uint64_t(t0 + wm) * yStride + n0 + wn;
        [ForceUnroll] for (int i = 0; i < 4; i++) [ForceUnroll] for (int j = 0; j < 4; j++) acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(y + uint64_t(i * 8) * yStride + j * 8, 0u, yStride);
        return;
    }
    // Partial tile: stage each accumulator through the subgroup's shared slice and store guarded (two values per lane).
    [ForceUnroll]
    for (int i = 0; i < 4; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < 4; j++)
        {
            acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(sgC, sg * 64, 8u);
            GroupMemoryBarrierWithGroupSync();
            [ForceUnroll]
            for (uint e = 0; e < 2; e++)
            {
                uint idx = lane + e * 32;
                uint t = t0 + wm + i * 8 + idx / 8;
                uint n = n0 + wn + j * 8 + idx % 8;
                if (t < p.T && n < N) yb[uint64_t(t) * yStride + n] = sgC[sg * 64 + idx];
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
#endif

#if defined(KERNEL_GEMM_CM2) || defined(KERNEL_GEMM_SK)
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
// GEMM_SPLIT (f32 weights): every value is split into f16 hi + lo parts and each product is computed as hi·hi + hi·lo + lo·hi
// (three tensor-core products). Each row of x and of W is first scaled by a power of two that puts its largest magnitude in
// [2^14, 2^15) (split_scale), so neither part over- or underflows f16 and the pair keeps 22-23 significant bits; the
// output undoes both scales. x arrives pre-split by split_rows (per row: K hi halves, K lo halves, the f32 inverse scale);
// W was split the same way on the host (p.w: per row K hi halves, then K lo halves; p.ws: the row scales). The products of each 16-value K slice
// accumulate from zero on the tensor cores and are then added to the f32 accumulators, so the tensor cores' truncating
// accumulation only rounds slice sums. Error against f64 is at or below the f32 SIMT GEMM's. Single buffered (GBK = 32).
#ifndef GBK
#define GBK 32
#endif
#ifndef GEMM_SPLIT
#define GEMM_SPLIT 0
#endif
// SPLIT_PARTS = 1 with GEMM_SPLIT (fast matrix precision): the row-scaled hi parts only, one product per multiply-add (11
// significant bits per input, like TF32) accumulated in f32 on the tensor cores; x rows from split_rows with SPLIT_PARTS 1
// (K hi halves, then the inverse scale), W per row K hi halves. Double buffered like the f16 GEMM.
#ifndef SPLIT_PARTS
#define SPLIT_PARTS 3
#endif
#if GEMM_SPLIT && SPLIT_PARTS > 1
#define SPLIT_LO 1
#define SPLIT_HALVES 2
#else
#define SPLIT_LO 0
#define SPLIT_HALVES 1
#endif
// GNBUF (macro): double-buffer the GBK = 64 tiles too where the shared memory allows it (CUDA's opt-in shared memory).
#ifndef GNBUF
#if GBK == 32 && !SPLIT_LO
#define GNBUF 2
#else
#define GNBUF 1
#endif
#endif
#define GQ (GBK / 8)
#define GAS (GBK + 8)
#define GA_PER_T (GBM * GBK / 8 / GTHREADS)
// Weights are fetched 16 values per item, or (MXFP4R, when the threads divide the tile evenly) one whole 32-value block per item,
// so that a block's 16 code bytes and its scale are loaded once.
#if WFMT == FMT_MXFP4R && (GBN * GBK / 32) % GTHREADS == 0
#define GW_UNIT 32
#else
#define GW_UNIT 16
#endif
#define GB_PER_T (GBN * GBK / GW_UNIT / GTHREADS)
// Output columns per workgroup.
#if DUAL
#define GBN_OUT (GBN / 2)
#else
#define GBN_OUT GBN
#endif
#if SL_TARGET_CUDA
#include "cuda_mma.slang"
typealias MatA2 = CuA16;
typealias MatB2 = CuB16;
typealias Acc2 = CuAccF;
typealias AccH2 = CuAccH;
#else
typealias MatA2 = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB2 = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias Acc2 = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
typealias AccH2 = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
#endif

#ifdef KERNEL_GEMM_CM2
// Tiles hold packed halves, 8 per uint4, so staging is one 16-byte store per 8 values (offsets/strides in uint4s).
groupshared uint4 gA[GNBUF * GBM * GAS / 8];
groupshared uint4 gB[GNBUF * GBN * GAS / 8];
#if SPLIT_LO
groupshared uint4 gAl[GBM * GAS / 8];
groupshared uint4 gBl[GBN * GAS / 8];
#endif
#endif

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
    uint4 q = LOAD_U4_ALIGNED(r + ROW_BYTES / 17 + uint64_t(g16) * 16, Q8R_ALIGN);
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
    // The block header (f16 d, f16 dmin, 12 scale bytes) as one aligned 16-byte load instead of a word and 2-3 byte loads.
    uint4 head = *(uint4*)blk;
    uint4 q = (*(uint4*)(blk + (KQ_BLOCK - 128 + 32 * j + l0)) >> (4 * upper)) & 0x0F0F0F0Fu;
#if WFMT == FMT_Q5_K
    q |= ((*(uint4*)(blk + 16 + l0) >> (2 * j + upper)) & 0x01010101u) << 4;
#endif
#undef KQ_BLOCK
    float sc, mn;
    scale_min_k4_words(uint3(head.y, head.z, head.w), 2 * j + upper, sc, mn);
    uint dm = head.x;
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
#elif WFMT == FMT_Q4Z_G32 || WFMT == FMT_Q4Z_G128
    // { f16 d; f16 z; u8 qs[16 * runs] } (quant_group.slang), each 32-value run laid out like Q4_0. The zero point is an
    // integer, so 1024 + z is exact in half and (1024 + q) - (1024 + z) = q - z is exact: one rounding, at * d.
#if WFMT == FMT_Q4Z_G32
    uint64_t at = uint64_t(rowStart(w, ROW_BYTES, row)) + uint64_t(g16 >> 1) * 20;
    uint4 q = (load16_any(at + 4) >> (4 * (g16 & 1))) & 0x0F0F0F0Fu;
#else
    uint64_t at = uint64_t(rowStart(w, ROW_BYTES, row)) + uint64_t(g16 >> 3) * 68;
    uint4 q = (load16_any(at + 4 + ((g16 >> 1) & 3) * 16) >> (4 * (g16 & 1))) & 0x0F0F0F0Fu;
#endif
    uint dz = *(uint*)at;
    half2 d = bit_cast<half2>((dz & 0xFFFFu) * 0x10001u);
    half offset = 1024.0h + bit_cast<half2>(dz).y;
    half2 z = half2(0.0h, 0.0h);
    lo = uint4(codes_h2(q.x, false, offset, d, z), codes_h2(q.x, true, offset, d, z), codes_h2(q.y, false, offset, d, z), codes_h2(q.y, true, offset, d, z));
    hi = uint4(codes_h2(q.z, false, offset, d, z), codes_h2(q.z, true, offset, d, z), codes_h2(q.w, false, offset, d, z), codes_h2(q.w, true, offset, d, z));
#elif WFMT == FMT_NVFP4 || WFMT == FMT_NVFP4_GS
    // One 16-value sub-block: 8 code bytes (value i in the low nibble of byte i, i + 8 in the high one) and its UE4M3 scale,
    // NVFP4_GS times the block's f32 global scale. Doubled E2M1 codes through half bits as MXFP4 below, the block scale
    // split in two factors when it leaves half's range.
#if WFMT == FMT_NVFP4
    uint64_t at = uint64_t(rowStart(w, ROW_BYTES, row)) + uint64_t(g16 >> 2) * 36;
    float sc = ue4m3_half(uint(*(uint8_t*)(at + (g16 & 3)))) * 32768.0;
    uint64_t codes = at + 4 + 8 * (g16 & 3);
#else
    uint64_t at = uint64_t(rowStart(w, ROW_BYTES, row)) + uint64_t(g16 >> 2) * 40;
    float sc = ue4m3_half(uint(*(uint8_t*)(at + 4 + (g16 & 3)))) * asfloat(*(uint*)at) * 32768.0;
    uint64_t codes = at + 8 + 8 * (g16 & 3);
#endif
    uint c0 = *(uint*)codes;
    uint c1 = *(uint*)(codes + 4);
    uint4 q = uint4(c0 & 0x0F0F0F0Fu, c1 & 0x0F0F0F0Fu, (c0 >> 4) & 0x0F0F0F0Fu, (c1 >> 4) & 0x0F0F0F0Fu);
    half2 s1 = half2(half(min(sc, 32768.0)));
    half2 s2 = half2(half(max(sc / 32768.0, 1.0)));
    lo = uint4(fp4_h2(q.x, false, s1, s2), fp4_h2(q.x, true, s1, s2), fp4_h2(q.y, false, s1, s2), fp4_h2(q.y, true, s1, s2));
    hi = uint4(fp4_h2(q.z, false, s1, s2), fp4_h2(q.z, true, s1, s2), fp4_h2(q.w, false, s1, s2), fp4_h2(q.w, true, s1, s2));
#elif WFMT == FMT_MXFP4R
    // As MXFP4 below, from the aligned split layout (common.slang): one aligned 16-byte load holds both nibble halves of the block.
    uint4 q = (*(uint4*)(mxfp4r_codes(w, ROW_BYTES, row) + uint64_t(g16 >> 1) * 16) >> (4 * (g16 & 1))) & 0x0F0F0F0Fu;
    float s = e8m0_half(mxfp4r_scale(w, ROW_BYTES, row, g16 >> 1)) * 32768.0;
    half2 s1 = half2(half(min(s, 32768.0)));
    half2 s2 = half2(half(max(s / 32768.0, 1.0)));
    lo = uint4(fp4_h2(q.x, false, s1, s2), fp4_h2(q.x, true, s1, s2), fp4_h2(q.y, false, s1, s2), fp4_h2(q.y, true, s1, s2));
    hi = uint4(fp4_h2(q.z, false, s1, s2), fp4_h2(q.z, true, s1, s2), fp4_h2(q.w, false, s1, s2), fp4_h2(q.w, true, s1, s2));
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
#elif WFMT == FMT_F16
    uint4* p16 = (uint4*)(rowStart(w, ROW_BYTES, row) + uint64_t(g16) * 32);
    lo = p16[0];
    hi = p16[1];
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
    uint2 q = LOAD_U2_ALIGNED(r + ROW_BYTES / 17 + uint64_t(g) * 8, Q8R_ALIGN);
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
// 8 bf16 values (packed pairs) as packed halves, the conversion of load8h.
uint4 bf16x8_h(uint4 u)
{
    return uint4(f32tof16(asfloat(u.x << 16)) | (f32tof16(asfloat(u.x & 0xFFFF0000u)) << 16),
                 f32tof16(asfloat(u.y << 16)) | (f32tof16(asfloat(u.y & 0xFFFF0000u)) << 16),
                 f32tof16(asfloat(u.z << 16)) | (f32tof16(asfloat(u.z & 0xFFFF0000u)) << 16),
                 f32tof16(asfloat(u.w << 16)) | (f32tof16(asfloat(u.w & 0xFFFF0000u)) << 16));
}

// Raw weight fetches of gemm_cm2. FETCH only issues the loads of a unit (GW_UNIT weights of one row) into GW_RAW registers;
// the unit is decoded to packed halves when it is staged, after the MMAs of the current step, so the loads stay in flight
// during the MMAs instead of stalling the warp at the decode. The row's base addresses are computed once (WRow). The decode
// is the arithmetic of load16h, so the staged halves are identical. Formats without a raw unit decode at fetch (load16h):
// for the K-quants and Q8_0 the raw units (9-13 registers per 16 values) measured no faster than decoding at fetch.
struct WRow { uint64_t a; uint64_t b; };
#if GW_UNIT == 32
// MXFP4R: one block per unit, [16 code bytes][scale byte].
#define GW_RAW 5
WRow wrow(uint8_t* w, uint row)
{
    WRow r;
    r.a = mxfp4r_codes(w, ROW_BYTES, row);
    r.b = uint64_t(w) + uint64_t(row >> 4) * 16 * ROW_BYTES + 256 * (ROW_BYTES / 17) + (row & 15);
    return r;
}
void wload(WRow r, uint g32, out uint raw[GW_RAW])
{
    uint4 c = *(uint4*)(r.a + uint64_t(g32) * 16);
    raw[0] = c.x; raw[1] = c.y; raw[2] = c.z; raw[3] = c.w;
    raw[4] = uint(*(uint8_t*)(r.b + uint64_t(g32) * 16));
}
void wdecode(uint raw[GW_RAW], out uint4 o[4])
{
    uint4 c = uint4(raw[0], raw[1], raw[2], raw[3]);
    float s = e8m0_half(raw[4]) * 32768.0;
    half2 s1 = half2(half(min(s, 32768.0)));
    half2 s2 = half2(half(max(s / 32768.0, 1.0)));
    [ForceUnroll]
    for (uint h = 0; h < 2; h++)
    {
        uint4 q = (c >> (4 * h)) & 0x0F0F0F0Fu;
        o[2 * h] = uint4(fp4_h2(q.x, false, s1, s2), fp4_h2(q.x, true, s1, s2), fp4_h2(q.y, false, s1, s2), fp4_h2(q.y, true, s1, s2));
        o[2 * h + 1] = uint4(fp4_h2(q.z, false, s1, s2), fp4_h2(q.z, true, s1, s2), fp4_h2(q.w, false, s1, s2), fp4_h2(q.w, true, s1, s2));
    }
}
#endif

#ifdef KERNEL_GEMM_CM2
groupshared float gC[WARPS_M * WARPS_N * 256];
#if MOE
groupshared uint gItem[GBM];
#endif
#if MOE_RANGED
groupshared float gRowInverse[GBM];
#endif

#if SPLITK_FUSED
#if MOE || DUAL || GEMM_SPLIT || !defined(SPLITK) || SPLITK < 2 || (MULTI > 1 && MM_EPI)
#error "the fused split-K reduction of gemm_cm2 needs SPLITK > 1, no MOE / DUAL / GEMM_SPLIT, and MULTI without MM_EPI"
#endif
#if MULTI > 1
#define SPLITK_STORE(p, s, t, n) MULTI_STORE(p, s, t, n)
#else
#define SPLITK_STORE(p, s, t, n) { p.yout[uint64_t(t) * Y_STRIDE + (n)] = MM_STORE(s, t, n); }
#endif
#ifdef X_RS
#define SPLITK_ROW_SCALE(p, t) f16_range_inverse(*(uint*)(uint64_t(p.x) + X_RS + uint64_t(t) * 4))
#else
#define SPLITK_ROW_SCALE(p, t) 1.0
#endif
#endif

[shader("compute")]
[numthreads(GTHREADS, 1, 1)]
void gemm_cm2(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    GRID_BIND(p);
    uint tid = ltid.x;
    uint sg = tid / 32, lane = tid % 32;
    // GEMM_GROUP: grouped tile order (common.slang) for launches without K splits.
#if DUAL && (GEMM_SPLIT || MM_EPI || !X_F16 || FRAG_N % 2 != 0 || (defined(SPLITK) && SPLITK != 1) || (MOE && (!X_SORTED || DUAL_Y16)))
#error "the dual gemm_cm2 needs f16 activations, an even FRAG_N and no K split (with MOE: expert-sorted rows, no f16 copy)"
#endif
#if !MOE && defined(GEMM_GROUP) && (!defined(SPLITK) || SPLITK == 1)
    gid.xy = grouped_tile(gid.xy, (N + GBN_OUT - 1) / GBN_OUT, (p.T + GBM - 1) / GBM, GEMM_GROUP);
#endif
#if MOE && MOE_TILE_MAJOR
    gid.xy = gid.yx;   // expert tiles along gid.x (see gemm_sk)
#endif
    uint n0 = gid.x * GBN_OUT;
#if MOE
    // The tile's expert and items; t indexes the tile's rows (0 .. rows - 1) instead of tokens.
    uint tile = gid.y;
    if (tile >= p.scratch[0]) return;
    uint expert = p.scratch[1 + 3 * tile], rowOff = p.scratch[2 + 3 * tile];
    uint tileRows = p.scratch[3 + 3 * tile];
#ifdef MOE_MIN_ROWS
    if (tileRows < MOE_MIN_ROWS) return;   // left to the half-height launch (MOE_MAX_ROWS)
#endif
#ifdef MOE_MAX_ROWS
    if (tileRows > MOE_MAX_ROWS) return;
#endif
    uint8_t* wbase = rowStart(p.w, EXPERT_BYTES, expert);
#if DUAL
    uint8_t* wbase2 = rowStart(p.w2, EXPERT_BYTES, expert);
#endif
    for (uint r = tid; r < GBM; r += GTHREADS) gItem[r] = r < tileRows ? p.scratch[1 + 3 * MAXTILES + rowOff + r] : 0xFFFFFFFFu;
#if MOE_RANGED
    for (uint r = tid; r < GBM; r += GTHREADS) gRowInverse[r] = r < tileRows ? f16_range_inverse(*(uint*)(uint64_t(p.x) + X_RS + uint64_t(rowOff + r) * 4)) : 1.0;
#endif
    GroupMemoryBarrierWithGroupSync();
    uint t0 = 0;
#if X_SORTED
    #define ROW_OF(t) (rowOff + (t))
#else
    #define ROW_OF(t) (PER_SLOT ? gItem[t] : gItem[t] / S)
#endif
    #define ROWS_LIVE tileRows
#else
    uint t0 = gid.y * GBM;
#if MULTI > 1
    uint8_t* wbase = MULTI_WBASE(p, n0);
#else
    uint8_t* wbase = p.w;
#endif
#if DUAL
    uint8_t* wbase2 = p.w2;
#endif
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

    // GPF register sets of prefetched operands: with 2 (single-buffered tiles with raw weight fetches), the loads of step s + 2 are
    // issued before the MMAs of step s and staged after those of step s + 1, so they have two steps to arrive.
#if GNBUF == 1 && defined(GW_RAW) && !defined(GPF)
#define GPF 2
#endif
#ifndef GPF
#define GPF 1
#endif
    uint4 xs[GPF][GA_PER_T];
#ifdef GW_RAW
    uint ws[GPF][GB_PER_T][GW_RAW];
#else
    uint4 ws[GPF][GB_PER_T][2];
#endif
#if SPLIT_LO
    uint4 xsl[GPF][GA_PER_T];   // lo parts (GEMM_SPLIT)
    uint4 wsl[GPF][GB_PER_T][2];
#endif

    // X_F16: activations were converted to dense f16 rows beforehand (to_f16), halving tile load traffic.
#if GEMM_SPLIT
    // Split mode: x rows from split_rows (SPLIT_ROW_BYTES each: K hi halves, K lo halves, the inverse scale); the lo parts go
    // to `lo` (xsl of FETCH's register set and item). Per-item row addresses as for X_F16.
    #define SPLIT_ROW_BYTES (2 * SPLIT_HALVES * uint64_t(K) + 16)
    uint64_t xrb[GA_PER_T];
    [ForceUnroll] for (uint it = 0; it < GA_PER_T; it++)
    {
        uint t = t0 + (tid + it * GTHREADS) / GQ;
        xrb[it] = uint64_t(p.x) + uint64_t(ROW_OF(t < ROWS_LIVE ? t : t0)) * SPLIT_ROW_BYTES + ((tid + it * GTHREADS) % GQ) * 16;
    }
#if SPLIT_LO
    #define GEMM_LOAD_X(t, k, dst, lo)                                                                   \
        { dst = 0; lo = 0; if ((t) < ROWS_LIVE) { dst = *(uint4*)(xrb[it] + uint64_t(k) * 2); lo = *(uint4*)(xrb[it] + (uint64_t(K) + (k)) * 2); } }
#else
    #define GEMM_LOAD_X(t, k, dst, lo)                                                                   \
        { dst = 0; if ((t) < ROWS_LIVE) dst = *(uint4*)(xrb[it] + uint64_t(k) * 2); }
#endif
#elif X_F16 && !MOE && GTHREADS % GQ == 0
    // Dense rows: item it of a thread reads row t0 + tid / GQ + it * (GTHREADS / GQ), so one base address and constant
    // offsets serve every item (rows past the tile are not loaded, so their addresses need no clamp).
    uint64_t xrb0 = uint64_t(p.x) + (uint64_t(t0 + tid / GQ) * X_STRIDE + (tid % GQ) * 8) * 2;
    #define GEMM_LOAD_X(t, k, dst, lo)                                                                   \
        { dst = 0; if ((t) < ROWS_LIVE) dst = *(uint4*)(xrb0 + (uint64_t(it) * (GTHREADS / GQ) * X_STRIDE + (k)) * 2); }
#elif X_F16
    // Per-item row addresses are loop invariant (the row is clamped; rows past the tile load zeros).
    uint64_t xrb[GA_PER_T];
    [ForceUnroll] for (uint it = 0; it < GA_PER_T; it++)
    {
        uint t = t0 + (tid + it * GTHREADS) / GQ;
        xrb[it] = uint64_t(p.x) + (uint64_t(ROW_OF(t < ROWS_LIVE ? t : t0)) * X_STRIDE + ((tid + it * GTHREADS) % GQ) * 8) * 2;
    }
    #define GEMM_LOAD_X(t, k, dst, lo)                                                                   \
        { dst = 0; if ((t) < ROWS_LIVE) dst = *(uint4*)(xrb[it] + uint64_t(k) * 2); }
#else
    #define GEMM_LOAD_X(t, k, dst, lo)                                                                   \
        { float4 a = 0, b = 0;                                                                          \
          if ((t) < ROWS_LIVE) { float4* xp = (float4*)(p.x + uint64_t(ROW_OF(t)) * X_STRIDE + (k)); a = xp[0]; b = xp[1]; } \
          dst = uint4(f32tof16(a.x) | (f32tof16(a.y) << 16), f32tof16(a.z) | (f32tof16(a.w) << 16),       \
                      f32tof16(b.x) | (f32tof16(b.y) << 16), f32tof16(b.z) | (f32tof16(b.w) << 16)); }
#endif
#ifdef GW_RAW
    WRow wr[GB_PER_T];
    [ForceUnroll] for (uint it = 0; it < GB_PER_T; it++)
    {
        uint idx = tid + it * GTHREADS;
#if DUAL
        // Tile row r: row n0 + (r / 32) * 16 + r % 16 of w (fragments 2i) or of w2 (fragments 2i + 1).
        uint r = idx / (GBK / GW_UNIT);
        wr[it] = wrow(((r >> 4) & 1) != 0 ? wbase2 : wbase, min(n0 + (r >> 5) * 16 + (r & 15), uint(N) - 1));
#else
        wr[it] = wrow(wbase, min(n0 + idx / (GBK / GW_UNIT), uint(N) - 1));
#endif
    }
    // Rows past N stage the decode of zero codes, which the stores never read back.
#if DUAL
    #define WLOAD(set, it, row, g)                                                                      \
        { uint tr = (row) - n0;                                                                         \
          [ForceUnroll] for (uint u = 0; u < GW_RAW; u++) ws[set][it][u] = 0;                             \
          if (N % GBN_OUT == 0 || n0 + (tr >> 5) * 16 + (tr & 15) < N) wload(wr[it], g, ws[set][it]); }
#else
    #define WLOAD(set, it, row, g) { [ForceUnroll] for (uint u = 0; u < GW_RAW; u++) ws[set][it][u] = 0; if (N % GBN == 0 || (row) < N) wload(wr[it], g, ws[set][it]); }
#endif
    #define WSTAGE(set, it, ob, row) { uint4 d[GW_UNIT / 8]; wdecode(ws[set][it], d); [ForceUnroll] for (uint u = 0; u < GW_UNIT / 8; u++) gB[(ob) + u] = d[u]; }
#elif GEMM_SPLIT && !SPLIT_LO
    #define WLOAD(set, it, row, g)                                                                      \
        { ws[set][it][0] = 0; ws[set][it][1] = 0;                                                       \
          if (N % GBN == 0 || (row) < N)                                                                \
          { uint4* hp = (uint4*)(uint64_t(wbase) + uint64_t(row) * 2 * uint64_t(K) + uint64_t(g) * 32);  \
            ws[set][it][0] = hp[0]; ws[set][it][1] = hp[1]; } }
    #define WSTAGE(set, it, ob, row) { gB[ob] = ws[set][it][0]; gB[(ob) + 1] = ws[set][it][1]; }
#elif GEMM_SPLIT
    #define WLOAD(set, it, row, g)                                                                      \
        { ws[set][it][0] = 0; ws[set][it][1] = 0; wsl[set][it][0] = 0; wsl[set][it][1] = 0;             \
          if (N % GBN == 0 || (row) < N)                                                                \
          { uint4* hp = (uint4*)(uint64_t(wbase) + uint64_t(row) * 4 * uint64_t(K) + uint64_t(g) * 32);  \
            uint4* lp = (uint4*)(uint64_t(hp) + 2 * uint64_t(K));                                       \
            ws[set][it][0] = hp[0]; ws[set][it][1] = hp[1]; wsl[set][it][0] = lp[0]; wsl[set][it][1] = lp[1]; } }
    #define WSTAGE(set, it, ob, row)                                                                    \
        { gB[ob] = ws[set][it][0]; gB[(ob) + 1] = ws[set][it][1]; gBl[ob] = wsl[set][it][0]; gBl[(ob) + 1] = wsl[set][it][1]; }
#elif WFMT == FMT_BF16 || WFMT == FMT_F16
    // 16-bit weights: the 16 raw values stay in the prefetch registers (as many as the decoded halves) and bf16 is converted
    // to halves when staged, after the MMAs, so the loads stay in flight during them (converted at the fetch, the warp waited
    // for its loads before the MMAs: bf16 [4096 → 12288] at 2048 rows 123 → 136 TFLOPS). Same conversion as load16h.
#if DUAL && (GTHREADS / (GBK / 16)) % 32 == 0
    // Item it of a thread reads tile row tid / (GBK / 16) + it * 32k: the same matrix (w or w2) for every item and weight rows
    // 16k apart, so one base address serves all items (dual bf16 [4096 → 12288] x2 at 2048 rows 130 → 143 TFLOPS with the above).
    uint dualTr0 = tid / (GBK / 16);
    uint dualRow0 = n0 + (dualTr0 >> 5) * 16 + (dualTr0 & 15);
    uint64_t dualBase = uint64_t(rowStart(((dualTr0 >> 4) & 1) != 0 ? wbase2 : wbase, ROW_BYTES, min(dualRow0, uint(N) - 1)));
    #define WLOAD(set, it, row, g)                                                                      \
        { uint wrow = dualRow0 + it * (GTHREADS / (GBK / 16)) / 2;                                      \
          ws[set][it][0] = 0; ws[set][it][1] = 0;                                                       \
          if (N % GBN_OUT == 0 || wrow < N)                                                             \
          { uint4* hp = (uint4*)(dualBase + uint64_t(it * (GTHREADS / (GBK / 16)) / 2) * ROW_BYTES + uint64_t(g) * 32); \
            ws[set][it][0] = hp[0]; ws[set][it][1] = hp[1]; } }
#elif DUAL
    #define WLOAD(set, it, row, g)                                                                      \
        { uint tr = (row) - n0, wrow = n0 + (tr >> 5) * 16 + (tr & 15);                                 \
          ws[set][it][0] = 0; ws[set][it][1] = 0;                                                       \
          if (N % GBN_OUT == 0 || wrow < N)                                                             \
          { uint4* hp = (uint4*)(rowStart(((tr >> 4) & 1) != 0 ? wbase2 : wbase, ROW_BYTES, wrow) + uint64_t(g) * 32); \
            ws[set][it][0] = hp[0]; ws[set][it][1] = hp[1]; } }
#else
    #define WLOAD(set, it, row, g)                                                                      \
        { ws[set][it][0] = 0; ws[set][it][1] = 0;                                                       \
          if (N % GBN == 0 || (row) < N)                                                                \
          { uint4* hp = (uint4*)(rowStart(wbase, ROW_BYTES, row) + uint64_t(g) * 32);                   \
            ws[set][it][0] = hp[0]; ws[set][it][1] = hp[1]; } }
#endif
#if WFMT == FMT_BF16
    #define WSTAGE(set, it, ob, row) { gB[ob] = bf16x8_h(ws[set][it][0]); gB[(ob) + 1] = bf16x8_h(ws[set][it][1]); }
#else
    #define WSTAGE(set, it, ob, row) { gB[ob] = ws[set][it][0]; gB[(ob) + 1] = ws[set][it][1]; }
#endif
#elif DUAL
    // B tile row r: row n0 + (r / 32) * 16 + r % 16 of w (fragments 2i) or of w2 (fragments 2i + 1); `row` is the tile row.
    #define WLOAD(set, it, row, g)                                                                      \
        { uint tr = (row) - n0, wrow = n0 + (tr >> 5) * 16 + (tr & 15);                                 \
          ws[set][it][0] = 0; ws[set][it][1] = 0;                                                       \
          if (N % GBN_OUT == 0 || wrow < N) load16h(((tr >> 4) & 1) != 0 ? wbase2 : wbase, wrow, g, ws[set][it][0], ws[set][it][1]); }
    #define WSTAGE(set, it, ob, row) { gB[ob] = ws[set][it][0]; gB[(ob) + 1] = ws[set][it][1]; }
#else
    #define WLOAD(set, it, row, g) { ws[set][it][0] = 0; ws[set][it][1] = 0; if (N % GBN == 0 || (row) < N) load16h(wbase, row, g, ws[set][it][0], ws[set][it][1]); }
    #define WSTAGE(set, it, ob, row) { gB[ob] = ws[set][it][0]; gB[(ob) + 1] = ws[set][it][1]; }
#endif
#if SPLIT_LO
    #define XS_LO(set, it) xsl[set][it]
    #define STAGE_LO_A(set, it, ia) gAl[ia] = xsl[set][it];
#else
    #define XS_LO(set, it) 0
    #define STAGE_LO_A(set, it, ia)
#endif
#if X_F16 || GEMM_SPLIT
    #define X_K(k0, q) (k0)
#else
    #define X_K(k0, q) ((k0) + (q) * 8)
#endif
    #define FETCH(set, k0)                                                                              \
    [ForceUnroll] for (uint it = 0; it < GA_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / GQ, q = idx % GQ;                                       \
        uint t = t0 + r;                                                                                \
        if (GEMM_EXP != 4 || (k0) == kBeg) GEMM_LOAD_X(t, X_K(k0, q), xs[set][it], XS_LO(set, it))      \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < GB_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / (GBK / GW_UNIT), q = idx % (GBK / GW_UNIT);           \
        if (GEMM_EXP != 3 || (k0) == kBeg) WLOAD(set, it, n0 + r, (k0) / GW_UNIT + q)                  \
    }
    #define STAGE(set, buf)                                                                             \
    [ForceUnroll] for (uint it = 0; it < GA_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / GQ, q = idx % GQ;                                       \
        uint ia = ((buf) * GBM * GAS + r * GAS + q * 8) / 8;                                            \
        gA[ia] = xs[set][it];                                                                           \
        STAGE_LO_A(set, it, ia)                                                                         \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < GB_PER_T; it++)                                                \
    {                                                                                                   \
        uint idx = tid + it * GTHREADS, r = idx / (GBK / GW_UNIT), q = idx % (GBK / GW_UNIT);           \
        uint ob = ((buf) * GBN * GAS + r * GAS + q * GW_UNIT) / 8;                                      \
        WSTAGE(set, it, ob, n0 + r)                                                                     \
    }
#if SPLIT_LO
    // The three products of a 16-value K slice (hi·lo, lo·hi, then hi·hi) accumulate on the tensor cores from zero, and the
    // slice sum is added to the f32 accumulator with an IEEE add: the tensor cores' accumulation (truncating alignment)
    // then only rounds 16-term slice sums instead of the whole running sum. GNBUF = 1: the tiles are at offset 0.
    #define COMPUTE(buf)                                                                                \
    {                                                                                                   \
        MatB2 b[2][FRAG_N];                                                                             \
        MatB2 bl[2][FRAG_N];                                                                            \
        [ForceUnroll] for (uint s = 0; s < 2; s++) [ForceUnroll] for (int j = 0; j < FRAG_N; j++)        \
        {                                                                                               \
            b[s][j] = MatB2.Load<CoopMatMatrixLayout.ColumnMajor>(gB, ((wn + j * 16) * GAS + s * 16) / 8, GAS / 8); \
            bl[s][j] = MatB2.Load<CoopMatMatrixLayout.ColumnMajor>(gBl, ((wn + j * 16) * GAS + s * 16) / 8, GAS / 8); \
        }                                                                                               \
        Acc2 t[FRAG_M][FRAG_N];                                                                         \
        [ForceUnroll] for (uint s = 0; s < 2; s++)                                                      \
        {                                                                                               \
            MatA2 a[FRAG_M];                                                                            \
            MatA2 al[FRAG_M];                                                                           \
            [ForceUnroll] for (int i = 0; i < FRAG_M; i++)                                              \
            {                                                                                           \
                a[i] = MatA2.Load<CoopMatMatrixLayout.RowMajor>(gA, ((wm + i * 16) * GAS + s * 16) / 8, GAS / 8); \
                al[i] = MatA2.Load<CoopMatMatrixLayout.RowMajor>(gAl, ((wm + i * 16) * GAS + s * 16) / 8, GAS / 8); \
            }                                                                                           \
            [ForceUnroll] for (int i = 0; i < FRAG_M; i++) [ForceUnroll] for (int j = 0; j < FRAG_N; j++) \
                t[i][j] = coopMatMulAdd<float, false>(al[i], b[s][j], s == 0 ? Acc2(0.0) : t[i][j]);    \
            [ForceUnroll] for (int i = 0; i < FRAG_M; i++) [ForceUnroll] for (int j = 0; j < FRAG_N; j++) \
                t[i][j] = coopMatMulAdd<float, false>(a[i], bl[s][j], t[i][j]);                         \
            [ForceUnroll] for (int i = 0; i < FRAG_M; i++) [ForceUnroll] for (int j = 0; j < FRAG_N; j++) \
                t[i][j] = coopMatMulAdd<float, false>(a[i], b[s][j], t[i][j]);                          \
        }                                                                                               \
        [ForceUnroll] for (int i = 0; i < FRAG_M; i++) [ForceUnroll] for (int j = 0; j < FRAG_N; j++) acc[i][j] = acc[i][j] + t[i][j]; \
    }
#else
    // The MMAs of the staged step in buffer `buf` (K order: 16-value slices in sequence).
    #define COMPUTE(buf)                                                                                \
    {                                                                                                   \
        uint ba = (buf) * GBM * GAS, bb = (buf) * GBN * GAS;                                            \
        [ForceUnroll]                                                                                   \
        for (uint kk = 0; kk < GBK; kk += 16)                                                           \
        {                                                                                               \
            MatB2 b[FRAG_N];                                                                            \
            [ForceUnroll] for (int j = 0; j < FRAG_N; j++) b[j] = MatB2.Load<CoopMatMatrixLayout.ColumnMajor>(gB, (bb + (wn + j * 16) * GAS + kk) / 8, GAS / 8); \
            [ForceUnroll]                                                                               \
            for (int i = 0; i < FRAG_M; i++)                                                            \
            {                                                                                           \
                MatA2 a = MatA2.Load<CoopMatMatrixLayout.RowMajor>(gA, (ba + (wm + i * 16) * GAS + kk) / 8, GAS / 8); \
                [ForceUnroll] for (int j = 0; j < FRAG_N; j++) acc[i][j] = coopMatMulAdd<float, false>(a, b[j], acc[i][j]); \
            }                                                                                           \
        }                                                                                               \
    }
#endif

    // GEMM_EXP (diagnostic, wrong results): 1 no loads after the first step, 2 no MMAs, 3 no weight loads, 4 no activation loads.
#ifndef GEMM_EXP
#define GEMM_EXP 0
#endif
#if GPF == 2 && GNBUF == 2
    // Double-buffered tiles with two register sets: entering step s, buffer s % 2 holds step s, register set (s + 1) % 2 the
    // loads of step s + 1 and set s % 2 those of step s + 2. Step s multiplies its buffer, stages step s + 1 into the other one
    // (last read by step s - 1, which every subgroup finished before the previous barrier), syncs once and issues the loads of
    // step s + 3 into the set it just staged, so every load has two steps of MMAs to arrive. The loop is unrolled by two to keep
    // the register sets static.
    uint steps = (kEnd - kBeg) / GBK;
    if (steps > 0)
    {
        FETCH(0, kBeg)
        STAGE(0, 0u)
    }
    if (steps > 1) { FETCH(1, kBeg + GBK) }
    if (steps > 2) { FETCH(0, kBeg + 2 * GBK) }
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 0; s < steps; s += 2)
    {
        COMPUTE(0u)
        if (s + 1 >= steps) break;
        STAGE(1, 1u)
        GroupMemoryBarrierWithGroupSync();
        if (s + 3 < steps) { FETCH(1, kBeg + (s + 3) * GBK) }
        COMPUTE(1u)
        if (s + 2 >= steps) break;
        STAGE(0, 0u)
        GroupMemoryBarrierWithGroupSync();
        if (s + 4 < steps) { FETCH(0, kBeg + (s + 4) * GBK) }
    }
#elif GPF == 2
    // Two register sets, the loop unrolled by two so that the sets stay static: entering step s, smem holds step s and set
    // (s + 1) % 2 the loads of step s + 1.
    uint steps = (kEnd - kBeg) / GBK;
    if (steps > 0)
    {
        FETCH(0, kBeg)
        STAGE(0, 0u)
    }
    if (steps > 1) { FETCH(1, kBeg + GBK) }
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 0; s < steps; s += 2)
    {
        if (s + 2 < steps) { FETCH(0, kBeg + (s + 2) * GBK) }
        COMPUTE(0u)
        GroupMemoryBarrierWithGroupSync();
        if (s + 1 >= steps) break;
        STAGE(1, 0u)
        GroupMemoryBarrierWithGroupSync();
        if (s + 3 < steps) { FETCH(1, kBeg + (s + 3) * GBK) }
        COMPUTE(0u)
        GroupMemoryBarrierWithGroupSync();
        if (s + 2 < steps)
        {
            STAGE(0, 0u)
            GroupMemoryBarrierWithGroupSync();
        }
    }
#elif GPF > 2 && GNBUF == 1
    // GPF register sets for a single-buffered tile: set j holds the loads of the steps j mod GPF, so the loads of step t + GPF
    // are issued right after step t + 1 was staged from the same set and have GPF - 1 steps of MMAs to arrive (more bytes in
    // flight for tiles with few weight units per thread). The loop is unrolled by GPF to keep the register sets static.
    uint steps = (kEnd - kBeg) / GBK;
    if (steps > 0)
    {
        FETCH(0, kBeg)
        STAGE(0, 0u)
    }
    [ForceUnroll] for (uint q = 1; q <= GPF; q++) { if (steps > q) { FETCH(q % GPF, kBeg + q * GBK) } }
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 0; s < steps; s += GPF)
    {
        [ForceUnroll]
        for (uint u = 0; u < GPF; u++)
        {
            uint t = s + u;
            if (t < steps)
            {
                COMPUTE(0u)
                GroupMemoryBarrierWithGroupSync();
                if (t + 1 < steps)
                {
                    STAGE((u + 1) % GPF, 0u)
                    GroupMemoryBarrierWithGroupSync();
                    if (t + 1 + GPF < steps) { FETCH((u + 1) % GPF, kBeg + (t + 1 + GPF) * GBK) }
                }
            }
        }
    }
#else
    // A split-K slice may be empty (K steps not divisible by SPLITK): never prefetch at or past K.
    if (kBeg < kEnd)
    {
        FETCH(0, kBeg)
        STAGE(0, 0u)
    }
    GroupMemoryBarrierWithGroupSync();

    uint cur = 0;
    for (uint k0 = kBeg; k0 < kEnd; k0 += GBK)
    {
        bool more = k0 + GBK < kEnd && GEMM_EXP != 1;
        if (more) { FETCH(0, k0 + GBK) }
#if GEMM_EXP != 2
        COMPUTE(cur)
#endif
#if GNBUF == 2
        if (more) { STAGE(0, cur ^ 1) }
        GroupMemoryBarrierWithGroupSync();
        cur ^= 1;
#else
        GroupMemoryBarrierWithGroupSync();
        if (more) { STAGE(0, 0u) }
        GroupMemoryBarrierWithGroupSync();
#endif
    }
#endif
    #undef COMPUTE
    #undef X_K
    #undef FETCH
    #undef STAGE
    #undef GEMM_LOAD_X
    #undef STAGE_LO_A
    #undef XS_LO
    #undef ROW_OF
    #undef ROWS_LIVE

#if GEMM_SPLIT
    // Undo the row scales of x and W: each subgroup stages its accumulators through its own shared slice.
    [ForceUnroll]
    for (int i = 0; i < FRAG_M; i++)
    {
        float xr[8];
        [ForceUnroll]
        for (uint e = 0; e < 8; e++)
        {
            uint t = t0 + wm + i * 16 + (lane + e * 32) / 16;
            xr[e] = t < p.T ? *(float*)(uint64_t(p.x) + uint64_t(t) * SPLIT_ROW_BYTES + 2 * SPLIT_HALVES * uint64_t(K)) : 0.0;
        }
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j++)
        {
            uint n = n0 + wn + j * 16 + lane % 16;
            float wr = n < N ? 1.0 / p.ws[n] : 0.0;
            acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(gC, sg * 256, 16u);
            GroupMemoryBarrierWithWaveSync();
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
                uint w = lane + e * 32;
                uint t = t0 + wm + i * 16 + w / 16;
                if (t < p.T && n < N) yb[uint64_t(t) * Y_STRIDE + n] = MM_STORE(gC[sg * 256 + w] * xr[e] * wr, t, n);
            }
            GroupMemoryBarrierWithWaveSync();
        }
    }
    #undef SPLIT_ROW_BYTES
    return;
#endif
#if DUAL && MOE
    // Expert-grouped: fragments j and j + 1 hold the products of w and w2 for the same 16 outputs, columns n0 + (wn + j * 16) / 2,
    // of the tile's rows (items gItem). Both pass through the subgroup's shared slice, so each lane knows the row and column of
    // its values; ranged rows are unscaled before the epilogue MOE_EPI(m0, m1, e, n) (defined by the including source).
    [ForceUnroll]
    for (int i = 0; i < FRAG_M; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j += 2)
        {
            uint tRow = wm + i * 16, nCol = n0 + (wn + j * 16) / 2;
            float m0[8];
            acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(gC, sg * 256, 16u);
            GroupMemoryBarrierWithWaveSync();
            [ForceUnroll] for (uint e = 0; e < 8; e++) m0[e] = gC[sg * 256 + lane + e * 32];
            GroupMemoryBarrierWithWaveSync();
            acc[i][j + 1].Store<CoopMatMatrixLayout.RowMajor>(gC, sg * 256, 16u);
            GroupMemoryBarrierWithWaveSync();
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
                uint w = lane + e * 32, t = tRow + w / 16, n = nCol + w % 16;
                float m1 = gC[sg * 256 + w];
                if (t < tileRows && n < N)
                {
#if MOE_RANGED
                    float inv = gRowInverse[t];
                    yb[uint64_t(gItem[t]) * N + n] = MOE_EPI(m0[e] * inv, m1 * inv, expert, n);
#else
                    yb[uint64_t(gItem[t]) * N + n] = MOE_EPI(m0[e], m1, expert, n);
#endif
                }
            }
            GroupMemoryBarrierWithWaveSync();
        }
    }
    return;
#elif DUAL
    // Fragments j and j + 1 hold the products of w and w2 for the same 16 outputs, columns n0 + (wn + j * 16) / 2.
#if DUAL_Y16_RANGE
    // The largest magnitude of the subgroup's outputs: from 2^15 on, it raises the flags of all the subgroup's rows (a bound
    // of each such row's largest value, which is all the fixup needs).
    float rangeMax = 0;
#endif
    [ForceUnroll]
    for (int i = 0; i < FRAG_M; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j += 2)
        {
            Acc2 r = Acc2(0.0);
            [ForceUnroll] for (uint e = 0; e < 8; e++) r[e] = epi2(acc[i][j][e], acc[i][j + 1][e]);
            uint tRow = t0 + wm + i * 16, nCol = n0 + (wn + j * 16) / 2;
            if (nCol >= N)
            {
                continue;   // a tile past the last output (N not a multiple of GBN_OUT)
            }
#if DUAL_Y16
#if DUAL_Y16_RANGE
            [ForceUnroll] for (uint e = 0; e < 8; e++) rangeMax = max(rangeMax, abs(r[e]));
#endif
            if (tRow + 16 <= p.T && nCol + 16 <= N)
            {
                AccH2 h = AccH2(r);
                h.Store<CoopMatMatrixLayout.RowMajor>((half*)p.y16 + uint64_t(tRow) * N + nCol, 0u, N);
            }
            else
            {
                r.Store<CoopMatMatrixLayout.RowMajor>(gC, sg * 256, 16u);
                GroupMemoryBarrierWithWaveSync();
                [ForceUnroll]
                for (uint e = 0; e < 8; e++)
                {
                    uint w = lane + e * 32, t = tRow + w / 16, n = nCol + w % 16;
                    if (t < p.T && n < N) *(uint16_t*)(p.y16 + (uint64_t(t) * N + n) * 2) = uint16_t(f32tof16(gC[sg * 256 + w]));
                }
                GroupMemoryBarrierWithWaveSync();
            }
#endif
            if (tRow + 16 <= p.T && nCol + 16 <= N)
            {
                r.Store<CoopMatMatrixLayout.RowMajor>(yb + uint64_t(tRow) * Y_STRIDE + nCol, 0u, Y_STRIDE);
                continue;
            }
            r.Store<CoopMatMatrixLayout.RowMajor>(gC, sg * 256, 16u);
            GroupMemoryBarrierWithWaveSync();
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
                uint w = lane + e * 32;
                uint t = tRow + w / 16;
                uint n = nCol + w % 16;
                if (t < p.T && n < N) yb[uint64_t(t) * Y_STRIDE + n] = gC[sg * 256 + w];
            }
            GroupMemoryBarrierWithWaveSync();
        }
    }
#if DUAL_Y16_RANGE
    if (WaveActiveAnyTrue(rangeMax >= 32768.0))
    {
        rangeMax = WaveActiveMax(rangeMax);
        for (uint r = lane; r < FRAG_M * 16; r += 32)
        {
            if (t0 + wm + r < p.T) InterlockedMax(p.y16_flags[t0 + wm + r], asuint(rangeMax));
        }
    }
#endif
    return;
#endif
#if SPLITK_FUSED
    // The partial slices are dense rows of N (the reduction reads them so); the last workgroup of the tile reduces it.
    #define GEMM_PART_STRIDE N
    #define GEMM_FINISH SPLITK_FINISH(p, tid, GTHREADS, GBM, GBN, gid.y * ((N + GBN - 1) / GBN) + gid.x, t0, n0)
#else
    #define GEMM_PART_STRIDE Y_STRIDE
    #define GEMM_FINISH
#endif
    if (!MOE && t0 + GBM <= p.T && (N % GBN == 0 || n0 + GBN <= N))
    {
        [ForceUnroll] for (int i = 0; i < FRAG_M; i++) [ForceUnroll] for (int j = 0; j < FRAG_N; j++)
            acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(yb + uint64_t(t0 + wm + i * 16) * GEMM_PART_STRIDE + n0 + wn + j * 16, 0u, GEMM_PART_STRIDE);
        GEMM_FINISH
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
#if MOE_RANGED
                if (t < tileRows && n < N) yb[uint64_t(gItem[t]) * N + n] = MOE_STORE1(gC[sg * 256 + w] * gRowInverse[t], expert, n, gItem[t]);
#else
                if (t < tileRows && n < N) yb[uint64_t(gItem[t]) * N + n] = MOE_STORE1(gC[sg * 256 + w], expert, n, gItem[t]);
#endif
#else
                if (t < p.T && n < N) yb[uint64_t(t) * GEMM_PART_STRIDE + n] = gC[sg * 256 + w];
#endif
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    GEMM_FINISH
    #undef GEMM_FINISH
    #undef GEMM_PART_STRIDE
}
#endif   // KERNEL_GEMM_CM2 entry
#include "matmul_skinny.slang"
#endif

#ifdef KERNEL_SPLITK_REDUCE
// Sums the SPLITK partial results of gemm_cm2 (x = partials [SPLITK][T][N]) into y. With X_RS, w is the row-ranged f16 copy the
// GEMM read, whose row scales are undone here.
[shader("compute")]
[numthreads(256, 1, 1)]
void splitk_reduce(uint3 dtid : SV_DispatchThreadID, uniform Params p)
{
    uint64_t i = dtid.x, total = uint64_t(p.T) * N;
    if (i >= total) return;
    float s = 0;
    [ForceUnroll] for (uint z = 0; z < SPLITK; z++) s += p.x[z * total + i];
#ifdef X_RS
    // The GEMM read a row-ranged f16 copy of its activations (common.slang, f16_range_scale), bound as w here.
    s *= f16_range_inverse(*(uint*)(uint64_t(p.w) + X_RS + (i / N) * 4));
#endif
#if MULTI > 1
    // The partials of a multi-output GEMM (splitk.slang): each column goes to its matrix's output, with its bias.
    MULTI_STORE(p, s, i / N, uint(i % N))
#else
    p.y[(i / N) * Y_STRIDE + i % N] = MM_STORE(s, i / N, uint(i % N));
#endif
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
#ifdef GEMM_GROUP
    gid.xy = grouped_tile(gid.xy, (N + NV_BN - 1) / NV_BN, (p.T + NV_BT - 1) / NV_BT, GEMM_GROUP);
#endif
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
