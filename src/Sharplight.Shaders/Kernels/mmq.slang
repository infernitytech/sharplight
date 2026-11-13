// Integer tensor-core GEMM for prefill-sized batches (DeviceFeatures.MatrixInt8): y[t, n] = sum_k x[t, k] * W[n, k].
//
// x arrives q8_1-quantized (quantize_q8_1 in mmvq.slang: per row [K int8 q][K/32 float2 (d, d * sum q)]), the weight
// blocks are unpacked to int8 codes with one scale (and min) per 32 values. Every 32-value K block is one 16x16x32 int8
// MMA into a fresh int32 accumulator, which is scaled into f32 registers:
//     acc[t, n] += d_x[t] * d_w[n] * C[t, n] - s_x[t] * m_w[n]           (m_w: the K-quant min term, d_min * m)
// Q6_K has a scale per 16 values: its block takes a second MMA against the codes of the upper half only,
//     acc[t, n] += d_x[t] * (d_lo[n] * C[t, n] + (d_hi[n] - d_lo[n]) * C_hi[t, n]).
//
// The scaling needs the (row, column) of each accumulator element a lane holds, which the cooperative-matrix API leaves
// to the implementation. The kernel reads it once by loading an index matrix. When the layout is the 2-row x 4-column
// pattern of NVIDIA's mma fragments (checked at run time over the subgroup, see PAIR_LAYOUT) a lane loads the scales of
// 2 rows and 4 columns per fragment and block; otherwise it loads them per element (correct for any layout, slower).
//
// Workgroup = MQ_WM x MQ_WN subgroups, each owning MQ_FM x MQ_FN 16x16 output fragments, i.e. a tile of
// (MQ_WM * MQ_FM * 16) tokens x (MQ_WN * MQ_FN * 16) weight rows. K is walked in stages of MQ_BK values (staged in shared
// memory, the next stage prefetched into registers during the MMAs). SPLITK > 1: workgroup z sums its share of K into
// slice z of y ([SPLITK][T][N], reduced by splitk_reduce in matmul.slang).
//
// Specialization macros: WFMT (q8_0r, q4_k, q5_k, q6_k, iq4_nl, iq4_xs), K, N, Y_STRIDE, ROW_BYTES, MQ_WM, MQ_WN, MQ_FM,
// MQ_FN, MQ_BK, SPLITK, MQ_GENERIC_LAYOUT (1: always use the per-element scale loads; tests).
// Requires K % MQ_BK == 0, 32-lane subgroups.
#include "common.slang"
using namespace linalg;

#define QROW (K + K / 4)

#ifndef MQ_WM
#define MQ_WM 2
#endif
#ifndef MQ_WN
#define MQ_WN 4
#endif
#ifndef MQ_FM
#define MQ_FM 4
#endif
#ifndef MQ_FN
#define MQ_FN 2
#endif
#ifndef MQ_BK
#define MQ_BK 64
#endif
#ifndef SPLITK
#define SPLITK 1
#endif
#ifndef MQ_GENERIC_LAYOUT
#define MQ_GENERIC_LAYOUT 0
#endif

#define MQ_THREADS (MQ_WM * MQ_WN * 32)
#define MQ_BM (MQ_WM * MQ_FM * 16)
#define MQ_BN (MQ_WN * MQ_FN * 16)
#define MQ_BLOCKS (MQ_BK / 32)
// Shared int8 rows are padded by 16 bytes (rows 80 or 144 bytes apart: conflict-free fragment loads); sizes in uint4.
#define MQ_RS (MQ_BK / 16 + 1)
// Prefetch items per thread: activation uint4s, activation block scales, weight units (one 32-value block of one row).
#define MQ_A_ITEMS ((MQ_BM * MQ_BK / 16 + MQ_THREADS - 1) / MQ_THREADS)
#define MQ_AS_ITEMS ((MQ_BM * MQ_BLOCKS + MQ_THREADS - 1) / MQ_THREADS)
#define MQ_B_ITEMS ((MQ_BN * MQ_BLOCKS + MQ_THREADS - 1) / MQ_THREADS)

#if WFMT == FMT_Q4_K || WFMT == FMT_Q5_K
#define MQ_MIN 1
#else
#define MQ_MIN 0
#endif
#if WFMT == FMT_Q6_K
#define MQ_SPLIT16 1
#else
#define MQ_SPLIT16 0
#endif

struct Params
{
    uint8_t* x;
    uint8_t* w;
    float* y;
    uint T;
};

typealias MatA8 = CoopMat<int8_t, MemoryScope.Subgroup, 16, 32, CoopMatMatrixUse.MatrixA>;
typealias MatB8 = CoopMat<int8_t, MemoryScope.Subgroup, 32, 16, CoopMatMatrixUse.MatrixB>;
typealias AccI = CoopMat<int, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

groupshared uint4 sA[MQ_BM * MQ_RS];
groupshared uint4 sB[MQ_BN * MQ_RS];
#if MQ_SPLIT16
groupshared uint4 sBh[MQ_BN * MQ_RS];
#endif
// Block scales, block-major: [MQ_BLOCKS][rows] float2. Activations (d, d * sum q); weights (d, min) or Q6_K (d_lo, d_hi - d_lo).
groupshared float2 sAS[MQ_BLOCKS * MQ_BM];
groupshared float2 sBS[MQ_BLOCKS * MQ_BN];
groupshared int sIndex[256];

// 16 bytes from a 2-byte aligned address.
uint4 load16_a2(uint8_t* p)
{
    uint16_t* h = (uint16_t*)p;
    return uint4(uint(h[0]) | (uint(h[1]) << 16), uint(h[2]) | (uint(h[3]) << 16), uint(h[4]) | (uint(h[5]) << 16), uint(h[6]) | (uint(h[7]) << 16));
}

#if WFMT == FMT_IQ4_NL || WFMT == FMT_IQ4_XS
// Four nibbles (one per byte of u, low 4 bits) through the IQ4 codebook, as four int8 bytes.
uint iq4_bytes(uint u)
{
    // kvalues = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113} packed four per word.
    const uint kv[4] = { 0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u };
    uint r = 0;
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
    {
        uint n = (u >> (8 * i)) & 0xF;
        r |= ((kv[n >> 2] >> (8 * (n & 3))) & 0xFF) << (8 * i);
    }
    return r;
}

uint4 iq4_bytes4(uint4 u) { return uint4(iq4_bytes(u.x), iq4_bytes(u.y), iq4_bytes(u.z), iq4_bytes(u.w)); }
#endif

// The 32 int8 codes (K order) of the 32-value block `b` (K offset 32b) of weight row `row`, and its scale pair.
void load_unit(uint8_t* w, uint row, uint b, out uint4 q0, out uint4 q1, out float2 s)
{
    uint8_t* r = rowStart(w, ROW_BYTES, row);
#if WFMT == FMT_Q8_0R
    // Repacked Q8_0 row: [K/32 f16 scales][K int8].
    s = float2(h2f(*(uint16_t*)(r + b * 2)), 0);
    uint4* q = (uint4*)(r + (ROW_BYTES / 17 + b * 32));
    q0 = q[0];
    q1 = q[1];
#elif WFMT == FMT_Q4_K || WFMT == FMT_Q5_K
#if WFMT == FMT_Q4_K
#define KQ_BYTES 144
#define KQ_QS 16
#else
#define KQ_BYTES 176
#define KQ_QS 48
#endif
    // Sub-block j of a super-block: the low (j even) or high nibbles of the 32 qs bytes of pair j / 2.
    uint8_t* blk = r + uint64_t(b >> 3) * KQ_BYTES;
    uint j = b & 7;
    uint4* qs = (uint4*)(blk + (KQ_QS + 32 * (j >> 1)));
    uint shift = 4 * (j & 1);
    q0 = (qs[0] >> shift) & 0x0F0F0F0Fu;
    q1 = (qs[1] >> shift) & 0x0F0F0F0Fu;
#if WFMT == FMT_Q5_K
    uint4* qh = (uint4*)(blk + 16);
    q0 |= ((qh[0] >> j) & 0x01010101u) << 4;
    q1 |= ((qh[1] >> j) & 0x01010101u) << 4;
#endif
#undef KQ_BYTES
#undef KQ_QS
    uint dm = *(uint*)blk;
    float sc, mn;
    scale_min_k4(blk + 4, j, sc, mn);
    s = float2(f16tof32(dm & 0xFFFF) * sc, f16tof32(dm >> 16) * mn);
#elif WFMT == FMT_Q6_K
    // Super-block of 256 = { ql[128]; qh[64]; int8 scales[16]; f16 d } (210 bytes, 2-byte aligned): sub-block b & 7 is
    // quadrant qd of half n; its two 16-value halves have their own scales.
    uint8_t* blk = r + uint64_t(b >> 3) * 210;
    uint j = b & 7;
    uint n = j >> 2, qd = j & 3;
    uint8_t* ql = blk + (n * 64 + (qd & 1) * 32);
    uint8_t* qh = blk + (128 + n * 32);
    uint lshift = 4 * (qd >> 1), hshift = 2 * qd;
    uint4 c0 = ((load16_a2(ql) >> lshift) & 0x0F0F0F0Fu) | (((load16_a2(qh) >> hshift) & 0x03030303u) << 4);
    uint4 c1 = ((load16_a2(ql + 16) >> lshift) & 0x0F0F0F0Fu) | (((load16_a2(qh + 16) >> hshift) & 0x03030303u) << 4);
    // Codes 0..63 minus 32 per byte, without borrows between bytes: (c + 96) ^ 0x80.
    q0 = (c0 + 0x60606060u) ^ 0x80808080u;
    q1 = (c1 + 0x60606060u) ^ 0x80808080u;
    float d = h2f(*(uint16_t*)(blk + 208));
    uint sc = uint(*(uint16_t*)(blk + (192 + n * 8 + 2 * qd)));
    float lo = d * float(int(sc << 24) >> 24);
    float hi = d * float(int(sc << 16) >> 24);
    s = float2(lo, hi - lo);
#elif WFMT == FMT_IQ4_NL
    // Blocks of 32 = { f16 d; u8 qs[16] } (18 bytes): values 0..15 are the low nibbles, 16..31 the high nibbles.
    uint8_t* blk = r + uint64_t(b) * 18;
    uint4 qs = load16_a2(blk + 2);
    q0 = iq4_bytes4(qs & 0x0F0F0F0Fu);
    q1 = iq4_bytes4((qs >> 4) & 0x0F0F0F0Fu);
    s = float2(h2f(*(uint16_t*)blk), 0);
#elif WFMT == FMT_IQ4_XS
    // Super-block of 256 = { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128] } (136 bytes); sub-block ib like IQ4_NL
    // with the 6-bit scale ls - 32.
    uint8_t* blk = r + uint64_t(b >> 3) * 136;
    uint ib = b & 7;
    uint head = *(uint*)blk;
    uint sh = head >> 16;
    uint ls = ((uint(blk[4 + (ib >> 1)]) >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    // 136-byte blocks are 8-byte aligned.
    uint2* q8 = (uint2*)(blk + (8 + 16 * ib));
    uint4 qs = uint4(q8[0], q8[1]);
    q0 = iq4_bytes4(qs & 0x0F0F0F0Fu);
    q1 = iq4_bytes4((qs >> 4) & 0x0F0F0F0Fu);
    s = float2(f16tof32(head & 0xFFFF) * float(int(ls) - 32), 0);
#else
#error "gemm_mmq: unsupported weight format"
#endif
}

// int32 -> f32 without the slow conversion instruction: the MMAs start from the bias 0x4B400000 (the bits of 1.5 * 2^23), so
// a block product |c| < 2^22 (at most 32 * 128 * 128 = 2^19) sits in the mantissa and one subtraction recovers it.
#define I2F_BIAS 0x4B400000
float i2f(int biased) { return asfloat(biased) - 12582912.0; }

// Adds one scaled int32 block product (and for Q6_K the upper-half product ch) to the f32 accumulators of one element.
void accumulate(inout float acc, int c, int ch, float2 x, float2 w)
{
#if MQ_SPLIT16
    acc += x.x * (i2f(c) * w.x + i2f(ch) * w.y);
#elif MQ_MIN
    acc += i2f(c) * (x.x * w.x) - x.y * w.y;
#else
    acc += i2f(c) * (x.x * w.x);
#endif
}

// The tile loop and the stores. PAIR: the accumulator layout is the pair layout (element e is row r0 + 8 * (e >> 2),
// column c0 + (e & 1) + 8 * ((e >> 1) & 1)), so a lane needs the scales of 2 rows and 4 columns per fragment; otherwise
// every element's row and column come from `index` (row * 16 + column).
void mmq_tile<let PAIR : bool>(uint3 gid, uint tid, Params p, int index[8])
{
    uint sg = tid / 32;
    uint n0 = gid.x * MQ_BN;
    uint t0 = gid.y * MQ_BM;
    uint wm = (sg / MQ_WN) * (MQ_FM * 16);
    uint wn = (sg % MQ_WN) * (MQ_FN * 16);
    uint r0 = uint(index[0]) >> 4, c0 = uint(index[0]) & 15;
#if SPLITK > 1
    const uint KS = ((K / MQ_BK + SPLITK - 1) / SPLITK) * MQ_BK;
    uint kBeg = gid.z * KS, kEnd = min(uint(K), kBeg + KS);
    float* yb = p.y + uint64_t(gid.z) * p.T * N;
#else
    uint kBeg = 0, kEnd = K;
    float* yb = p.y;
#endif

    float acc[MQ_FM][MQ_FN][8];
    [ForceUnroll] for (int i = 0; i < MQ_FM; i++) [ForceUnroll] for (int j = 0; j < MQ_FN; j++) [ForceUnroll] for (int e = 0; e < 8; e++) acc[i][j][e] = 0;

    uint4 xa[MQ_A_ITEMS];
    float2 xsc[MQ_AS_ITEMS];
    uint4 wq[MQ_B_ITEMS][2];
    float2 wsc[MQ_B_ITEMS];

    // Rows past T or N are clamped for loading (their results are never stored).
    #define FETCH(k0)                                                                                   \
    [ForceUnroll] for (uint it = 0; it < MQ_A_ITEMS; it++)                                              \
    {                                                                                                   \
        uint idx = tid + it * MQ_THREADS, r = idx / (MQ_BK / 16), q = idx % (MQ_BK / 16);               \
        uint t = min(t0 + r, p.T - 1);                                                                  \
        if (idx < MQ_BM * MQ_BK / 16) xa[it] = *(uint4*)byteAt(p.x, uint64_t(t) * QROW + (k0) + q * 16); \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < MQ_AS_ITEMS; it++)                                             \
    {                                                                                                   \
        uint idx = tid + it * MQ_THREADS, r = idx % MQ_BM, bl = idx / MQ_BM;                            \
        uint t = min(t0 + r, p.T - 1);                                                                  \
        if (idx < MQ_BM * MQ_BLOCKS) xsc[it] = *(float2*)byteAt(p.x, uint64_t(t) * QROW + K + ((k0) / 32 + bl) * 8); \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < MQ_B_ITEMS; it++)                                              \
    {                                                                                                   \
        uint idx = tid + it * MQ_THREADS, r = idx / MQ_BLOCKS, bl = idx % MQ_BLOCKS;                    \
        if (idx < MQ_BN * MQ_BLOCKS) load_unit(p.w, min(n0 + r, uint(N) - 1), (k0) / 32 + bl, wq[it][0], wq[it][1], wsc[it]); \
    }
#if MQ_SPLIT16
    #define STAGE_HI(o) sBh[o] = uint4(0); sBh[(o) + 1] = wq[it][1];
#else
    #define STAGE_HI(o)
#endif
    #define STAGE()                                                                                     \
    [ForceUnroll] for (uint it = 0; it < MQ_A_ITEMS; it++)                                              \
    {                                                                                                   \
        uint idx = tid + it * MQ_THREADS, r = idx / (MQ_BK / 16), q = idx % (MQ_BK / 16);               \
        if (idx < MQ_BM * MQ_BK / 16) sA[r * MQ_RS + q] = xa[it];                                       \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < MQ_AS_ITEMS; it++)                                             \
    {                                                                                                   \
        uint idx = tid + it * MQ_THREADS;                                                               \
        if (idx < MQ_BM * MQ_BLOCKS) sAS[idx] = xsc[it];                                                \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < MQ_B_ITEMS; it++)                                              \
    {                                                                                                   \
        uint idx = tid + it * MQ_THREADS, r = idx / MQ_BLOCKS, bl = idx % MQ_BLOCKS;                    \
        if (idx < MQ_BN * MQ_BLOCKS)                                                                    \
        {                                                                                               \
            uint o = r * MQ_RS + bl * 2;                                                                \
            sB[o] = wq[it][0]; sB[o + 1] = wq[it][1];                                                   \
            STAGE_HI(o)                                                                                 \
            sBS[bl * MQ_BN + r] = wsc[it];                                                              \
        }                                                                                               \
    }

    // A split-K slice may be empty (K steps not divisible by SPLITK): never prefetch at or past K.
    if (kBeg < kEnd)
    {
        FETCH(kBeg)
        STAGE()
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint k0 = kBeg; k0 < kEnd; k0 += MQ_BK)
    {
        bool more = k0 + MQ_BK < kEnd;
        if (more) { FETCH(k0 + MQ_BK) }

        [ForceUnroll]
        for (uint bl = 0; bl < MQ_BLOCKS; bl++)
        {
            MatB8 b[MQ_FN];
#if MQ_SPLIT16
            MatB8 bh[MQ_FN];
#endif
            float2 ws[MQ_FN][4];
            [ForceUnroll]
            for (int j = 0; j < MQ_FN; j++)
            {
                b[j] = MatB8.Load<CoopMatMatrixLayout.ColumnMajor>(sB, (wn + j * 16) * MQ_RS + bl * 2, MQ_RS);
#if MQ_SPLIT16
                bh[j] = MatB8.Load<CoopMatMatrixLayout.ColumnMajor>(sBh, (wn + j * 16) * MQ_RS + bl * 2, MQ_RS);
#endif
                if (PAIR)
                {
                    uint base = bl * MQ_BN + wn + j * 16 + c0;
                    ws[j][0] = sBS[base]; ws[j][1] = sBS[base + 1]; ws[j][2] = sBS[base + 8]; ws[j][3] = sBS[base + 9];
                }
            }
            [ForceUnroll]
            for (int i = 0; i < MQ_FM; i++)
            {
                MatA8 a = MatA8.Load<CoopMatMatrixLayout.RowMajor>(sA, (wm + i * 16) * MQ_RS + bl * 2, MQ_RS);
                float2 xs[2];
                if (PAIR)
                {
                    uint base = bl * MQ_BM + wm + i * 16 + r0;
                    xs[0] = sAS[base]; xs[1] = sAS[base + 8];
                }
                [ForceUnroll]
                for (int j = 0; j < MQ_FN; j++)
                {
                    AccI c = coopMatMulAdd<int, false>(a, b[j], AccI(I2F_BIAS));
#if MQ_SPLIT16
                    AccI ch = coopMatMulAdd<int, false>(a, bh[j], AccI(I2F_BIAS));
#else
                    AccI ch = c;
#endif
                    [ForceUnroll]
                    for (uint e = 0; e < 8; e++)
                    {
                        if (PAIR)
                        {
                            accumulate(acc[i][j][e], c[e], ch[e], xs[e >> 2], ws[j][e & 3]);
                        }
                        else
                        {
                            accumulate(acc[i][j][e], c[e], ch[e], sAS[bl * MQ_BM + wm + i * 16 + (uint(index[e]) >> 4)],
                                       sBS[bl * MQ_BN + wn + j * 16 + (uint(index[e]) & 15)]);
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
    #undef STAGE_HI

    [ForceUnroll]
    for (int i = 0; i < MQ_FM; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < MQ_FN; j++)
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
[numthreads(MQ_THREADS, 1, 1)]
void gemm_mmq(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    uint tid = ltid.x;
    // Element layout of the int32 accumulator: load the matrix whose element (r, c) is r * 16 + c.
    for (uint i = tid; i < 256; i += MQ_THREADS) sIndex[i] = int(i);
    GroupMemoryBarrierWithGroupSync();
    AccI probe = AccI.Load<CoopMatMatrixLayout.RowMajor>(sIndex, 0, 16);
    int index[8];
    [ForceUnroll] for (uint e = 0; e < 8; e++) index[e] = probe[e];
    uint r0 = uint(index[0]) >> 4, c0 = uint(index[0]) & 15;
    bool pairLane = (c0 & 1) == 0 && r0 < 8 && c0 < 8;
    [ForceUnroll]
    for (uint e = 0; e < 8; e++)
        pairLane = pairLane && uint(index[e]) == (r0 + 8 * (e >> 2)) * 16 + c0 + (e & 1) + 8 * ((e >> 1) & 1);
    // Uniform over the workgroup: every subgroup sees the same layout.
    if (!MQ_GENERIC_LAYOUT && WaveActiveAllTrue(pairLane))
        mmq_tile<true>(gid, tid, p, index);
    else
        mmq_tile<false>(gid, tid, p, index);
}
