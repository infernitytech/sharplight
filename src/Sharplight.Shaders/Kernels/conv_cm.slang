// 2-D convolution as an implicit GEMM on cooperative matrices (16x16x16, f32 accumulate):
//   C[o, pix] = sum_k W[g*OG + o, k] * im2col(x)[k, pix] per group g, pix over N*HO*WO.
// Workgroup = WARPS_M x 4 subgroups of 32; each subgroup owns 4 x 2 accumulators (64 o x 32 pix), so the workgroup
// tile is (WARPS_M * 64) o x (FRAG_N * 64) pix. K advances in steps of 32; the next step's global loads are issued into
// registers (packed operands) before this step's products.
// PARTS = 3: every operand is split into bf16 hi + mid + lo (24 mantissa bits, exact for f32 inputs up to rounding of the
//            last part); hi*hi, hi*mid + mid*hi and mid*mid + hi*lo + lo*hi accumulate separately in f32 and are summed
//            smallest first: f32-equivalent products, the strict-precision path. Needs bf16 cooperative matrices.
// PARTS = 2: bf16 hi + lo; hi*hi and hi*lo + lo*hi (about 16 mantissa bits): reduced precision without f16 matrices.
// PARTS = 1: one f16 product per step (reduced precision). Needs f16 cooperative matrices.
// PARTS = 4: split f16 (strict precision with f16 cooperative matrices): hi + lo f16 parts, hi*hi + hi*lo + lo*hi. The
//            weights come split on the host (p.w: per output channel KTOT hi halves, then KTOT lo halves, of the row scaled
//            by p.ws, see split_scale); the im2col values are scaled per K step and block of 16 pixels by the power of two
//            that puts the block's largest magnitude in [2^14, 2^15), so neither part leaves f16's normal range. A step's
//            products sum from zero on the tensor cores and are added, times the block's inverse scale, to f32
//            accumulators (the tensor cores' truncating accumulation then only rounds step sums). Needs every block's
//            pixels in one subgroup: TBN * KG / NT (B_ITEMS) of at least KG / 2.
//            HI_ONLY = 1 (fast matrix precision): the same scaling with the hi parts only, one product per step (11
//            significant bits per input, like TF32); the weights hold KTOT hi halves per output channel. A block keeps its
//            scale across steps while the values fit it (split_b), so the products accumulate on the tensor cores like TF32's.
// Same layout and macros as conv_gemm (conv.slang): x [N, C, H, W] strided (XS_*), w [O, CG, KH, KW] dense (WFMT
// f32/f16/bf16), y dense [N, O, HO, WO]. SPLITK > 1 writes partials [SPLITK][N][O][HO*WO] for conv_splitk_reduce.
// TAP_MAJOR = 1 (PARTS 4): the reduction runs tap-major, k = (kh * KW + kw) * CG + c (the host reorders the split weights); with
// CG % 8 == 0 a staged item of 8 k is 8 channels of one input pixel, otherwise its position is stepped per value.
// Extra macros: PARTS (1-3) FRAG_N (1 or 2 pixel fragments per subgroup) WARPS_M (1 or 2) A_VEC (K % 8 == 0: weights loaded in groups of 8) SPLITK KPS (K per split,
// multiple of 32).
#include "common.slang"
#include "conv_epilogue.slang"
using namespace linalg;

#ifndef HI_ONLY
#define HI_ONLY 0
#endif
#ifndef TAP_MAJOR
#define TAP_MAJOR 0
#endif
// Lo parts staged and multiplied (bf16 forms and the strict split-f16 form).
#if PARTS > 1 && !HI_ONLY
#define HAS_LO 1
#else
#define HAS_LO 0
#endif
#if HI_ONLY
#define W_BYTES_PER_K 2
#else
#define W_BYTES_PER_K 4
#endif

#define KHW (KH * KW)
#define KTOT (CG * KH * KW)
#define WARPS_N 4
#define FRAG_M 4
#define NT (WARPS_M * WARPS_N * 32)
#define TBM (WARPS_M * FRAG_M * 16)
#define TBN (WARPS_N * FRAG_N * 16)
#ifndef TBK
#define TBK 32
#endif
// 8-value k groups per K step.
#define KG (TBK / 8)
// Row stride of the shared tiles in halves: 32 k plus 8 halves of padding.
#define SAS (TBK + 8)
// Groups of 8 consecutive k per thread and step: A rows x 4 groups, B pixels x 4 groups.
#define A_ITEMS (TBM * KG / NT)
#define B_ITEMS (TBN * KG / NT)
#if PARTS == 4
// The B_TPP threads of a pixel are adjacent lanes (pixel tid / B_TPP, k groups tid % B_TPP + it * B_TPP), so a block of 16
// pixels is 16 * B_TPP lanes of one subgroup.
#define B_TPP (KG / B_ITEMS)
#define B_STRIDE B_TPP
#else
#define B_STRIDE (NT / TBN)
#endif

#if PARTS == 4
typealias Elem = half;
#elif PARTS > 1
typealias Elem = BFloat16;
#else
typealias Elem = half;
#endif
#if SL_TARGET_CUDA
#include "cuda_mma.slang"
#if PARTS > 1 && PARTS != 4
typealias MatA = CuA16bf;
typealias MatB = CuB16bf;
#else
typealias MatA = CuA16;
typealias MatB = CuB16;
#endif
typealias Acc = CuAccF;
#else
typealias MatA = CoopMat<Elem, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<Elem, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias Acc = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
#endif

struct ConvParams
{
    float* x;
    uint8_t* w;
    float* bias;
    float* y;
#if PARTS == 4
    float* ws;   // per output channel: the power-of-two scale the split weights carry
#endif
#if EPI_RES
    float* res;
#endif
    uint N;
};

// CONV_FRAG8 (DeviceFeatures.MatrixF16Small, Metal simdgroup_matrix; single-product form only): each 16x16 accumulator is 2 x 2
// fragments of 8x8 over K steps of 8, and the tiles are staged as half arrays (Slang's Metal target cannot load fragments from
// packed uint4 arrays).
#ifndef CONV_FRAG8
#define CONV_FRAG8 0
#endif
#if CONV_FRAG8 && PARTS != 1
#error "CONV_FRAG8 takes the single-product form (PARTS 1)"
#endif
#if CONV_FRAG8
typealias Mat8A = CoopMat<half, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixA>;
typealias Mat8B = CoopMat<half, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixB>;
typealias Acc8 = CoopMat<float, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixAccumulator>;
groupshared half sAh[TBM * SAS];
groupshared half sBh[TBN * SAS];
// 8 packed halves at half index `index`.
#define PUT8H(arr, index, v) \
    { uint4 pv = (v); uint pw[4] = { pv.x, pv.y, pv.z, pv.w }; \
      [ForceUnroll] for (uint pe = 0; pe < 4; pe++) { arr[(index) + 2 * pe] = half(f16tof32(pw[pe] & 0xFFFFu)); arr[(index) + 2 * pe + 1] = half(f16tof32(pw[pe] >> 16)); } }
#else
// A: weights [o][k] (row-major), B: im2col [pix][k] (column-major B); 8 packed operands per uint4, one tile per part.
groupshared uint4 sAh[TBM * SAS / 8];
groupshared uint4 sBh[TBN * SAS / 8];
#endif
#if HAS_LO
groupshared uint4 sAl[TBM * SAS / 8];
groupshared uint4 sBl[TBN * SAS / 8];
#endif
#if PARTS == 3
groupshared uint4 sAm[TBM * SAS / 8];
groupshared uint4 sBm[TBN * SAS / 8];
#endif
#if PARTS == 4
groupshared float sScB[TBN / 16];   // inverse scale of each 16-pixel block of the staged step (HI_ONLY: accumulator factor)
#if HI_ONLY
groupshared float sInvB[TBN / 16];  // HI_ONLY: inverse of each block's final scale
#endif
#endif
groupshared float sC[WARPS_M * WARPS_N * 256];

float cm_weight(uint8_t* base, uint row, uint k)
{
    uint8_t* r = base + uint64_t(ROW_BYTES) * row;
#if WFMT == FMT_F32
    return ((float*)r)[k];
#elif WFMT == FMT_F16
    return h2f(((uint16_t*)r)[k]);
#else
    return bf2f(((uint16_t*)r)[k]);
#endif
}

// bf16 bits of v, rounded to nearest even (inputs are finite).
uint bf16_round(float v)
{
    uint u = asuint(v);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

uint4 pack_words(uint e[8])
{
    return uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16), e[4] | (e[5] << 16), e[6] | (e[7] << 16));
}

// Packs 8 values into operand words: hi, mid (PARTS 3) and lo (PARTS > 1) parts; each part is the bf16 rounding of what
// the previous parts leave over. Unused parts are zero.
void pack8(float v[8], out uint4 hi, out uint4 mid, out uint4 lo)
{
    uint h[8];
    uint m[8];
    uint l[8];
    [ForceUnroll]
    for (int j = 0; j < 8; j++)
    {
#if PARTS > 1
        h[j] = bf16_round(v[j]);
        float r = v[j] - asfloat(h[j] << 16);
#if PARTS > 2
        m[j] = bf16_round(r);
        r -= asfloat(m[j] << 16);
#else
        m[j] = 0;
#endif
        l[j] = bf16_round(r);
#else
        m[j] = 0;
        l[j] = 0;
#endif
    }
#if PARTS == 1
    hi = uint4(bit_cast<uint>(half2(half(v[0]), half(v[1]))), bit_cast<uint>(half2(half(v[2]), half(v[3]))),
               bit_cast<uint>(half2(half(v[4]), half(v[5]))), bit_cast<uint>(half2(half(v[6]), half(v[7]))));
#else
    hi = pack_words(h);
#endif
    mid = pack_words(m);
    lo = pack_words(l);
}

// Global loads of the K step at k0 into packed registers; zero outside the tensors and at or past kEnd.
// A item it: row idx / KG, k group idx % KG (idx = tid + it * NT). B item it: the thread's pixel, k group bq0 + it * B_STRIDE.
void fetch(ConvParams p, uint g, uint mBase, uint k0, uint kEnd, uint tid, uint bq0, bool bvalid, int ih0, int iw0,
    float* xb, out uint4 ah[A_ITEMS], out uint4 am[A_ITEMS], out uint4 al[A_ITEMS], out uint4 bh[B_ITEMS], out uint4 bm[B_ITEMS],
    out uint4 bl[B_ITEMS], inout float bv[B_ITEMS][8])
{
    [ForceUnroll]
    for (uint it = 0; it < A_ITEMS; it++)
    {
        uint idx = tid + it * NT;
        uint r = idx / KG;
        uint kk = k0 + (idx % KG) * 8;
#if PARTS == 4
        // Split weights: row o holds KTOT hi halves, then KTOT lo halves.
        ah[it] = 0;
        al[it] = 0;
        am[it] = 0;
        if (mBase + r < OG && kk < kEnd)
        {
            uint64_t row = uint64_t(p.w) + uint64_t(g * OG + mBase + r) * W_BYTES_PER_K * uint64_t(KTOT);
#if A_VEC
            ah[it] = *(uint4*)(row + uint64_t(kk) * 2);
#if !HI_ONLY
            al[it] = *(uint4*)(row + (uint64_t(KTOT) + kk) * 2);
#endif
#else
            uint h[8];
            uint l[8];
            [ForceUnroll]
            for (uint j = 0; j < 8; j++)
            {
                h[j] = kk + j < kEnd ? uint(*(uint16_t*)(row + uint64_t(kk + j) * 2)) : 0u;
                l[j] = kk + j < kEnd && HI_ONLY == 0 ? uint(*(uint16_t*)(row + (uint64_t(KTOT) + kk + j) * 2)) : 0u;
            }
            ah[it] = pack_words(h);
            al[it] = pack_words(l);
#endif
        }
        continue;
#endif
        float v[8];
        [ForceUnroll] for (int j = 0; j < 8; j++) { v[j] = 0.0; }
        if (mBase + r < OG && kk < kEnd)
        {
#if A_VEC
            load8(p.w, ROW_BYTES, g * OG + mBase + r, kk / 8, v);
#else
            [ForceUnroll]
            for (uint j = 0; j < 8; j++)
            {
                if (kk + j < kEnd)
                {
                    v[j] = cm_weight(p.w, g * OG + mBase + r, kk + j);
                }
            }
#endif
        }
        pack8(v, ah[it], am[it], al[it]);
    }
    [ForceUnroll]
    for (uint it = 0; it < B_ITEMS; it++)
    {
        uint kq = k0 + (bq0 + it * B_STRIDE) * 8;
        float v[8];
#if TAP_MAJOR && CG % 8 == 0
        // k = tap * CG + c with CG % 8 == 0: the item's 8 k are channels c0..c0+7 of one input pixel.
        {
            uint tap = kq / CG;
            uint c0 = kq - tap * CG;
            uint kh = tap / KW;
            int ih = ih0 + int(kh) * DH;
            int iw = iw0 + int(tap - kh * KW) * DW;
#if NOPAD
            bool inside = bvalid && kq < kEnd;
#else
            bool inside = bvalid && kq < kEnd && uint(ih) < uint(H) && uint(iw) < uint(W);
#endif
            [ForceUnroll]
            for (uint j = 0; j < 8; j++)
            {
                v[j] = 0.0;
            }
            if (inside)
            {
                float* src = xb + uint64_t(c0) * XS_C + ih * XS_H + iw * XS_W;
                [ForceUnroll]
                for (uint j = 0; j < 8; j++)
                {
                    v[j] = src[uint64_t(j) * XS_C];
                }
            }
        }
#elif TAP_MAJOR
        // k = tap * CG + c, few channels (stems): the item's position is decomposed once and then stepped channel by channel.
        {
            uint tap = kq / CG;
            uint c = kq - tap * CG;
            uint kh = tap / KW;
            uint kw = tap - kh * KW;
            [ForceUnroll]
            for (uint j = 0; j < 8; j++)
            {
                v[j] = 0.0;
                int ih = ih0 + int(kh) * DH;
                int iw = iw0 + int(kw) * DW;
#if NOPAD
                if (bvalid && kq + j < kEnd)
#else
                if (bvalid && kq + j < kEnd && uint(ih) < uint(H) && uint(iw) < uint(W))
#endif
                {
                    v[j] = xb[uint64_t(c) * XS_C + ih * XS_H + iw * XS_W];
                }
                c++;
                if (c == CG)
                {
                    c = 0;
                    kw++;
                    if (kw == KW)
                    {
                        kw = 0;
                        kh++;
                    }
                }
            }
        }
#else
        [ForceUnroll]
        for (uint j = 0; j < 8; j++)
        {
            uint k = kq + j;
            v[j] = 0.0;
            if (bvalid && k < kEnd)
            {
#if KHW == 1
                uint c = k;
                int ih = ih0;
                int iw = iw0;
#else
                uint c = k / KHW;
                uint r = k - c * KHW;
                uint kh = r / KW;
                uint kw = r - kh * KW;
                int ih = ih0 + int(kh) * DH;
                int iw = iw0 + int(kw) * DW;
#endif
#if NOPAD
                v[j] = xb[uint64_t(c) * XS_C + ih * XS_H + iw * XS_W];
#else
                if (uint(ih) < uint(H) && uint(iw) < uint(W))
                {
                    v[j] = xb[uint64_t(c) * XS_C + ih * XS_H + iw * XS_W];
                }
#endif
            }
        }
#endif
#if PARTS == 4
        [ForceUnroll]
        for (uint j = 0; j < 8; j++)
        {
            bv[it][j] = v[j];
        }
        bh[it] = 0;
        bm[it] = 0;
        bl[it] = 0;
#else
        pack8(v, bh[it], bm[it], bl[it]);
#endif
    }
}

#if PARTS == 4
// Scales and splits the raw im2col values bv of a step (fetch only issues their loads, so that they stay in flight during the
// previous step's products): the block's largest magnitude, from the thread's values and then the 16 * B_TPP lanes of its 16
// pixels, sets the power-of-two scale. Returns the block's inverse scale.
// HI_ONLY: the block keeps its scale (cur, 0 before the first nonzero step) while the step's largest magnitude scales into
// [2^3, 2^15): no overflow, and values below 2^-18 of the largest still convert to normal halves, whose error is far below
// the largest product's rounding. Otherwise the scale moves to the step's and the return value is the factor that converts the
// block's accumulators (new scale / old, a power of two; 1 when unchanged).
float split_b(float bv[B_ITEMS][8], uint tid, out uint4 bh[B_ITEMS], out uint4 bl[B_ITEMS], inout float cur)
{
    float m = 0;
    [ForceUnroll]
    for (uint it = 0; it < B_ITEMS; it++)
    {
        [ForceUnroll]
        for (uint j = 0; j < 8; j++)
        {
            m = max(m, abs(bv[it][j]));
        }
    }
    uint lane = tid % 32;
    [ForceUnroll]
    for (uint d = 1; d < 16 * B_TPP; d *= 2)
    {
        m = max(m, WaveReadLaneAt(m, lane ^ d));
    }
#if HI_ONLY
    float factor = 1.0;
    float scaled = m * cur;
    if (m > 0 && (cur == 0 || scaled >= 32768.0 || scaled < 8.0))
    {
        float s = split_scale(m);
        factor = cur == 0 ? 1.0 : s / cur;
        cur = s;
    }
    float s = cur == 0 ? 1.0 : cur;
#else
    float s = split_scale(m);
#endif
    [ForceUnroll]
    for (uint it = 0; it < B_ITEMS; it++)
    {
        bh[it] = split8_scaled(bv[it], s, bl[it]);
    }
#if HI_ONLY
    return factor;
#else
    return 1.0 / s;
#endif
}
#endif

Acc mma(MatA a, MatB b, Acc c) { return coopMatMulAdd<float, false>(a, b, c); }

[shader("compute")]
[numthreads(NT, 1, 1)]
void conv_cm(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform ConvParams p)
{
    const uint tid = ltid.x;
    const uint sg = tid / 32;
    const uint g = gid.z / SPLITK;
    const uint split = gid.z % SPLITK;
    const uint HW = HO * WO;
    const uint NC = p.N * HW;
    const uint mBase = gid.y * TBM;
    const uint colBase = gid.x * TBN;
    const uint kBeg = split * KPS;
    const uint kEnd = min(uint(KTOT), kBeg + KPS);
    const uint wm = (sg / WARPS_N) * (FRAG_M * 16);
    const uint wn = (sg % WARPS_N) * (FRAG_N * 16);

    // im2col: the thread stages pixel column bcol of the tile.
#if PARTS == 4
    const uint bcol = tid / B_TPP;
    const uint bq0 = tid % B_TPP;
#else
    const uint bcol = tid % TBN;
    const uint bq0 = tid / TBN;
#endif
    const uint bj = colBase + bcol;
    const bool bvalid = bj < NC;
    const uint bn = bj / HW;
    const uint bpix = bj - bn * HW;
    const uint boh = bpix / WO;
    const uint bow = bpix - boh * WO;
    const int ih0 = int(boh) * SH - PH;
    const int iw0 = int(bow) * SW - PW;
    float* xb = p.x + uint64_t(bn) * XS_N + uint64_t(g * CG) * XS_C;

#if CONV_FRAG8
    Acc8 acc8[2 * FRAG_M][2 * FRAG_N];
    [ForceUnroll] for (int i = 0; i < 2 * FRAG_M; i++) [ForceUnroll] for (int j = 0; j < 2 * FRAG_N; j++) acc8[i][j] = Acc8(0.0);
#endif
#if !CONV_FRAG8
    Acc acc[FRAG_M][FRAG_N];
#if PARTS > 1 && PARTS < 4
    // Correction terms accumulate separately by magnitude (independent dependency chains for the matrix units, and the
    // small terms are not rounded away against the large ones): PARTS 2: cor = hi*lo + lo*hi; PARTS 3: cor = hi*mid + mid*hi,
    // cor2 = mid*mid + hi*lo + lo*hi.
    Acc cor[FRAG_M][FRAG_N];
#endif
#if PARTS == 3
    Acc cor2[FRAG_M][FRAG_N];
#endif
    [ForceUnroll]
    for (int i = 0; i < FRAG_M; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j++)
        {
            acc[i][j] = Acc(0.0);
#if PARTS > 1 && PARTS < 4
            cor[i][j] = Acc(0.0);
#endif
#if PARTS == 3
            cor2[i][j] = Acc(0.0);
#endif
        }
    }
#endif

    uint4 ah[A_ITEMS];
    uint4 am[A_ITEMS];
    uint4 al[A_ITEMS];
    uint4 bh[B_ITEMS];
    uint4 bm[B_ITEMS];
    uint4 bl[B_ITEMS];
    float bv[B_ITEMS][8];
    float bcur = 0;   // HI_ONLY: the current scale of the thread's 16-pixel block (see split_b)
    fetch(p, g, mBase, kBeg, kEnd, tid, bq0, bvalid, ih0, iw0, xb, ah, am, al, bh, bm, bl, bv);
    for (uint k0 = kBeg; k0 < kEnd; k0 += TBK)
    {
#if PARTS == 4
        float bsc = split_b(bv, tid, bh, bl, bcur);
#endif
        [ForceUnroll]
        for (uint it = 0; it < A_ITEMS; it++)
        {
            uint idx = tid + it * NT;
            uint o = (idx / KG) * (SAS / 8) + idx % KG;
#if CONV_FRAG8
            PUT8H(sAh, o * 8, ah[it])
#else
            sAh[o] = ah[it];
#endif
#if HAS_LO
            sAl[o] = al[it];
#endif
#if PARTS == 3
            sAm[o] = am[it];
#endif
        }
        [ForceUnroll]
        for (uint it = 0; it < B_ITEMS; it++)
        {
            uint o = bcol * (SAS / 8) + bq0 + it * B_STRIDE;
#if CONV_FRAG8
            PUT8H(sBh, o * 8, bh[it])
#else
            sBh[o] = bh[it];
#endif
#if HAS_LO
            sBl[o] = bl[it];
#endif
#if PARTS == 3
            sBm[o] = bm[it];
#endif
        }
#if PARTS == 4
        if (bcol % 16 == 0 && bq0 == 0)
        {
            sScB[bcol / 16] = bsc;
        }
#endif
        GroupMemoryBarrierWithGroupSync();
        if (k0 + TBK < kEnd)
        {
            fetch(p, g, mBase, k0 + TBK, kEnd, tid, bq0, bvalid, ih0, iw0, xb, ah, am, al, bh, bm, bl, bv);
        }

#if PARTS == 4 && HI_ONLY
        // One product per step, accumulated on the tensor cores across steps like TF32; a block whose scale moved converts its
        // accumulators first (sScB holds the factors).
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j++)
        {
            float f = sScB[(wn + j * 16) / 16];
            [ForceUnroll]
            for (int i = 0; i < FRAG_M; i++)
            {
                acc[i][j] = acc[i][j] * f;
            }
        }
        [ForceUnroll]
        for (uint kk = 0; kk < TBK; kk += 16)
        {
            MatB b[FRAG_N];
            [ForceUnroll]
            for (int j = 0; j < FRAG_N; j++)
            {
                b[j] = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sBh, ((wn + j * 16) * SAS + kk) / 8, SAS / 8);
            }
            [ForceUnroll]
            for (int i = 0; i < FRAG_M; i++)
            {
                MatA a = MatA.Load<CoopMatMatrixLayout.RowMajor>(sAh, ((wm + i * 16) * SAS + kk) / 8, SAS / 8);
                [ForceUnroll]
                for (int j = 0; j < FRAG_N; j++)
                {
                    acc[i][j] = mma(a, b[j], acc[i][j]);
                }
            }
        }
#elif PARTS == 4
        Acc t[FRAG_M][FRAG_N];
        [ForceUnroll]
        for (uint kk = 0; kk < TBK; kk += 16)
        {
            MatB b[FRAG_N];
#if !HI_ONLY
            MatB bLo[FRAG_N];
#endif
            [ForceUnroll]
            for (int j = 0; j < FRAG_N; j++)
            {
                uint ob = ((wn + j * 16) * SAS + kk) / 8;
                b[j] = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sBh, ob, SAS / 8);
#if !HI_ONLY
                bLo[j] = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sBl, ob, SAS / 8);
#endif
            }
            [ForceUnroll]
            for (int i = 0; i < FRAG_M; i++)
            {
                uint oa = ((wm + i * 16) * SAS + kk) / 8;
                MatA a = MatA.Load<CoopMatMatrixLayout.RowMajor>(sAh, oa, SAS / 8);
#if HI_ONLY
                [ForceUnroll]
                for (int j = 0; j < FRAG_N; j++)
                {
                    t[i][j] = mma(a, b[j], kk == 0 ? Acc(0.0) : t[i][j]);
                }
#else
                MatA aLo = MatA.Load<CoopMatMatrixLayout.RowMajor>(sAl, oa, SAS / 8);
                [ForceUnroll]
                for (int j = 0; j < FRAG_N; j++)
                {
                    t[i][j] = mma(aLo, b[j], kk == 0 ? Acc(0.0) : t[i][j]);
                    t[i][j] = mma(a, bLo[j], t[i][j]);
                    t[i][j] = mma(a, b[j], t[i][j]);
                }
#endif
            }
        }
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j++)
        {
            float r = sScB[(wn + j * 16) / 16];
            [ForceUnroll] for (int i = 0; i < FRAG_M; i++) { acc[i][j] = acc[i][j] + t[i][j] * r; }
        }
#elif CONV_FRAG8
        [ForceUnroll]
        for (uint kk = 0; kk < TBK; kk += 8)
        {
            Mat8B b[2 * FRAG_N];
            [ForceUnroll] for (int j = 0; j < 2 * FRAG_N; j++) b[j] = Mat8B.Load<CoopMatMatrixLayout.ColumnMajor>(sBh, (wn + j * 8) * SAS + kk, SAS);
            [ForceUnroll]
            for (int i = 0; i < 2 * FRAG_M; i++)
            {
                Mat8A a = Mat8A.Load<CoopMatMatrixLayout.RowMajor>(sAh, (wm + i * 8) * SAS + kk, SAS);
                [ForceUnroll] for (int j = 0; j < 2 * FRAG_N; j++) acc8[i][j] = coopMatMulAdd<float, false>(a, b[j], acc8[i][j]);
            }
        }
#else
        [ForceUnroll]
        for (uint kk = 0; kk < TBK; kk += 16)
        {
            MatB b[FRAG_N];
#if PARTS > 1
            MatB bLo[FRAG_N];
#endif
#if PARTS > 2
            MatB bMid[FRAG_N];
#endif
            [ForceUnroll]
            for (int j = 0; j < FRAG_N; j++)
            {
                uint ob = ((wn + j * 16) * SAS + kk) / 8;
                b[j] = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sBh, ob, SAS / 8);
#if PARTS > 1
                bLo[j] = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sBl, ob, SAS / 8);
#endif
#if PARTS > 2
                bMid[j] = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sBm, ob, SAS / 8);
#endif
            }
            [ForceUnroll]
            for (int i = 0; i < FRAG_M; i++)
            {
                uint oa = ((wm + i * 16) * SAS + kk) / 8;
                MatA a = MatA.Load<CoopMatMatrixLayout.RowMajor>(sAh, oa, SAS / 8);
#if PARTS > 1
                MatA aLo = MatA.Load<CoopMatMatrixLayout.RowMajor>(sAl, oa, SAS / 8);
#endif
#if PARTS > 2
                MatA aMid = MatA.Load<CoopMatMatrixLayout.RowMajor>(sAm, oa, SAS / 8);
#endif
                [ForceUnroll]
                for (int j = 0; j < FRAG_N; j++)
                {
                    acc[i][j] = mma(a, b[j], acc[i][j]);
#if PARTS > 2
                    cor[i][j] = mma(aMid, b[j], mma(a, bMid[j], cor[i][j]));
                    cor2[i][j] = mma(aLo, b[j], mma(a, bLo[j], mma(aMid, bMid[j], cor2[i][j])));
#elif PARTS > 1
                    cor[i][j] = mma(aLo, b[j], mma(a, bLo[j], cor[i][j]));
#endif
                }
            }
        }
#endif
        GroupMemoryBarrierWithGroupSync();
    }

    // ---- epilogue: each subgroup stages one accumulator at a time in its shared slice; guarded store (+ bias) ----
#if PARTS == 4 && HI_ONLY
    if (bcol % 16 == 0 && bq0 == 0)
    {
        sInvB[bcol / 16] = bcur == 0 ? 1.0 : 1.0 / bcur;
    }
#endif
    uint64_t outBase = uint64_t(split) * uint64_t(p.N) * (O * HW);
    const uint lane = tid % 32;
    [ForceUnroll]
    for (int i = 0; i < FRAG_M; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j++)
        {
#if PARTS == 3
            Acc c = acc[i][j] + (cor[i][j] + cor2[i][j]);
#elif PARTS == 2
            Acc c = acc[i][j] + cor[i][j];
#elif CONV_FRAG8
            [ForceUnroll] for (int ii = 0; ii < 2; ii++) [ForceUnroll] for (int jj = 0; jj < 2; jj++)
                acc8[2 * i + ii][2 * j + jj].Store<CoopMatMatrixLayout.RowMajor>(sC, sg * 256 + ii * 8 * 16 + jj * 8, 16u);
#else
            Acc c = acc[i][j];
#endif
#if !CONV_FRAG8
            c.Store<CoopMatMatrixLayout.RowMajor>(sC, sg * 256, 16u);
#endif
            GroupMemoryBarrierWithGroupSync();
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
                uint w = lane + e * 32;
                uint m = mBase + wm + i * 16 + w / 16;
                uint col = colBase + wn + j * 16 + w % 16;
                if (m < OG && col < NC)
                {
                    uint n = col / HW;
                    uint pix = col - n * HW;
                    uint o = g * OG + m;
                    float v = sC[sg * 256 + w];
#if PARTS == 4 && HI_ONLY
                    v = v * sInvB[(wn + j * 16) / 16] / p.ws[o];
#elif PARTS == 4
                    v /= p.ws[o];
#endif
                    uint64_t idx = outBase + uint64_t(n) * (O * HW) + uint64_t(o) * HW + pix;
#if SPLITK == 1
#if HAS_BIAS
                    v += p.bias[o];
#endif
                    CONV_EPI(v, idx);
#endif
                    p.y[idx] = v;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
