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
// Same layout and macros as conv_gemm (conv.slang): x [N, C, H, W] strided (XS_*), w [O, CG, KH, KW] dense (WFMT
// f32/f16/bf16), y dense [N, O, HO, WO]. SPLITK > 1 writes partials [SPLITK][N][O][HO*WO] for conv_splitk_reduce.
// Extra macros: PARTS (1-3) FRAG_N (1 or 2 pixel fragments per subgroup) WARPS_M (1 or 2) A_VEC (K % 8 == 0: weights loaded in groups of 8) SPLITK KPS (K per split,
// multiple of 32).
#include "common.slang"
using namespace linalg;

#define KHW (KH * KW)
#define KTOT (CG * KH * KW)
#define WARPS_N 4
#define FRAG_M 4
#define NT (WARPS_M * WARPS_N * 32)
#define TBM (WARPS_M * FRAG_M * 16)
#define TBN (WARPS_N * FRAG_N * 16)
#define TBK 32
// Row stride of the shared tiles in halves: 32 k plus 8 halves of padding.
#define SAS (TBK + 8)
// Groups of 8 consecutive k per thread and step: A rows x 4 groups, B pixels x 4 groups.
#define A_ITEMS (TBM * 4 / NT)
#define B_ITEMS (TBN * 4 / NT)
#define B_STRIDE (NT / TBN)

#if PARTS > 1
typealias Elem = BFloat16;
#else
typealias Elem = half;
#endif
typealias MatA = CoopMat<Elem, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<Elem, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias Acc = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

struct ConvParams
{
    float* x;
    uint8_t* w;
    float* bias;
    float* y;
    uint N;
};

// A: weights [o][k] (row-major), B: im2col [pix][k] (column-major B); 8 packed operands per uint4, one tile per part.
groupshared uint4 sAh[TBM * SAS / 8];
groupshared uint4 sBh[TBN * SAS / 8];
#if PARTS > 1
groupshared uint4 sAl[TBM * SAS / 8];
groupshared uint4 sBl[TBN * SAS / 8];
#endif
#if PARTS > 2
groupshared uint4 sAm[TBM * SAS / 8];
groupshared uint4 sBm[TBN * SAS / 8];
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
        h[j] = f32tof16(v[j]);
        m[j] = 0;
        l[j] = 0;
#endif
    }
    hi = pack_words(h);
    mid = pack_words(m);
    lo = pack_words(l);
}

// Global loads of the K step at k0 into packed registers; zero outside the tensors and at or past kEnd.
// A item it: row idx / 4, k group idx % 4 (idx = tid + it * NT). B item it: the thread's pixel, k group bq0 + it * B_STRIDE.
void fetch(ConvParams p, uint g, uint mBase, uint k0, uint kEnd, uint tid, uint bq0, bool bvalid, int ih0, int iw0,
    float* xb, out uint4 ah[A_ITEMS], out uint4 am[A_ITEMS], out uint4 al[A_ITEMS], out uint4 bh[B_ITEMS], out uint4 bm[B_ITEMS],
    out uint4 bl[B_ITEMS])
{
    [ForceUnroll]
    for (uint it = 0; it < A_ITEMS; it++)
    {
        uint idx = tid + it * NT;
        uint r = idx / 4;
        uint kk = k0 + (idx % 4) * 8;
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
        pack8(v, bh[it], bm[it], bl[it]);
    }
}

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
    const uint bcol = tid % TBN;
    const uint bq0 = tid / TBN;
    const uint bj = colBase + bcol;
    const bool bvalid = bj < NC;
    const uint bn = bj / HW;
    const uint bpix = bj - bn * HW;
    const uint boh = bpix / WO;
    const uint bow = bpix - boh * WO;
    const int ih0 = int(boh) * SH - PH;
    const int iw0 = int(bow) * SW - PW;
    float* xb = p.x + uint64_t(bn) * XS_N + uint64_t(g * CG) * XS_C;

    Acc acc[FRAG_M][FRAG_N];
#if PARTS > 1
    // Correction terms accumulate separately by magnitude (independent dependency chains for the matrix units, and the
    // small terms are not rounded away against the large ones): PARTS 2: cor = hi*lo + lo*hi; PARTS 3: cor = hi*mid + mid*hi,
    // cor2 = mid*mid + hi*lo + lo*hi.
    Acc cor[FRAG_M][FRAG_N];
#endif
#if PARTS > 2
    Acc cor2[FRAG_M][FRAG_N];
#endif
    [ForceUnroll]
    for (int i = 0; i < FRAG_M; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j++)
        {
            acc[i][j] = Acc(0.0);
#if PARTS > 1
            cor[i][j] = Acc(0.0);
#endif
#if PARTS > 2
            cor2[i][j] = Acc(0.0);
#endif
        }
    }

    uint4 ah[A_ITEMS];
    uint4 am[A_ITEMS];
    uint4 al[A_ITEMS];
    uint4 bh[B_ITEMS];
    uint4 bm[B_ITEMS];
    uint4 bl[B_ITEMS];
    fetch(p, g, mBase, kBeg, kEnd, tid, bq0, bvalid, ih0, iw0, xb, ah, am, al, bh, bm, bl);
    for (uint k0 = kBeg; k0 < kEnd; k0 += TBK)
    {
        [ForceUnroll]
        for (uint it = 0; it < A_ITEMS; it++)
        {
            uint idx = tid + it * NT;
            uint o = (idx / 4) * (SAS / 8) + idx % 4;
            sAh[o] = ah[it];
#if PARTS > 1
            sAl[o] = al[it];
#endif
#if PARTS > 2
            sAm[o] = am[it];
#endif
        }
        [ForceUnroll]
        for (uint it = 0; it < B_ITEMS; it++)
        {
            uint o = bcol * (SAS / 8) + bq0 + it * B_STRIDE;
            sBh[o] = bh[it];
#if PARTS > 1
            sBl[o] = bl[it];
#endif
#if PARTS > 2
            sBm[o] = bm[it];
#endif
        }
        GroupMemoryBarrierWithGroupSync();
        if (k0 + TBK < kEnd)
        {
            fetch(p, g, mBase, k0 + TBK, kEnd, tid, bq0, bvalid, ih0, iw0, xb, ah, am, al, bh, bm, bl);
        }

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
        GroupMemoryBarrierWithGroupSync();
    }

    // ---- epilogue: each subgroup stages one accumulator at a time in its shared slice; guarded store (+ bias) ----
    uint64_t outBase = uint64_t(split) * uint64_t(p.N) * (O * HW);
    const uint lane = tid % 32;
    [ForceUnroll]
    for (int i = 0; i < FRAG_M; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j++)
        {
#if PARTS > 2
            Acc c = acc[i][j] + (cor[i][j] + cor2[i][j]);
#elif PARTS > 1
            Acc c = acc[i][j] + cor[i][j];
#else
            Acc c = acc[i][j];
#endif
            c.Store<CoopMatMatrixLayout.RowMajor>(sC, sg * 256, 16u);
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
#if HAS_BIAS && SPLITK == 1
                    v += p.bias[o];
#endif
                    p.y[outBase + uint64_t(n) * (O * HW) + uint64_t(o) * HW + pix] = v;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
