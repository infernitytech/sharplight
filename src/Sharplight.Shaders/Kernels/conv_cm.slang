// 2-D convolution as an implicit GEMM on cooperative matrices (16x16x16, f32 accumulate):
//   C[o, pix] = sum_k W[g*OG + o, k] * im2col(x)[k, pix] per group g, pix over N*HO*WO.
// Workgroup = 8 subgroups of 32 (2 along o x 4 along pix) computing a 64 (o) x 128 (pix) tile; K in steps of 32.
// PRECISE = 1: every operand is split into bf16 hi + lo and each step accumulates hi*hi + hi*lo + lo*hi (about 16
//              mantissa bits, f32 range): the strict-precision path. Needs bf16 cooperative matrices.
// PRECISE = 0: one f16 product per step (reduced precision). Needs f16 cooperative matrices.
// Same layout and macros as conv_gemm (conv.slang): x [N, C, H, W] strided (XS_*), w [O, CG, KH, KW] dense (WFMT
// f32/f16/bf16), y dense [N, O, HO, WO]. SPLITK > 1 writes partials [SPLITK][N][O][HO*WO] for conv_splitk_reduce.
// Extra macros: PRECISE A_VEC (K % 8 == 0: weights loaded in groups of 8) SPLITK KPS (K per split, multiple of 32).
#include "common.slang"
using namespace linalg;

#define KHW (KH * KW)
#define KTOT (CG * KH * KW)
#define TBM 64
#define TBN 128
#define TBK 32
#define SAS (TBK + 8)
#define SBS (TBN + 8)

#if PRECISE
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

// A: weights [o][k], B: im2col [k][pix] (row-major B), each as hi (and lo) parts.
groupshared Elem sAh[TBM * SAS];
groupshared Elem sBh[TBK * SBS];
#if PRECISE
groupshared Elem sAl[TBM * SAS];
groupshared Elem sBl[TBK * SBS];
#define STORE_A(i, v) { float v_ = (v); Elem h_ = Elem(v_); sAh[i] = h_; sAl[i] = Elem(v_ - float(h_)); }
#define STORE_B(i, v) { float v_ = (v); Elem h_ = Elem(v_); sBh[i] = h_; sBl[i] = Elem(v_ - float(h_)); }
#else
#define STORE_A(i, v) { sAh[i] = Elem(v); }
#define STORE_B(i, v) { sBh[i] = Elem(v); }
#endif
groupshared float sC[8 * 256];

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

Acc step(MatA a, MatB b, Acc c) { return coopMatMulAdd<float, false>(a, b, c); }

// Global loads of one K step: 8 weights (A_VEC: one group of 8 of row tid / 4) and TBK / 2 im2col values of the
// thread's column (k rows bk0, bk0 + 2, ...). Out-of-range elements are zero.
void load_tiles(ConvParams p, uint g, uint mBase, uint k0, uint kEnd, uint tid, bool bvalid, int ih0, int iw0, float* xb,
    out float av[8], out float bv[TBK / 2])
{
#if A_VEC
    {
        uint r = tid / 4;
        uint kk = k0 + (tid % 4) * 8;
        if (mBase + r < OG && kk < kEnd)
        {
            load8(p.w, ROW_BYTES, g * OG + mBase + r, kk / 8, av);
        }
        else
        {
            [ForceUnroll] for (int j = 0; j < 8; j++) av[j] = 0.0;
        }
    }
#else
    [ForceUnroll]
    for (uint j = 0; j < 8; j++)
    {
        uint i = tid + j * 256;
        uint r = i / TBK;
        uint kk = i % TBK;
        av[j] = mBase + r < OG && k0 + kk < kEnd ? cm_weight(p.w, g * OG + mBase + r, k0 + kk) : 0.0;
    }
#endif
    const uint bk0 = tid / TBN;
    [ForceUnroll]
    for (uint it = 0; it < TBK / 2; it++)
    {
        uint k = k0 + bk0 + it * 2;
        float v = 0.0;
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
            v = xb[uint64_t(c) * XS_C + ih * XS_H + iw * XS_W];
#else
            if (uint(ih) < uint(H) && uint(iw) < uint(W))
            {
                v = xb[uint64_t(c) * XS_C + ih * XS_H + iw * XS_W];
            }
#endif
        }
        bv[it] = v;
    }
}

void store_tiles(uint tid, float av[8], float bv[TBK / 2])
{
#if A_VEC
    {
        uint o = (tid / 4) * SAS + (tid % 4) * 8;
        [ForceUnroll] for (int j = 0; j < 8; j++) STORE_A(o + j, av[j]);
    }
#else
    [ForceUnroll]
    for (uint j = 0; j < 8; j++)
    {
        uint i = tid + j * 256;
        STORE_A((i / TBK) * SAS + i % TBK, av[j]);
    }
#endif
    const uint bcol = tid % TBN;
    const uint bk0 = tid / TBN;
    [ForceUnroll]
    for (uint it = 0; it < TBK / 2; it++)
    {
        STORE_B((bk0 + it * 2) * SBS + bcol, bv[it]);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
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
    const uint wm = (sg / 4) * 32;
    const uint wn = (sg % 4) * 32;

    // im2col: the thread owns column tid % TBN of the tile and k rows tid / TBN, + 2, ...
    const uint bj = colBase + tid % TBN;
    const bool bvalid = bj < NC;
    const uint bn = bj / HW;
    const uint bpix = bj - bn * HW;
    const uint boh = bpix / WO;
    const uint bow = bpix - boh * WO;
    const int ih0 = int(boh) * SH - PH;
    const int iw0 = int(bow) * SW - PW;
    float* xb = p.x + uint64_t(bn) * XS_N + uint64_t(g * CG) * XS_C;

    Acc acc00 = Acc(0.0), acc01 = Acc(0.0), acc10 = Acc(0.0), acc11 = Acc(0.0);
#if PRECISE
    // Correction terms (hi*lo + lo*hi) accumulate separately: independent dependency chains for the matrix units.
    Acc cor00 = Acc(0.0), cor01 = Acc(0.0), cor10 = Acc(0.0), cor11 = Acc(0.0);
#endif

    // Global loads of step k0 go to registers first, so the next step's loads overlap this step's products.
    float av[8];
    float bv[TBK / 2];
    load_tiles(p, g, mBase, kBeg, kEnd, tid, bvalid, ih0, iw0, xb, av, bv);
    for (uint k0 = kBeg; k0 < kEnd; k0 += TBK)
    {
        store_tiles(tid, av, bv);
        GroupMemoryBarrierWithGroupSync();
        if (k0 + TBK < kEnd)
        {
            load_tiles(p, g, mBase, k0 + TBK, kEnd, tid, bvalid, ih0, iw0, xb, av, bv);
        }

        [ForceUnroll]
        for (uint kk = 0; kk < TBK; kk += 16)
        {
            MatA a0 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sAh, wm * SAS + kk, SAS);
            MatA a1 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sAh, (wm + 16) * SAS + kk, SAS);
            MatB b0 = MatB.Load<CoopMatMatrixLayout.RowMajor>(sBh, kk * SBS + wn, SBS);
            MatB b1 = MatB.Load<CoopMatMatrixLayout.RowMajor>(sBh, kk * SBS + wn + 16, SBS);
#if PRECISE
            MatA l0 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sAl, wm * SAS + kk, SAS);
            MatA l1 = MatA.Load<CoopMatMatrixLayout.RowMajor>(sAl, (wm + 16) * SAS + kk, SAS);
            MatB m0 = MatB.Load<CoopMatMatrixLayout.RowMajor>(sBl, kk * SBS + wn, SBS);
            MatB m1 = MatB.Load<CoopMatMatrixLayout.RowMajor>(sBl, kk * SBS + wn + 16, SBS);
            cor00 = step(l0, b0, step(a0, m0, cor00));
            cor01 = step(l0, b1, step(a0, m1, cor01));
            cor10 = step(l1, b0, step(a1, m0, cor10));
            cor11 = step(l1, b1, step(a1, m1, cor11));
#endif
            acc00 = step(a0, b0, acc00);
            acc01 = step(a0, b1, acc01);
            acc10 = step(a1, b0, acc10);
            acc11 = step(a1, b1, acc11);
        }
        GroupMemoryBarrierWithGroupSync();
    }

#if PRECISE
    acc00 = acc00 + cor00;
    acc01 = acc01 + cor01;
    acc10 = acc10 + cor10;
    acc11 = acc11 + cor11;
#endif

    // ---- epilogue: stage each accumulator through shared memory, guarded scattered store (+ bias) ----
    uint64_t outBase = uint64_t(split) * uint64_t(p.N) * (O * HW);
    [ForceUnroll]
    for (uint f = 0; f < 4; f++)
    {
        Acc c = f == 0 ? acc00 : f == 1 ? acc01 : f == 2 ? acc10 : acc11;
        c.Store<CoopMatMatrixLayout.RowMajor>(sC, sg * 256, 16u);
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint e = 0; e < 8; e++)
        {
            uint idx = tid + e * 256;
            uint s = idx / 256;
            uint w = idx % 256;
            uint m = mBase + (s / 4) * 32 + (f / 2) * 16 + w / 16;
            uint col = colBase + (s % 4) * 32 + (f % 2) * 16 + w % 16;
            if (m < OG && col < NC)
            {
                uint n = col / HW;
                uint pix = col - n * HW;
                uint o = g * OG + m;
                float v = sC[idx];
#if HAS_BIAS && SPLITK == 1
                v += p.bias[o];
#endif
                p.y[outBase + uint64_t(n) * (O * HW) + uint64_t(o) * HW + pix] = v;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
