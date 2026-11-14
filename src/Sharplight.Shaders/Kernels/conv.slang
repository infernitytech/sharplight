// 2-D convolution kernels.
//  conv_gemm        : implicit-GEMM (weights [OG x K] times im2col(x) [K x N*HO*WO]) per group, shared-memory
//                     tiles, register micro-tiles, fp32 accumulate. Optional split-K (partials to a workspace).
//  conv_splitk_reduce: sums split-K partials and adds bias.
//  conv_dw          : depthwise / channel-multiplier convolution (groups == C, C/groups == 1).
// x [N, C, H, W] with strides (XS_N, XS_C, XS_H, XS_W); w [O, CG, KH, KW] dense (format WFMT); y dense [N, O, HO, WO].
// Common macros: C H W O HO WO CG OG G KH KW SH SW PH PW DH DW XS_* WFMT ROW_BYTES HAS_BIAS NOPAD
#include "common.slang"
#include "conv_epilogue.slang"

#define KHW (KH * KW)
#define KTOT (CG * KH * KW)

#ifdef KERNEL_CONV_GEMM
// Extra macros: BM BN BK TM TN NT (= (BM/TM)*(BN/TN)) A_VEC SPLITK KPS (K elements per split, multiple of BK)
struct ConvParams
{
    float* x;
    uint8_t* w;
    float* bias;
    float* y;
#if EPI_RES
    float* res;
#endif
    uint N;
};

groupshared float As[BK][BM + 1];
groupshared float Bs[BK][BN + 1];

float w_scalar(uint8_t* base, uint row, uint k)
{
    uint8_t* r = base + uint64_t(ROW_BYTES) * row;
#if WFMT == FMT_F32
    return ((float*)r)[k];
#elif WFMT == FMT_F16
    return h2f(((uint16_t*)r)[k]);
#elif WFMT == FMT_BF16
    return bf2f(((uint16_t*)r)[k]);
#else
    return 0.0;
#endif
}

[shader("compute")]
[numthreads(NT, 1, 1)]
void conv_gemm(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform ConvParams p)
{
    const uint tid = ltid.x;
    const uint g = gid.z / SPLITK;
    const uint split = gid.z % SPLITK;
    const uint HW = HO * WO;
    const uint NC = p.N * HW;
    const uint mBase = gid.y * BM;
    const uint colBase = gid.x * BN;
    const uint kBeg = split * KPS;
    const uint kEnd = min(uint(KTOT), kBeg + KPS);

    // Per-thread im2col column (fixed for the whole kernel since NT % BN == 0).
    const uint bcol = tid % BN;
    const uint bk0 = tid / BN;
    const uint bj = colBase + bcol;
    const bool bvalid = bj < NC;
    const uint bn = bj / HW;
    const uint bpix = bj - bn * HW;
    const uint boh = bpix / WO;
    const uint bow = bpix - boh * WO;
    const int ih0 = int(boh) * SH - PH;
    const int iw0 = int(bow) * SW - PW;
    float* xb = p.x + uint64_t(bn) * XS_N + uint64_t(g * CG) * XS_C;

    float acc[TM][TN];
    [ForceUnroll] for (int i = 0; i < TM; i++)
        [ForceUnroll] for (int j = 0; j < TN; j++) acc[i][j] = 0.0;

    const uint tx = tid % (BN / TN);
    const uint ty = tid / (BN / TN);

    for (uint k0 = kBeg; k0 < kEnd; k0 += BK)
    {
        // ---- A tile (weights) ----
#if A_VEC
                for (uint i = tid; i < (BM * BK) / 8; i += NT)
        {
            uint m = i % BM;
            uint kg = i / BM;
            uint kk = k0 + kg * 8;
            float v[8];
            if (mBase + m < OG && kk < kEnd) load8(p.w, ROW_BYTES, g * OG + mBase + m, kk / 8, v);
            else { [ForceUnroll] for (int j = 0; j < 8; j++) v[j] = 0.0; }
            [ForceUnroll] for (int j = 0; j < 8; j++) As[kg * 8 + j][m] = v[j];
        }
#else
                for (uint i = tid; i < BM * BK; i += NT)
        {
            uint m = i % BM;
            uint kk = i / BM;
            float v = 0.0;
            if (mBase + m < OG && k0 + kk < kEnd) v = w_scalar(p.w, g * OG + mBase + m, k0 + kk);
            As[kk][m] = v;
        }
#endif
        // ---- B tile (im2col of x) ----
        [ForceUnroll]
        for (uint it = 0; it < (BK * BN) / NT; it++)
        {
            uint kk = bk0 + it * (NT / BN);
            uint k = k0 + kk;
            float v = 0.0;
            if (bvalid && k < kEnd)
            {
#if KHW == 1
                uint c = k;
                int ih = ih0, iw = iw0;
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
                    v = xb[uint64_t(c) * XS_C + ih * XS_H + iw * XS_W];
#endif
            }
            Bs[kk][bcol] = v;
        }
        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll]
        for (uint kk = 0; kk < BK; kk++)
        {
            float a[TM], b[TN];
            [ForceUnroll] for (int i = 0; i < TM; i++) a[i] = As[kk][ty * TM + i];
            [ForceUnroll] for (int j = 0; j < TN; j++) b[j] = Bs[kk][tx * TN + j];
            [ForceUnroll] for (int i = 0; i < TM; i++)
                [ForceUnroll] for (int j = 0; j < TN; j++) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // ---- epilogue ----
    uint64_t outBase = uint64_t(split) * uint64_t(p.N) * (O * HW);
    [ForceUnroll]
    for (int j = 0; j < TN; j++)
    {
        uint col = colBase + tx * TN + j;
        if (col >= NC) continue;
        uint n = col / HW;
        uint pix = col - n * HW;
        [ForceUnroll]
        for (int i = 0; i < TM; i++)
        {
            uint m = mBase + ty * TM + i;
            if (m >= OG) continue;
            uint o = g * OG + m;
            float v = acc[i][j];
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
}
#endif

#ifdef KERNEL_CONV_SPLITK_REDUCE
// Macros: SPLITK O HW HAS_BIAS
struct ReduceParams
{
    float* ws;
    float* bias;
    float* y;
#if EPI_RES
    float* res;
#endif
    uint total;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void conv_splitk_reduce(uint3 dtid : SV_DispatchThreadID, uniform ReduceParams p)
{
    uint i = dtid.x;
    if (i >= p.total) return;
    float s = 0.0;
    for (uint k = 0; k < SPLITK; k++) s += p.ws[uint64_t(k) * p.total + i];
#if HAS_BIAS
    s += p.bias[(i / HW) % O];
#endif
    CONV_EPI(s, i);
    p.y[i] = s;
}
#endif

#ifdef KERNEL_CONV_DW
struct DwParams
{
    float* x;
    uint8_t* w;
    float* bias;
    float* y;
#if EPI_RES
    float* res;
#endif
    uint total;
};

float dw_weight(uint8_t* base, uint idx)
{
#if WFMT == FMT_F32
    return ((float*)base)[idx];
#elif WFMT == FMT_F16
    return h2f(((uint16_t*)base)[idx]);
#else
    return bf2f(((uint16_t*)base)[idx]);
#endif
}

[shader("compute")]
[numthreads(256, 1, 1)]
void conv_dw(uint3 dtid : SV_DispatchThreadID, uniform DwParams p)
{
    uint idx = dtid.x;
    if (idx >= p.total) return;
    uint ow = idx % WO;
    uint t = idx / WO;
    uint oh = t % HO;
    uint no = t / HO;
    uint o = no % O, n = no / O;
    uint c = o / OG;
    float* xp = p.x + uint64_t(n) * XS_N + uint64_t(c) * XS_C;
    int ih0 = int(oh) * SH - PH;
    int iw0 = int(ow) * SW - PW;
    float acc = 0.0;
    [ForceUnroll]
    for (int kh = 0; kh < KH; kh++)
    {
        int ih = ih0 + kh * DH;
#if !NOPAD
        if (uint(ih) >= uint(H)) continue;
#endif
        [ForceUnroll]
        for (int kw = 0; kw < KW; kw++)
        {
            int iw = iw0 + kw * DW;
#if !NOPAD
            if (uint(iw) >= uint(W)) continue;
#endif
            acc = fma(xp[ih * XS_H + iw * XS_W], dw_weight(p.w, o * KHW + kh * KW + kw), acc);
        }
    }
#if HAS_BIAS
    acc += p.bias[o];
#endif
    CONV_EPI(acc, idx);
    p.y[idx] = acc;
}
#endif
