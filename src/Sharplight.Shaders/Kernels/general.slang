// Kernels for general-network ops: GroupNorm, ConvTranspose, GridSample.
#include "common.slang"

#ifdef KERNEL_GROUPNORM
// x dense [N, C, S]; one workgroup (256 threads) per (n, group). Macros: C S G EPS HAS_SCALE HAS_BIAS
struct GnParams { float* x; float* scale; float* bias; float* y; uint N; };
groupshared float sSum[256 / SG];

float block_sum(float v, uint tid)
{
    v = WaveActiveSum(v);
    if (tid % SG == 0) sSum[tid / SG] = v;
    GroupMemoryBarrierWithGroupSync();
    float r = 0;
    for (uint i = 0; i < 256 / SG; i++) r += sSum[i];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void groupnorm(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform GnParams p)
{
    const uint CG = C / G;
    const uint E = CG * S;
    uint tid = ltid.x;
    uint n = gid.x / G, g = gid.x % G;
    uint64_t base = (uint64_t(n) * C + uint64_t(g) * CG) * S;
    float* xr = p.x + base;
    float* yr = p.y + base;
    float s = 0;
    for (uint i = tid; i < E; i += 256) s += xr[i];
    float mean = block_sum(s, tid) / float(E);
    float v = 0;
    for (uint i = tid; i < E; i += 256) { float d = xr[i] - mean; v += d * d; }
    float rstd = rsqrt(block_sum(v, tid) / float(E) + EPS);
    for (uint i = tid; i < E; i += 256)
    {
        float r = (xr[i] - mean) * rstd;
        uint ch = g * CG + i / S;
#if HAS_SCALE
        r *= p.scale[ch];
#endif
#if HAS_BIAS
        r += p.bias[ch];
#endif
        yr[i] = r;
    }
}
#endif

#ifdef KERNEL_CONV_TRANSPOSE
// y[n,o,oh,ow] = bias[o] + sum_{c in group, kh, kw} x[n,c,ih,iw] * w[c, o%OG, kh, kw] with oh + PH = ih*SH + kh*DH.
// Macros: C H W O HO WO CG OG G KH KW SH SW PH PW DH DW XS_N XS_C XS_H XS_W WFMT HAS_BIAS
struct CtParams { float* x; uint8_t* w; float* bias; float* y; uint total; };

float ct_weight(uint8_t* base, uint idx)
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
void conv_transpose(uint3 dtid : SV_DispatchThreadID, uniform CtParams p)
{
    uint idx = dtid.x;
    if (idx >= p.total) return;
    uint ow = idx % WO;
    uint t = idx / WO;
    uint oh = t % HO;
    uint no = t / HO;
    uint o = no % O, n = no / O;
    uint g = o / OG, ol = o % OG;
    float acc = 0.0;
    for (int kh = 0; kh < KH; kh++)
    {
        int th = int(oh) + PH - kh * DH;
        if (th < 0 || th % SH != 0) continue;
        int ih = th / SH;
        if (ih >= H) continue;
        for (int kw = 0; kw < KW; kw++)
        {
            int tw = int(ow) + PW - kw * DW;
            if (tw < 0 || tw % SW != 0) continue;
            int iw = tw / SW;
            if (iw >= W) continue;
            for (uint cg = 0; cg < CG; cg++)
            {
                uint c = g * CG + cg;
                acc = fma(p.x[uint64_t(n) * XS_N + uint64_t(c) * XS_C + ih * XS_H + iw * XS_W],
                          ct_weight(p.w, ((c * OG + ol) * KH + kh) * KW + kw), acc);
            }
        }
    }
#if HAS_BIAS
    acc += p.bias[o];
#endif
    p.y[idx] = acc;
}
#endif

#ifdef KERNEL_GRID_SAMPLE
// x [N,C,H,W] strided, grid dense [N,HO,WO,2], y dense [N,C,HO,WO].
// Macros: C H W HO WO XS_* MODE (0 linear 1 nearest 2 cubic) PAD (0 zeros 1 border 2 reflection) ALIGN
struct GsParams { float* x; float* grid; float* y; uint total; };

float gs_denorm(float n, int len)
{
#if ALIGN
    return (n + 1.0) * 0.5 * float(len - 1);
#else
    return ((n + 1.0) * float(len) - 1.0) * 0.5;
#endif
}

float gs_reflect(float x, float lo, float hi)
{
    if (hi == lo) return lo;
    float r = hi - lo;
    float d = abs(x - lo);
    float extra = fmod(d, r);
    int flips = int(floor(d / r));
    return (flips % 2 == 0) ? extra + lo : hi - extra;
}

// Applies the padding mode to a coordinate (border / reflection); zeros leave it unchanged.
float gs_pad(float x, int len)
{
#if PAD == 1
    return clamp(x, 0.0, float(len - 1));
#elif PAD == 2
#if ALIGN
    x = gs_reflect(x, 0.0, float(len - 1));
#else
    x = gs_reflect(x, -0.5, float(len) - 0.5);
#endif
    return clamp(x, 0.0, float(len - 1));
#else
    return x;
#endif
}

float gs_tap(float* xp, int yy, int xx)
{
    if (yy < 0 || yy >= H || xx < 0 || xx >= W) return 0.0;
    return xp[yy * XS_H + xx * XS_W];
}

float cubic_coeff(float t, int k)
{
    const float A = -0.75;
    if (k == 0) return ((A * (t + 1.0) - 5.0 * A) * (t + 1.0) + 8.0 * A) * (t + 1.0) - 4.0 * A;
    if (k == 1) return ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0;
    if (k == 2) { float u = 1.0 - t; return ((A + 2.0) * u - (A + 3.0)) * u * u + 1.0; }
    float u = 2.0 - t;
    return ((A * u - 5.0 * A) * u + 8.0 * A) * u - 4.0 * A;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void grid_sample(uint3 dtid : SV_DispatchThreadID, uniform GsParams p)
{
    uint idx = dtid.x;
    if (idx >= p.total) return;
    uint ox = idx % WO;
    uint t = idx / WO;
    uint oy = t % HO;
    uint nc = t / HO;
    uint c = nc % C, n = nc / C;
    float* xp = p.x + uint64_t(n) * XS_N + uint64_t(c) * XS_C;
    float* gp = p.grid + (uint64_t(n) * HO * WO + uint64_t(oy) * WO + ox) * 2;
    float fx = gs_denorm(gp[0], W), fy = gs_denorm(gp[1], H);
#if MODE == 1
    fx = gs_pad(fx, W); fy = gs_pad(fy, H);
    int ix = int(round(fx / 1.0)), iy = int(round(fy / 1.0));
    // nearbyint: round half to even
    float rx = floor(fx + 0.5), ry = floor(fy + 0.5);
    if (rx - fx == 0.5 && fmod(rx, 2.0) != 0.0) rx -= 1.0;
    if (ry - fy == 0.5 && fmod(ry, 2.0) != 0.0) ry -= 1.0;
    p.y[idx] = gs_tap(xp, int(ry), int(rx));
#elif MODE == 2
    float x0f = floor(fx), y0f = floor(fy);
    float tx = fx - x0f, ty = fy - y0f;
    int x0 = int(x0f), y0 = int(y0f);
    float acc = 0.0;
    for (int j = 0; j < 4; j++)
    {
        float row = 0.0;
        for (int i = 0; i < 4; i++)
        {
            // each tap coordinate goes through the padding mode
            float cx = gs_pad(float(x0 - 1 + i), W), cy = gs_pad(float(y0 - 1 + j), H);
            row += cubic_coeff(tx, i) * gs_tap(xp, int(cy), int(cx));
        }
        acc += cubic_coeff(ty, j) * row;
    }
    p.y[idx] = acc;
#else
    fx = gs_pad(fx, W); fy = gs_pad(fy, H);
    float x0f = floor(fx), y0f = floor(fy);
    int x0 = int(x0f), y0 = int(y0f);
    float dx = fx - x0f, dy = fy - y0f;
    float v00 = gs_tap(xp, y0, x0), v01 = gs_tap(xp, y0, x0 + 1);
    float v10 = gs_tap(xp, y0 + 1, x0), v11 = gs_tap(xp, y0 + 1, x0 + 1);
    p.y[idx] = (v00 * (1.0 - dx) + v01 * dx) * (1.0 - dy) + (v10 * (1.0 - dx) + v11 * dx) * dy;
#endif
}
#endif
