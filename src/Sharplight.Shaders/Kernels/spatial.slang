// Spatial / data-movement kernels: pooling, padding, resize, general reductions.
// All tensors are f32; inputs may be strided (macros), outputs are dense.
#include "common.slang"

#ifdef KERNEL_POOL
// ---------------------------------------------------------------------------------------------
// 2-D pooling, one thread per output element. x [N, C, H, W] strides (XS_N, XS_C, XS_H, XS_W),
// y dense [N, C, HO, WO]. MODE 0 = max, 1 = avg.
// Macros: C H W HO WO KH KW SH SW PH PW PB PR DH DW XS_* MODE COUNT_PAD
struct PoolParams { float* x; float* y; uint total; };

[shader("compute")]
[numthreads(256, 1, 1)]
void pool(uint3 dtid : SV_DispatchThreadID, uniform PoolParams p)
{
    uint idx = dtid.x;
    if (idx >= p.total) return;
    uint ow = idx % WO;
    uint t = idx / WO;
    uint oh = t % HO;
    uint nc = t / HO;
    uint c = nc % C, n = nc / C;
    float* xp = p.x + uint64_t(n) * XS_N + uint64_t(c) * XS_C;
    int hs = int(oh) * SH - PH;
    int ws = int(ow) * SW - PW;
#if MODE == 0
    float acc = -3.4028235e38;
    for (int kh = 0; kh < KH; kh++)
    {
        int ih = hs + kh * DH;
        if (uint(ih) >= uint(H)) continue;
        for (int kw = 0; kw < KW; kw++)
        {
            int iw = ws + kw * DW;
            if (uint(iw) >= uint(W)) continue;
            acc = max(acc, xp[ih * XS_H + iw * XS_W]);
        }
    }
    p.y[idx] = acc;
#else
    float acc = 0;
    int cnt = 0;
    for (int kh = 0; kh < KH; kh++)
    {
        int ih = hs + kh * DH;
        if (uint(ih) >= uint(H)) continue;
        for (int kw = 0; kw < KW; kw++)
        {
            int iw = ws + kw * DW;
            if (uint(iw) >= uint(W)) continue;
            acc += xp[ih * XS_H + iw * XS_W];
            cnt++;
        }
    }
#if COUNT_PAD
    // taps inside the padded input (ceil_mode overhang is not counted)
    int th = min(KH, (H + PB - 1 - hs) / DH + 1), tw = min(KW, (W + PR - 1 - ws) / DW + 1);
    float div = float(th * tw);
#else
    float div = float(cnt);
#endif
    p.y[idx] = div > 0.0 ? acc / div : 0.0;
#endif
}
#endif

#ifdef KERNEL_PAD
// ---------------------------------------------------------------------------------------------
// Pad: out dense [OSH...], in strided. Per-axis arrays: OSH (out shape), ISH (in shape), IST (in
// strides), PBG (pad begin). MODE 0 = constant (VALUE), 1 = reflect, 2 = edge.
// Macros: RANK, OSH, ISH, IST, PBG (brace-less initializer lists), MODE, VALUE
struct PadParams { float* x; float* y; uint total; };
static const int P_OSH[RANK] = { OSH };
static const int P_ISH[RANK] = { ISH };
static const int P_IST[RANK] = { IST };
static const int P_PBG[RANK] = { PBG };

[shader("compute")]
[numthreads(256, 1, 1)]
void pad(uint3 dtid : SV_DispatchThreadID, uniform PadParams p)
{
    uint idx = dtid.x;
    if (idx >= p.total) return;
    uint rem = idx;
    uint64_t addr = 0;
    bool valid = true;
    [ForceUnroll]
    for (int d = RANK - 1; d >= 0; d--)
    {
        int c;
        if (d > 0) { c = int(rem % uint(P_OSH[d])); rem /= uint(P_OSH[d]); }
        else c = int(rem);
        int s = c - P_PBG[d];
        int n = P_ISH[d];
        if (s < 0 || s >= n)
        {
#if MODE == 0
            valid = false;
#elif MODE == 1
            if (n == 1) s = 0;
            else
            {
                int per = 2 * (n - 1);
                int t = int(uint(abs(s)) % uint(per));
                s = t >= n ? per - t : t;
            }
#else
            s = clamp(s, 0, n - 1);
#endif
        }
        addr += uint64_t(s) * uint64_t(P_IST[d]);
    }
    p.y[idx] = valid ? p.x[addr] : VALUE;
}
#endif

#ifdef KERNEL_RESIZE
// ---------------------------------------------------------------------------------------------
// Resize (nearest / bilinear) following ONNX Runtime semantics (float32 coordinate math).
// x [N, C, H, W] strided, y dense [N, C, OH, OW]. Scales are out/in as float.
// Macros: C H W OH OW XS_* LINEAR COORD NEAREST   (COORD: 0 half_pixel 1 align_corners 2 asymmetric 3 pytorch_half_pixel)
//                                                 (NEAREST: 0 round_prefer_floor 1 round_prefer_ceil 2 floor 3 ceil)
struct ResizeParams { float* x; float* y; uint total; };

float orig_coord(float xr, float scale, float lenR, float lenO)
{
#if COORD == 0
    return (xr + 0.5) / scale - 0.5;
#elif COORD == 1
    return lenR == 1.0 ? 0.0 : xr * (lenO - 1.0) / (lenR - 1.0);
#elif COORD == 2
    return xr / scale;
#else
    return lenR > 1.0 ? (xr + 0.5) / scale - 0.5 : 0.0;
#endif
}

int nearest_pixel(float v)
{
#if NEAREST == 0
    return (v == floor(v) + 0.5) ? int(floor(v)) : int(round(v));
#elif NEAREST == 1
    return (v == floor(v) + 0.5) ? int(ceil(v)) : int(round(v));
#elif NEAREST == 2
    return int(floor(v));
#else
    return int(ceil(v));
#endif
}

[shader("compute")]
[numthreads(256, 1, 1)]
void resize(uint3 dtid : SV_DispatchThreadID, uniform ResizeParams p)
{
    uint idx = dtid.x;
    if (idx >= p.total) return;
    uint ox = idx % OW;
    uint t = idx / OW;
    uint oy = t % OH;
    uint nc = t / OH;
    uint c = nc % C, n = nc / C;
    float* xp = p.x + uint64_t(n) * XS_N + uint64_t(c) * XS_C;
    float sh = float(OH) / float(H), sw = float(OW) / float(W);
    float fy = orig_coord(float(oy), sh, float(OH), float(H));
    float fx = orig_coord(float(ox), sw, float(OW), float(W));
#if LINEAR
    fy = clamp(fy, 0.0, float(H - 1));
    fx = clamp(fx, 0.0, float(W - 1));
    int y0 = int(floor(fy)), x0 = int(floor(fx));
    int y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
    float dy = fy - float(y0), dx = fx - float(x0);
    float v00 = xp[y0 * XS_H + x0 * XS_W], v01 = xp[y0 * XS_H + x1 * XS_W];
    float v10 = xp[y1 * XS_H + x0 * XS_W], v11 = xp[y1 * XS_H + x1 * XS_W];
    float top = v00 * (1.0 - dx) + v01 * dx;
    float bot = v10 * (1.0 - dx) + v11 * dx;
    p.y[idx] = top * (1.0 - dy) + bot * dy;
#else
    int iy = clamp(nearest_pixel(fy), 0, H - 1);
    int ix = clamp(nearest_pixel(fx), 0, W - 1);
    p.y[idx] = xp[iy * XS_H + ix * XS_W];
#endif
}
#endif

#ifdef KERNEL_REDUCE
// ---------------------------------------------------------------------------------------------
// General reduction. Input dims are split into kept dims (KN, shapes KSH / strides KST, outermost
// may be symbolic and is then the quotient) and reduced dims (RN, shapes RSH / strides RST).
// Output dense over the kept dims. TPO threads cooperate on one output (SG = subgroup, 256 = workgroup).
// Macros: KN KSH KST RN RSH RST R_TOTAL TPO (R_TOTAL is p.nRed when a reduced axis is symbolic)
//   MODE: 0 sum 1 mean 2 max 3 min 4 prod 5 l1 6 l2 7 sumsquare 8 logsumexp
struct ReduceParams { float* x; float* y; uint nOut; uint nRed; };
static const int R_KSH[KN] = { KSH };
static const int R_KST[KN] = { KST };
static const int R_RSH[RN] = { RSH };
static const int R_RST[RN] = { RST };

groupshared float sRed[256 / SG];

// CLS: 0 sum, 1 max, 2 min, 3 prod
float wave_cls(float v, int cls)
{
    if (cls == 0) return WaveActiveSum(v);
    if (cls == 1) return WaveActiveMax(v);
    if (cls == 2) return WaveActiveMin(v);
    return WaveActiveProduct(v);
}
float comb_cls(float a, float b, int cls)
{
    if (cls == 0) return a + b;
    if (cls == 1) return max(a, b);
    if (cls == 2) return min(a, b);
    return a * b;
}
float init_cls(int cls)
{
    if (cls == 0) return 0.0;
    if (cls == 1) return -1.0 / 0.0;   // max/min of all -inf (+inf) rows stay infinite (masked attention rows)
    if (cls == 2) return 1.0 / 0.0;
    return 1.0;
}

// Reduces v over the TPO threads of one output; every participating thread gets the result.
float reduce_threads(float v, uint tid, int cls)
{
    v = wave_cls(v, cls);
#if TPO == 256
    if (tid % SG == 0) sRed[tid / SG] = v;
    GroupMemoryBarrierWithGroupSync();
    float r = sRed[0];
    for (uint i = 1; i < 256 / SG; i++) r = comb_cls(r, sRed[i], cls);
    GroupMemoryBarrierWithGroupSync();
    return r;
#else
    return v;
#endif
}

[shader("compute")]
[numthreads(256, 1, 1)]
void reduce(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform ReduceParams p)
{
    uint tid = ltid.x;
#if TPO == 256
    uint o = gid.x;
    uint lane = tid;
#else
    uint o = gid.x * (256 / TPO) + tid / TPO;
    uint lane = tid % TPO;
#endif
    bool active = o < p.nOut;
    uint64_t base = 0;
    if (active)
    {
        uint rem = o;
        [ForceUnroll]
        for (int d = KN - 1; d >= 0; d--)
        {
            uint c;
            if (d > 0) { c = rem % uint(R_KSH[d]); rem /= uint(R_KSH[d]); } else c = rem;
            base += uint64_t(c) * uint64_t(R_KST[d]);
        }
    }
#if MODE == 2
    const int CLS = 1;
#elif MODE == 3
    const int CLS = 2;
#elif MODE == 4
    const int CLS = 3;
#elif MODE == 8
    const int CLS = 1;
#else
    const int CLS = 0;
#endif
#if MODE == 8
    float M = -3.4028235e38;
    for (int pass = 0; pass < 2; pass++)
#else
    const int pass = 0;
#endif
    {
        float acc = init_cls(pass == 0 ? CLS : 0);   // (logsumexp: pass 0 = max, pass 1 = sum)
        if (active)
        {
            for (uint r = lane; r < R_TOTAL; r += TPO)
            {
                uint rem = r;
                uint64_t off = base;
                [ForceUnroll]
                for (int d = RN - 1; d >= 0; d--)
                {
                    uint c = d > 0 ? rem % uint(R_RSH[d]) : rem;
                    if (d > 0) rem /= uint(R_RSH[d]);
                    off += uint64_t(c) * uint64_t(R_RST[d]);
                }
                float v = p.x[off];
#if MODE == 5
                v = abs(v);
#elif MODE == 6 || MODE == 7
                v = v * v;
#elif MODE == 8
                if (pass == 1) v = exp(v - M); else v = v;
#endif
                acc = comb_cls(acc, v, pass == 0 ? CLS : 0);
            }
        }
        acc = reduce_threads(acc, tid, pass == 0 ? CLS : 0);
#if MODE == 8
        if (pass == 0) { M = max(acc, -3.4028235e38); continue; }   // finite shift: an all -inf row gives log(0) = -inf
        acc = M + log(acc);
#endif
#if MODE == 1
        acc /= float(R_TOTAL);
#elif MODE == 6
        acc = sqrt(acc);
#endif
#if TPO == 256
        if (tid == 0 && active) p.y[o] = acc;
#else
        if (lane == 0 && active) p.y[o] = acc;
#endif
    }
}
#endif
