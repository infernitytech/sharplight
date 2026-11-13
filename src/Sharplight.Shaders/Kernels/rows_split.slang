// Softmax and GroupNorm over long rows, split into chunks of CHUNK elements across workgroups (256 threads each):
// a stats pass writes per-chunk partials, an apply pass combines the partials of its row and writes its chunk.
// Grids are (chunks, rows). Reductions use shared memory only (no subgroup operations).
#include "common.slang"

#define WG 256

groupshared float sRed[WG];
groupshared float sRed2[WG];

// Sum of v over the workgroup (result in every thread).
float wg_sum(float v, uint tid)
{
    sRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sRed[tid] += sRed[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float r = sRed[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Maximum of v over the workgroup (result in every thread).
float wg_max(float v, uint tid)
{
    sRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sRed[tid] = max(sRed[tid], sRed[tid + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float r = sRed[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

#define CHUNKS ((D + CHUNK - 1) / CHUNK)

#if defined(KERNEL_SOFTMAX_STATS) || defined(KERNEL_SOFTMAX_APPLY)
// y = softmax(x * SCALE) (LOG_SOFTMAX: log-softmax) over rows of D. Row r = t * R1 + r1 starts at t * S0 + r1 * S1
// in x (dense rows of D) and at r * D in y. part: [rows][CHUNKS] float2 (chunk max, sum of exp(v - max)).
// Macros: D R1 S0 S1 SCALE LOG_SOFTMAX CHUNK
struct SoftmaxSplitParams { float* x; float* y; float2* part; };

float* sm_row(float* x, uint row) { return x + uint64_t(row / R1) * S0 + uint64_t(row % R1) * S1; }
#endif

#ifdef KERNEL_SOFTMAX_STATS
[shader("compute")]
[numthreads(WG, 1, 1)]
void softmax_stats(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform SoftmaxSplitParams p)
{
    const uint tid = ltid.x;
    const uint row = gid.y;
    float* xr = sm_row(p.x, row);
    const uint beg = gid.x * CHUNK;
    const uint end = min(uint(D), beg + CHUNK);
    float m = -3.4028235e38;
    for (uint i = beg + tid; i < end; i += WG)
    {
        m = max(m, xr[i] * SCALE);
    }
    m = wg_max(m, tid);
    float s = 0.0;
    for (uint i = beg + tid; i < end; i += WG)
    {
        s += exp(xr[i] * SCALE - m);
    }
    s = wg_sum(s, tid);
    if (tid == 0)
    {
        p.part[uint64_t(row) * CHUNKS + gid.x] = float2(m, s);
    }
}
#endif

#ifdef KERNEL_SOFTMAX_APPLY
[shader("compute")]
[numthreads(WG, 1, 1)]
void softmax_apply(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform SoftmaxSplitParams p)
{
    const uint tid = ltid.x;
    const uint row = gid.y;
    float2* pr = p.part + uint64_t(row) * CHUNKS;
    float m = -3.4028235e38;
    for (uint c = tid; c < CHUNKS; c += WG)
    {
        m = max(m, pr[c].x);
    }
    m = wg_max(m, tid);
    float s = 0.0;
    for (uint c = tid; c < CHUNKS; c += WG)
    {
        s += pr[c].y * exp(pr[c].x - m);
    }
    s = wg_sum(s, tid);

    float* xr = sm_row(p.x, row);
    float* yr = p.y + uint64_t(row) * D;
    const uint beg = gid.x * CHUNK;
    const uint end = min(uint(D), beg + CHUNK);
#if LOG_SOFTMAX
    float ls = log(s);
    for (uint i = beg + tid; i < end; i += WG)
    {
        yr[i] = xr[i] * SCALE - m - ls;
    }
#else
    float inv = 1.0 / s;
    for (uint i = beg + tid; i < end; i += WG)
    {
        yr[i] = exp(xr[i] * SCALE - m) * inv;
    }
#endif
}
#endif

#if defined(KERNEL_GROUPNORM_STATS) || defined(KERNEL_GROUPNORM_APPLY)
// GroupNorm of x dense [N, C, S] (G groups of C / G channels; row = n * G + g holds E = C / G * S elements).
// part: [rows][CHUNKS] float2 (chunk mean, sum of squared deviations from it), combined with Chan's formula.
// Macros: C S G EPS HAS_SCALE HAS_BIAS CHUNK
#define E (C / G * S)
#define D E
struct GnSplitParams { float* x; float* scale; float* bias; float* y; float2* part; };
#endif

#ifdef KERNEL_GROUPNORM_STATS
[shader("compute")]
[numthreads(WG, 1, 1)]
void groupnorm_stats(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform GnSplitParams p)
{
    const uint tid = ltid.x;
    const uint row = gid.y;
    float* xr = p.x + uint64_t(row) * E;
    const uint beg = gid.x * CHUNK;
    const uint end = min(uint(E), beg + CHUNK);
    float s = 0.0;
    for (uint i = beg + tid; i < end; i += WG)
    {
        s += xr[i];
    }
    float mean = wg_sum(s, tid) / float(end - beg);
    float v = 0.0;
    for (uint i = beg + tid; i < end; i += WG)
    {
        float d = xr[i] - mean;
        v += d * d;
    }
    v = wg_sum(v, tid);
    if (tid == 0)
    {
        p.part[uint64_t(row) * CHUNKS + gid.x] = float2(mean, v);
    }
}
#endif

#ifdef KERNEL_GROUPNORM_APPLY
[shader("compute")]
[numthreads(WG, 1, 1)]
void groupnorm_apply(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform GnSplitParams p)
{
    const uint tid = ltid.x;
    const uint row = gid.y;
    float2* pr = p.part + uint64_t(row) * CHUNKS;
    // Mean of the chunk means weighted by chunk sizes, then M2 = sum(M2_c + n_c * (mean_c - mean)^2).
    float s = 0.0;
    for (uint c = tid; c < CHUNKS; c += WG)
    {
        s += pr[c].x * float(min(uint(CHUNK), uint(E) - c * CHUNK));
    }
    float mean = wg_sum(s, tid) / float(E);
    float m2 = 0.0;
    for (uint c = tid; c < CHUNKS; c += WG)
    {
        float d = pr[c].x - mean;
        m2 += pr[c].y + float(min(uint(CHUNK), uint(E) - c * CHUNK)) * d * d;
    }
    float rstd = rsqrt(wg_sum(m2, tid) / float(E) + EPS);

    const uint g = row % G;
    float* xr = p.x + uint64_t(row) * E;
    float* yr = p.y + uint64_t(row) * E;
    const uint beg = gid.x * CHUNK;
    const uint end = min(uint(E), beg + CHUNK);
    for (uint i = beg + tid; i < end; i += WG)
    {
        float r = (xr[i] - mean) * rstd;
#if HAS_SCALE || HAS_BIAS
        uint ch = g * (C / G) + i / S;
#endif
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
