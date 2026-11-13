// Row kernels over the innermost axis (one subgroup per row): Softmax, LayerNorm, ReduceSum/Max, ArgMax; LayerNorm of
// few rows also with one workgroup per row.
// Rows are addressed as [T, R1, D] with strides (S0, S1) elements; outputs are dense.
// Macros: D, R1, S0, S1, SG, ROWS_PER_WG (default 4)
#include "common.slang"

#ifndef ROWS_PER_WG
#define ROWS_PER_WG 4
#endif

#ifdef KERNEL_SOFTMAX
// y = softmax(x * SCALE) (LOG_SOFTMAX: log-softmax)
struct SoftmaxParams { float* x; float* y; uint T; };

[shader("compute")]
[numthreads(SG * ROWS_PER_WG, 1, 1)]
void softmax(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform SoftmaxParams p)
{
    uint lane = ltid.x % SG;
    uint row = gid.x * ROWS_PER_WG + ltid.x / SG;
    if (row >= p.T * R1) return;
    uint t = row / R1, r1 = row % R1;
    float* xr = p.x + uint64_t(t) * S0 + uint64_t(r1) * S1;
    float* yr = p.y + uint64_t(row) * D;
    float m = -3.4028235e38;
    for (uint i = lane; i < D; i += SG) m = max(m, xr[i] * SCALE);
    m = WaveActiveMax(m);
    float s = 0;
    for (uint i = lane; i < D; i += SG) s += exp(xr[i] * SCALE - m);
    s = WaveActiveSum(s);
#if LOG_SOFTMAX
    float ls = log(s);
    for (uint i = lane; i < D; i += SG) yr[i] = xr[i] * SCALE - m - ls;
#else
    float inv = 1.0 / s;
    for (uint i = lane; i < D; i += SG) yr[i] = exp(xr[i] * SCALE - m) * inv;
#endif
}
#endif

#ifdef KERNEL_LAYERNORM
// y = (x - mean) / sqrt(var + EPS) * w + b   (HAS_W / HAS_B)
struct LnParams { float* x; float* w; float* b; float* y; uint T; };

[shader("compute")]
[numthreads(SG * ROWS_PER_WG, 1, 1)]
void layernorm(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform LnParams p)
{
    uint lane = ltid.x % SG;
    uint row = gid.x * ROWS_PER_WG + ltid.x / SG;
    if (row >= p.T * R1) return;
    uint t = row / R1, r1 = row % R1;
    float* xr = p.x + uint64_t(t) * S0 + uint64_t(r1) * S1;
    float* yr = p.y + uint64_t(row) * D;
    float s = 0;
    for (uint i = lane; i < D; i += SG) s += xr[i];
    float mean = WaveActiveSum(s) / float(D);
    float v = 0;
    for (uint i = lane; i < D; i += SG) { float d = xr[i] - mean; v += d * d; }
    float rstd = rsqrt(WaveActiveSum(v) / float(D) + EPS);
    for (uint i = lane; i < D; i += SG)
    {
        float r = (xr[i] - mean) * rstd;
#if HAS_W
        r *= p.w[i];
#endif
#if HAS_B
        r += p.b[i];
#endif
        yr[i] = r;
    }
}
#endif

#ifdef KERNEL_LAYERNORM_WG
// LayerNorm of few rows (decode): one 256-thread workgroup per row, the row held in registers between the passes
// (a single subgroup walking a few-thousand-wide row is latency bound). Same math and macros as layernorm.
#define LN_THREADS 256
#define LN_PER ((D + LN_THREADS - 1) / LN_THREADS)
struct LnParams { float* x; float* w; float* b; float* y; uint T; };
groupshared float lnPart[LN_THREADS];

float ln_block_sum(float v, uint tid)
{
    v = WaveActiveSum(v);
    uint lanes = WaveGetLaneCount();
    if (WaveIsFirstLane()) lnPart[tid / lanes] = v;
    GroupMemoryBarrierWithGroupSync();
    float total = 0;
    for (uint i = 0; i < LN_THREADS / lanes; i++) total += lnPart[i];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[shader("compute")]
[numthreads(LN_THREADS, 1, 1)]
void layernorm_wg(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform LnParams p)
{
    uint tid = ltid.x;
    uint row = gid.x;
    uint t = row / R1, r1 = row % R1;
    float* xr = p.x + uint64_t(t) * S0 + uint64_t(r1) * S1;
    float* yr = p.y + uint64_t(row) * D;
    float v[LN_PER];
    float s = 0;
    [ForceUnroll]
    for (uint k = 0; k < LN_PER; k++)
    {
        uint i = tid + k * LN_THREADS;
        v[k] = i < D ? xr[i] : 0.0;
        s += v[k];
    }
    float mean = ln_block_sum(s, tid) / float(D);
    float q = 0;
    [ForceUnroll]
    for (uint k = 0; k < LN_PER; k++)
    {
        float d = v[k] - mean;
        q += tid + k * LN_THREADS < D ? d * d : 0.0;
    }
    float rstd = rsqrt(ln_block_sum(q, tid) / float(D) + EPS);
    [ForceUnroll]
    for (uint k = 0; k < LN_PER; k++)
    {
        uint i = tid + k * LN_THREADS;
        if (i < D)
        {
            float r = (v[k] - mean) * rstd;
#if HAS_W
            r *= p.w[i];
#endif
#if HAS_B
            r += p.b[i];
#endif
            yr[i] = r;
        }
    }
}
#endif

#ifdef KERNEL_ROWREDUCE_NARROW
// Reduction of short rows (D up to 32, e.g. the k expert slots of a MoE block): one thread per row, with an inner
// stride SI so permuted views ([T, H, k] over [T, k, H]) need no dense copy and neighbouring threads read neighbouring
// addresses. MODE as in rowreduce.
struct RrParams { float* x; float* y; uint T; };

[shader("compute")]
[numthreads(256, 1, 1)]
void rowreduce_narrow(uint3 tid : SV_DispatchThreadID, uniform RrParams p)
{
    uint row = tid.x;
    if (row >= p.T * R1) return;
    uint t = row / R1, r1 = row % R1;
    float* xr = p.x + uint64_t(t) * S0 + uint64_t(r1) * S1;
#if MODE == 0
    float s = 0;
    [ForceUnroll] for (uint i = 0; i < D; i++) s += xr[uint64_t(i) * SI];
    p.y[row] = s;
#elif MODE == 1
    float m = -3.4028235e38;
    [ForceUnroll] for (uint i = 0; i < D; i++) m = max(m, xr[uint64_t(i) * SI]);
    p.y[row] = m;
#else
    float m = xr[0];
    uint best = 0;
    [ForceUnroll] for (uint i = 1; i < D; i++) { float v = xr[uint64_t(i) * SI]; if (v > m) { m = v; best = i; } }
    ((uint*)p.y)[row] = best;
#endif
}
#endif

#ifdef KERNEL_ROWREDUCE
// MODE 0 = sum, 1 = max, 2 = argmax (i32 output, first maximum). Output dense [T*R1].
struct RrParams { float* x; float* y; uint T; };

[shader("compute")]
[numthreads(SG * ROWS_PER_WG, 1, 1)]
void rowreduce(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform RrParams p)
{
    uint lane = ltid.x % SG;
    uint row = gid.x * ROWS_PER_WG + ltid.x / SG;
    if (row >= p.T * R1) return;
    uint t = row / R1, r1 = row % R1;
    float* xr = p.x + uint64_t(t) * S0 + uint64_t(r1) * S1;
#if MODE == 0
    float s = 0;
    for (uint i = lane; i < D; i += SG) s += xr[i];
    s = WaveActiveSum(s);
    if (lane == 0) p.y[row] = s;
#elif MODE == 1
    float m = -3.4028235e38;
    for (uint i = lane; i < D; i += SG) m = max(m, xr[i]);
    m = WaveActiveMax(m);
    if (lane == 0) p.y[row] = m;
#else
    float m = -3.4028235e38;
    uint bi = 0xFFFFFFFFu;
    for (uint i = lane; i < D; i += SG) { float v = xr[i]; if (v > m || bi == 0xFFFFFFFFu) { m = v; bi = i; } }
    float gm = WaveActiveMax(m);
    uint cand = (m == gm) ? bi : 0xFFFFFFFFu;
    uint best = WaveActiveMin(cand);
    if (lane == 0) ((uint*)p.y)[row] = best;
#endif
}
#endif
