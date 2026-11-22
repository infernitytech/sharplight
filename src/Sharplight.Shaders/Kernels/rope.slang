// Rotary embedding. x: [T, H, D] with strides (S0, S1), inner contiguous; positions: i32 [T, A];
// y: dense [T, H, D]. One thread per (t, h, pair).
// Macros: H, D, S0, S1, A, ROT (rotated dims), THETA_SCALE (= base^(-2/ROT)), FREQ_SCALE, ATTN_FACTOR (cos/sin multiplier, YaRN magnitude),
//         NEOX (1 = pairs (i, i+ROT/2), 0 = pairs (2i, 2i+1)),
//         MROPE (0/1), INTERLEAVED (0/1), SEC0..SEC3 (multi-axis sections),
//         PER_ELEMENT (0/1): the sections index the rotated elements two by two (element e takes the axis of pair section e / 2), so the two
//         elements of a neox pair can follow different axes (XD-RoPE),
//         HAS_FF (0/1): per-frequency divisors ff[ROT/2] (theta_i /= ff[i]; llama3 / yarn / longrope scaling)
#include "common.slang"

struct Params
{
    float* x;
    int* pos;
    float* y;
    float* ff;
    uint T;
};

uint axis_for_pair(uint i)
{
#if MROPE
    const uint secDims = SEC0 + SEC1 + SEC2 + SEC3;
    uint sector = i % secDims;
#if INTERLEAVED
    if (sector % 3 == 1 && sector < 3 * SEC1) return 1;
    if (sector % 3 == 2 && sector < 3 * SEC2) return 2;
    if (sector % 3 == 0 && sector < 3 * SEC0) return 0;
    return 3;
#else
    if (sector < SEC0) return 0;
    if (sector < SEC0 + SEC1) return 1;
    if (sector < SEC0 + SEC1 + SEC2) return 2;
    return 3;
#endif
#else
    return 0;
#endif
}

// The rotation angle of frequency i at the position on an axis.
float angle(int* pos, float* ff, uint t, uint axis, uint i)
{
    float position = float(pos[t * A + axis]);
    float theta = FREQ_SCALE * position * pow(THETA_SCALE, float(i));
#if HAS_FF
    theta /= ff[i];
#endif
    return theta;
}

// cos and sin (times ATTN_FACTOR) of the angles of pair i, whose elements are i0 and i1: (c, s) for i0 and (c2, s2) for i1. Without
// per-element sections both elements share one angle.
void pair_angles(int* pos, float* ff, uint t, uint i, uint i0, uint i1, out float c, out float s, out float c2, out float s2)
{
#if PER_ELEMENT
    float theta = angle(pos, ff, t, axis_for_pair(i0 / 2), i);
    float theta2 = angle(pos, ff, t, axis_for_pair(i1 / 2), i);
#else
    float theta = angle(pos, ff, t, axis_for_pair(i), i);
    float theta2 = theta;
#endif
    c = cos(theta) * ATTN_FACTOR;
    s = sin(theta) * ATTN_FACTOR;
    c2 = cos(theta2) * ATTN_FACTOR;
    s2 = sin(theta2) * ATTN_FACTOR;
}

#if !defined(KERNEL_ROPE_ROW) && !defined(KERNEL_ROPE_QKV)
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uniform Params p)
{
    const uint PAIRS = D / 2;
    uint idx = dtid.x;
    if (idx >= p.T * H * PAIRS) return;
    uint i = idx % PAIRS;
    uint h = (idx / PAIRS) % H;
    uint t = idx / (PAIRS * H);
    float* xr = p.x + uint64_t(t) * S0 + uint64_t(h) * S1;
    float* yr = p.y + (uint64_t(t) * H + h) * D;

    uint i0, i1;
#if NEOX
    if (i < ROT / 2) { i0 = i; i1 = i + ROT / 2; }
    else { uint j = ROT + 2 * (i - ROT / 2); i0 = j; i1 = j + 1; }
#else
    i0 = 2 * i; i1 = 2 * i + 1;
#endif
    float x0 = xr[i0], x1 = xr[i1];
    uint pairIndex = NEOX ? i : i; // frequency index for rotated pairs
    if (pairIndex < ROT / 2)
    {
        float c, s, c2, s2;
        pair_angles(p.pos, p.ff, t, pairIndex, i0, i1, c, s, c2, s2);
        yr[i0] = x0 * c - x1 * s;
        yr[i1] = x0 * s2 + x1 * c2;
    }
    else
    {
        yr[i0] = x0;
        yr[i1] = x1;
    }
}
#endif

#ifdef KERNEL_ROPE_ROW
// ---------------------------------------------------------------------------------------------
// Row variant: one workgroup per (t, h). NORM: RmsNorm (weight w[D], EPS) of the row before rotating.
// CACHE_OUT: store into a [ctx, H, D] cache at row rt[RT_N_PAST] + t (CACHE_F16: f16 cache) instead of y.
struct RowParams
{
    float* x;
    int* pos;
    float* w;
    uint8_t* y;
    uint* rt;
    float* ff;
#if MULTI_SEQ
    int* seq;   // [T, 4] (slot, position, ...): CACHE_OUT row of t is seq_row(slot, position)
#endif
#if PAGED
    int* pages; // [slots, PAGE_MB] block table
#endif
    uint T;
};

groupshared float red[64];

void store(RowParams p, uint t, uint h, uint i, float v)
{
#if CACHE_OUT
#if MULTI_SEQ
    uint64_t di = (seq_row(SEQ_PAGES(p), uint(p.seq[t * 4]), uint(p.seq[t * 4 + 1])) * H + h) * D + i;
#else
    uint64_t di = (uint64_t(p.rt[RT_N_PAST] + t) * H + h) * D + i;
#endif
#if CACHE_F16
    *(uint16_t*)byteAt((uint8_t*)p.y, di * 2) = f2h(v);
#else
    *(float*)byteAt((uint8_t*)p.y, di * 4) = v;
#endif
#else
    ((float*)p.y)[(uint64_t(t) * H + h) * D + i] = v;
#endif
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rope_row(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform RowParams p)
{
    const uint PAIRS = D / 2;
    uint row = gid.x;
    if (row >= p.T * H) return;
    uint t = row / H, h = row % H, lt = ltid.x;
    float* xr = p.x + uint64_t(t) * S0 + uint64_t(h) * S1;

    float scale = 1.0;
#if NORM
    float ss = 0;
    for (uint i = lt; i < D; i += 64) { float v = xr[i]; ss += v * v; }
    red[lt] = ss;
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint s = 32; s > 0; s >>= 1) { if (lt < s) red[lt] += red[lt + s]; GroupMemoryBarrierWithGroupSync(); }
    scale = rsqrt(red[0] / float(D) + EPS);
#endif

    for (uint i = lt; i < PAIRS; i += 64)
    {
        uint i0, i1;
#if NEOX
        if (i < ROT / 2) { i0 = i; i1 = i + ROT / 2; }
        else { uint j = ROT + 2 * (i - ROT / 2); i0 = j; i1 = j + 1; }
#else
        i0 = 2 * i; i1 = 2 * i + 1;
#endif
        float x0 = xr[i0] * scale, x1 = xr[i1] * scale;
#if NORM
        x0 *= p.w[i0]; x1 *= p.w[i1];
#endif
        if (i < ROT / 2)
        {
            float c, s, c2, s2;
            pair_angles(p.pos, p.ff, t, i, i0, i1, c, s, c2, s2);
            store(p, t, h, i0, x0 * c - x1 * s);
            store(p, t, h, i1, x0 * s2 + x1 * c2);
        }
        else
        {
            store(p, t, h, i0, x0);
            store(p, t, h, i1, x1);
        }
    }
}
#endif

#ifdef KERNEL_ROPE_QKV
// ---------------------------------------------------------------------------------------------
// The q, k and v rows of one attention layer in one launch (one workgroup per QKV_RPW (t, head of q, k or v) rows): workgroups h < HQ rotate
// query head h of xq (strides QS0, QS1; NORM_Q: RmsNorm with wq, EPS_Q, first) into yq [T, HQ, D]; HQ <= h < HQ + HK rotate key
// head h - HQ of xk (KS0, KS1; NORM_K with wk, EPS_K) into the key cache kc; the last HK copy value head h - HQ - HK of xv (VS0, VS1)
// into the value cache vc. Both caches are [ctx, HK, D] rows of the same positions (rope_row's CACHE_OUT row; CACHE_F16: f16).
// The rope macros (D, ROT, NEOX, ...) are those of rope_row, shared by q and k.
struct QkvParams
{
    float* xq;
    float* xk;
    float* xv;
    int* pos;
    float* wq;
    float* wk;
    float* yq;
    uint8_t* kc;
    uint8_t* vc;
    uint* rt;
    float* ff;
#if MULTI_SEQ
    int* seq;
#endif
#if PAGED
    int* pages;
#endif
    uint T;
};

groupshared float qkvRed[64];

// Cache element index of (t, key head h, dimension i).
uint64_t qkv_cache_index(QkvParams p, uint t, uint h, uint i)
{
#if MULTI_SEQ
    return (seq_row(SEQ_PAGES(p), uint(p.seq[t * 4]), uint(p.seq[t * 4 + 1])) * HK + h) * D + i;
#else
    return (uint64_t(p.rt[RT_N_PAST] + t) * HK + h) * D + i;
#endif
}

void qkv_cache_store(uint8_t* cache, uint64_t di, float v)
{
#if CACHE_F16
    *(uint16_t*)byteAt(cache, di * 2) = f2h(v);
#else
    *(float*)byteAt(cache, di * 4) = v;
#endif
}

// The reciprocal RMS of row xr (all 64 threads take part).
float qkv_rms_scale(float* xr, uint lt, float eps)
{
    float ss = 0;
    for (uint i = lt; i < D; i += 64) { float v = xr[i]; ss += v * v; }
    qkvRed[lt] = ss;
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint s = 32; s > 0; s >>= 1) { if (lt < s) qkvRed[lt] += qkvRed[lt + s]; GroupMemoryBarrierWithGroupSync(); }
    float scale = rsqrt(qkvRed[0] / float(D) + eps);
    // The workgroup's next row reuses qkvRed (QKV_RPW > 1).
    GroupMemoryBarrierWithGroupSync();
    return scale;
}

// Query head hh (Q) or key head hh of row t: the norm (with weight w), the rotation, the store to yq or to the key cache. The row and
// weight pointers are arguments (not selected inside: the SPIR-V target does not take a select of two buffer pointers).
void qkv_rope_head(QkvParams p, float* xr, float* w, bool norm, float eps, bool Q, uint t, uint hh, uint lt)
{
    const uint PAIRS = D / 2;
    float scale = 1.0;
    if (norm)
    {
        scale = qkv_rms_scale(xr, lt, eps);
    }

    for (uint i = lt; i < PAIRS; i += 64)
    {
        uint i0, i1;
#if NEOX
        if (i < ROT / 2) { i0 = i; i1 = i + ROT / 2; }
        else { uint j = ROT + 2 * (i - ROT / 2); i0 = j; i1 = j + 1; }
#else
        i0 = 2 * i; i1 = 2 * i + 1;
#endif
        float x0 = xr[i0] * scale, x1 = xr[i1] * scale;
        if (norm)
        {
            x0 *= w[i0];
            x1 *= w[i1];
        }

        float y0 = x0, y1 = x1;
        if (i < ROT / 2)
        {
            float c, s, c2, s2;
            pair_angles(p.pos, p.ff, t, i, i0, i1, c, s, c2, s2);
            y0 = x0 * c - x1 * s;
            y1 = x0 * s2 + x1 * c2;
        }

        if (Q)
        {
            p.yq[(uint64_t(t) * HQ + hh) * D + i0] = y0;
            p.yq[(uint64_t(t) * HQ + hh) * D + i1] = y1;
        }
        else
        {
            qkv_cache_store(p.kc, qkv_cache_index(p, t, hh, i0), y0);
            qkv_cache_store(p.kc, qkv_cache_index(p, t, hh, i1), y1);
        }
    }
}

#ifndef QKV_WARP
#define QKV_WARP 0
#endif
#if QKV_WARP
// QKV_WARP (CUDA, 32-lane warps): one warp per (t, head) row, four rows per 128-thread workgroup at a time, no shared memory and no
// barrier. The q / k norm's reduction is the 64-thread form's tree, bit for bit: lane l holds the partials of threads l and l + 32
// (their sum is the tree's first level), and the xor butterfly computes every later level of lane 0's chain from the same operands.
// The rotation runs the same arithmetic per pair, two pairs per lane where the 64-thread form ran one.
float qkv_rms_scale_warp(float* xr, uint lane, float eps)
{
    float a = 0, b = 0;
    for (uint i = lane; i < D; i += 64) { float v = xr[i]; a += v * v; }
    for (uint i = lane + 32; i < D; i += 64) { float v = xr[i]; b += v * v; }
    float s = a + b;
    [ForceUnroll]
    for (uint m = 16; m > 0; m >>= 1) { s += WaveReadLaneAt(s, lane ^ m); }
    return rsqrt(WaveReadLaneAt(s, 0) / float(D) + eps);
}

void qkv_rope_head_warp(QkvParams p, float* xr, float* w, bool norm, float eps, bool Q, uint t, uint hh, uint lane)
{
    const uint PAIRS = D / 2;
    float scale = 1.0;
    if (norm)
    {
        scale = qkv_rms_scale_warp(xr, lane, eps);
    }

    for (uint i = lane; i < PAIRS; i += 32)
    {
        uint i0, i1;
#if NEOX
        if (i < ROT / 2) { i0 = i; i1 = i + ROT / 2; }
        else { uint j = ROT + 2 * (i - ROT / 2); i0 = j; i1 = j + 1; }
#else
        i0 = 2 * i; i1 = 2 * i + 1;
#endif
        float x0 = xr[i0] * scale, x1 = xr[i1] * scale;
        if (norm)
        {
            x0 *= w[i0];
            x1 *= w[i1];
        }

        float y0 = x0, y1 = x1;
        if (i < ROT / 2)
        {
            float c, s, c2, s2;
            pair_angles(p.pos, p.ff, t, i, i0, i1, c, s, c2, s2);
            y0 = x0 * c - x1 * s;
            y1 = x0 * s2 + x1 * c2;
        }

        if (Q)
        {
            p.yq[(uint64_t(t) * HQ + hh) * D + i0] = y0;
            p.yq[(uint64_t(t) * HQ + hh) * D + i1] = y1;
        }
        else
        {
            qkv_cache_store(p.kc, qkv_cache_index(p, t, hh, i0), y0);
            qkv_cache_store(p.kc, qkv_cache_index(p, t, hh, i1), y1);
        }
    }
}

void rope_qkv_row_warp(QkvParams p, uint row, uint lane)
{
    const uint HEADS = HQ + 2 * HK;
    uint t = row / HEADS, h = row % HEADS;
    if (h < HQ)
    {
        qkv_rope_head_warp(p, p.xq + uint64_t(t) * QS0 + uint64_t(h) * QS1, p.wq, NORM_Q != 0, EPS_Q, true, t, h, lane);
    }
    else if (h < HQ + HK)
    {
        qkv_rope_head_warp(p, p.xk + uint64_t(t) * KS0 + uint64_t(h - HQ) * KS1, p.wk, NORM_K != 0, EPS_K, false, t, h - HQ, lane);
    }
    else
    {
        uint hv = h - HQ - HK;
        float* vr = p.xv + uint64_t(t) * VS0 + uint64_t(hv) * VS1;
        for (uint i = lane; i < D; i += 32)
        {
            qkv_cache_store(p.vc, qkv_cache_index(p, t, hv, i), vr[i]);
        }
    }
}
#endif

// QKV_RPW (rows per workgroup, default 1): each workgroup runs that many consecutive (t, head) rows one after another; a launch of
// short rows (64 values, no norm) is otherwise bound by how fast workgroups start.
#ifndef QKV_RPW
#define QKV_RPW 1
#endif

void rope_qkv_row(QkvParams p, uint row, uint lt)
{
    const uint HEADS = HQ + 2 * HK;
    uint t = row / HEADS, h = row % HEADS;
    if (h < HQ)
    {
        qkv_rope_head(p, p.xq + uint64_t(t) * QS0 + uint64_t(h) * QS1, p.wq, NORM_Q != 0, EPS_Q, true, t, h, lt);
    }
    else if (h < HQ + HK)
    {
        qkv_rope_head(p, p.xk + uint64_t(t) * KS0 + uint64_t(h - HQ) * KS1, p.wk, NORM_K != 0, EPS_K, false, t, h - HQ, lt);
    }
    else
    {
        uint hv = h - HQ - HK;
        float* vr = p.xv + uint64_t(t) * VS0 + uint64_t(hv) * VS1;
        for (uint i = lt; i < D; i += 64)
        {
            qkv_cache_store(p.vc, qkv_cache_index(p, t, hv, i), vr[i]);
        }
    }
}

#if QKV_WARP
// QKV_RPW rows per warp (consecutive), four warps per workgroup.
[shader("compute")]
[numthreads(128, 1, 1)]
void rope_qkv(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform QkvParams p)
{
    const uint rows = p.T * (HQ + 2 * HK);
    uint lane = ltid.x % 32;
    uint first = (gid.x * 4 + ltid.x / 32) * QKV_RPW;
    for (uint r = 0; r < QKV_RPW; r++)
    {
        uint row = first + r;
        if (row >= rows) return;
        rope_qkv_row_warp(p, row, lane);
    }
}
#else
[shader("compute")]
[numthreads(64, 1, 1)]
void rope_qkv(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform QkvParams p)
{
    const uint rows = p.T * (HQ + 2 * HK);
    for (uint r = 0; r < QKV_RPW; r++)
    {
        uint row = gid.x * QKV_RPW + r;
        if (row >= rows) return;
        rope_qkv_row(p, row, ltid.x);
    }
}
#endif
#endif
