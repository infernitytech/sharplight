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

#ifndef KERNEL_ROPE_ROW
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
