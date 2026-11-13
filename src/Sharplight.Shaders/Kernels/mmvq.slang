// Integer-dot matrix-vector products for small row counts (decode, small batches), DeviceFeatures.IntDot.
// Activations are first quantized per 32-value block to int8 (quantize_q8_1); the weight kernel then multiplies
// packed int8 quads with dot4add_i8packed (dp4a) and applies the block scales once per block, so every weight
// block is read once for all activation rows and the activation traffic is one byte per value.
//
// Quantized activation row (QROW bytes): [K int8 q][K/32 float2 (d, d * sum(q))], x ≈ d * q per 32-value block.
// Specialization macros: WFMT (q8_0r, q4_k, q5_k, q6_k), K, N, Y_STRIDE, ROW_BYTES, SG;
// MQ_COLS (activation rows handled per launch, >= T), MQ_ROWS (weight rows per workgroup), MQ_SG (subgroups per workgroup).
#include "common.slang"

#ifndef DUAL
#define DUAL 0
#endif
#define QROW (K + K / 4)

// ---------------------------------------------------------------------------------------------
// Activation quantization: one thread per 4 values, 8 threads per 32-value block (X_STRIDE: f32 row stride).
#ifdef KERNEL_QUANTIZE_Q8_1
struct QuantParams
{
    float* x;
    uint8_t* q;
    uint T;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void quantize_q8_1(uint3 dtid : SV_DispatchThreadID, uniform QuantParams p)
{
    uint i = dtid.x;
    uint t = i / (K / 4);
    uint k4 = i % (K / 4);
    bool live = t < p.T;
    float4 v = 0;
    if (live) v = *(float4*)(p.x + uint64_t(t) * X_STRIDE + k4 * 4);
    float amax = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));
    uint lane = WaveGetLaneIndex();
    [ForceUnroll]
    for (uint m = 1; m < 8; m <<= 1) amax = max(amax, WaveReadLaneAt(amax, lane ^ m));
    float d = amax / 127.0;
    float id = d > 0 ? 1.0 / d : 0.0;
    int4 q = int4(round(v * id));
    int s = q.x + q.y + q.z + q.w;
    [ForceUnroll]
    for (uint m = 1; m < 8; m <<= 1) s += WaveReadLaneAt(s, lane ^ m);
    if (!live) return;
    uint8_t* row = p.q + uint64_t(t) * QROW;
    *(uint*)(row + k4 * 4) = (uint(q.x) & 0xFF) | ((uint(q.y) & 0xFF) << 8) | ((uint(q.z) & 0xFF) << 16) | (uint(q.w) << 24);
    if ((k4 & 7) == 0) *(float2*)(row + K + (k4 / 8) * 8) = float2(d, d * float(s));
}
#endif

// ---------------------------------------------------------------------------------------------
// Weight chunks: each thread handles one chunk of a weight row per iteration (CHUNK_VALUES values). Per format:
//   WChunk load_chunk(w, row, ch)            weights of chunk ch of row `row`, unpacked to int8 quads + scales
//   XChunk load_xchunk(xrow, ch)             the matching activation quads and block scales
//   float dot_chunk(WChunk, XChunk, ch)      their contribution to the row's dot product
#if defined(KERNEL_MMVQ) || defined(KERNEL_MOE_MMVQ)
int dot4(uint a, uint b, int acc) { return dot4add_i8packed(a, b, acc); }
int dot16(uint4 a, uint4 b) { return dot4(a.w, b.w, dot4(a.z, b.z, dot4(a.y, b.y, dot4(a.x, b.x, 0)))); }

// 16 bytes from a 2-byte aligned address.
uint4 load16_a2(uint8_t* p)
{
    uint16_t* h = (uint16_t*)p;
    return uint4(uint(h[0]) | (uint(h[1]) << 16), uint(h[2]) | (uint(h[3]) << 16), uint(h[4]) | (uint(h[5]) << 16), uint(h[6]) | (uint(h[7]) << 16));
}

#if WFMT == FMT_Q8_0R
// Repacked Q8_0 row: [K/32 f16 scales][K int8]. Chunk = 16 values (one 16-byte load, coalesced across lanes).
#define CHUNK_VALUES 16
struct WChunk { uint4 q; float d; };
struct XChunk { uint4 q; float d; };
WChunk load_chunk(uint8_t* w, uint row, uint ch)
{
    uint8_t* r = rowStart(w, ROW_BYTES, row);
    WChunk c;
    c.d = h2f(*(uint16_t*)(r + (ch >> 1) * 2));
    c.q = *(uint4*)(r + ROW_BYTES / 17 + uint64_t(ch) * 16);
    return c;
}
XChunk load_xchunk(uint8_t* x, uint ch)
{
    XChunk c;
    c.q = *(uint4*)(x + ch * 16);
    c.d = *(float*)(x + K + (ch >> 1) * 8);
    return c;
}
float dot_chunk(WChunk w, XChunk x, uint ch) { return w.d * x.d * float(dot16(w.q, x.q)); }

#elif WFMT == FMT_Q4_K || WFMT == FMT_Q5_K
// Q4_K / Q5_K super-block of 256 values (144 / 176 bytes, 16-byte aligned): chunk i (0..7) holds the 16 quant bytes
// at 16i of qs, i.e. 16 values of sub-block 2j (low nibbles) and 16 of sub-block 2j+1 (high nibbles), j = i / 2.
#define CHUNK_VALUES 32
#if WFMT == FMT_Q4_K
#define KQ_BYTES 144
#define KQ_QS 16
#else
#define KQ_BYTES 176
#define KQ_QS 48
#endif
struct WChunk { uint4 lo; uint4 hi; float dlo; float dhi; float mlo; float mhi; };
struct XChunk { uint4 lo; uint4 hi; float2 dlo; float2 dhi; };

uint byte_of(uint3 s, uint i)
{
    uint w = i < 4 ? s.x : i < 8 ? s.y : s.z;
    return (w >> (8 * (i & 3))) & 0xFF;
}

// ggml get_scale_min_k4 on the 12 scale bytes held in registers.
void scale_min_regs(uint3 s, uint j, out float sc, out float mn)
{
    if (j < 4)
    {
        sc = float(byte_of(s, j) & 63);
        mn = float(byte_of(s, j + 4) & 63);
    }
    else
    {
        sc = float((byte_of(s, j + 4) & 0xF) | ((byte_of(s, j - 4) >> 6) << 4));
        mn = float((byte_of(s, j + 4) >> 4) | ((byte_of(s, j) >> 6) << 4));
    }
}

WChunk load_chunk(uint8_t* w, uint row, uint ch)
{
    uint8_t* blk = rowStart(w, ROW_BYTES, row) + uint64_t(ch >> 3) * KQ_BYTES;
    uint i = ch & 7;
    uint j = i >> 1;
    uint4 head = *(uint4*)blk;
    uint4 qs = *(uint4*)(blk + KQ_QS + 16 * i);
    float d = f16tof32(head.x & 0xFFFF);
    float dmin = f16tof32(head.x >> 16);
    float sc0, mn0, sc1, mn1;
    scale_min_regs(head.yzw, 2 * j, sc0, mn0);
    scale_min_regs(head.yzw, 2 * j + 1, sc1, mn1);
    WChunk c;
    c.lo = qs & 0x0F0F0F0Fu;
    c.hi = (qs >> 4) & 0x0F0F0F0Fu;
#if WFMT == FMT_Q5_K
    // Fifth bits: bit 2j (low nibbles) and 2j+1 (high nibbles) of qh[l], l = position inside the 32-value sub-block.
    uint4 qh = *(uint4*)(blk + 16 + 16 * (i & 1));
    c.lo |= ((qh >> (2 * j)) & 0x01010101u) << 4;
    c.hi |= ((qh >> (2 * j + 1)) & 0x01010101u) << 4;
#endif
    c.dlo = d * sc0;
    c.dhi = d * sc1;
    // The min term covers whole 32-value sub-blocks: only the first chunk of each sub-block pair adds it.
    bool first = (i & 1) == 0;
    c.mlo = first ? dmin * mn0 : 0.0;
    c.mhi = first ? dmin * mn1 : 0.0;
    return c;
}
XChunk load_xchunk(uint8_t* x, uint ch)
{
    uint i = ch & 7;
    uint base = (ch >> 3) * 256 + (i >> 1) * 64 + (i & 1) * 16;
    uint sb = (ch >> 3) * 8 + (i >> 1) * 2;
    XChunk c;
    c.lo = *(uint4*)(x + base);
    c.hi = *(uint4*)(x + base + 32);
    float4 ds = *(float4*)(x + K + sb * 8);
    c.dlo = ds.xy;
    c.dhi = ds.zw;
    return c;
}
float dot_chunk(WChunk w, XChunk x, uint ch)
{
    return w.dlo * x.dlo.x * float(dot16(w.lo, x.lo)) + w.dhi * x.dhi.x * float(dot16(w.hi, x.hi))
         - w.mlo * x.dlo.y - w.mhi * x.dhi.y;
}

#elif WFMT == FMT_Q6_K
// Q6_K super-block { ql[128]; qh[64]; int8 scales[16]; f16 d } (210 bytes, 2-byte aligned). Chunk = 64 values: half n
// (128 values) and positions l0..l0+15 of its four 32-value quarters, each quarter 16 values with one scale.
#define CHUNK_VALUES 64
struct WChunk { uint4 q0; uint4 q1; uint4 q2; uint4 q3; float4 s; };
struct XChunk { uint4 q0; uint4 q1; uint4 q2; uint4 q3; float4 d; };

// q - 32 for bytes holding 6-bit q: (q & 31) when bit 5 is set, else q | 0xE0.
uint4 minus32(uint4 q) { return (q & 0x1F1F1F1Fu) | (((~q) & 0x20202020u) * 7u); }

WChunk load_chunk(uint8_t* w, uint row, uint ch)
{
    uint8_t* blk = rowStart(w, ROW_BYTES, row) + uint64_t(ch >> 2) * 210;
    uint n = (ch >> 1) & 1;
    uint l0 = (ch & 1) * 16;
    uint4 qa = load16_a2(blk + n * 64 + l0);
    uint4 qb = load16_a2(blk + n * 64 + 32 + l0);
    uint4 qh = load16_a2(blk + 128 + n * 32 + l0);
    float d = h2f(*(uint16_t*)(blk + 208));
    WChunk c;
    c.q0 = minus32((qa & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4));
    c.q1 = minus32((qb & 0x0F0F0F0Fu) | (((qh >> 2) & 0x03030303u) << 4));
    c.q2 = minus32(((qa >> 4) & 0x0F0F0F0Fu) | (((qh >> 4) & 0x03030303u) << 4));
    c.q3 = minus32(((qb >> 4) & 0x0F0F0F0Fu) | (((qh >> 6) & 0x03030303u) << 4));
    // Scales of the four quarters: bytes 0, 2, 4, 6 from (l0 / 16), as sign-extended int8.
    uint8_t* sc = blk + 192 + n * 8 + (l0 >> 4);
    c.s = d * float4(float(int(uint(sc[0]) << 24) >> 24), float(int(uint(sc[2]) << 24) >> 24),
                     float(int(uint(sc[4]) << 24) >> 24), float(int(uint(sc[6]) << 24) >> 24));
    return c;
}
XChunk load_xchunk(uint8_t* x, uint ch)
{
    uint base = (ch >> 2) * 256 + ((ch >> 1) & 1) * 128;
    uint l0 = (ch & 1) * 16;
    XChunk c;
    c.q0 = *(uint4*)(x + base + l0);
    c.q1 = *(uint4*)(x + base + 32 + l0);
    c.q2 = *(uint4*)(x + base + 64 + l0);
    c.q3 = *(uint4*)(x + base + 96 + l0);
    float4 a = *(float4*)(x + K + (base >> 5) * 8);
    float4 b = *(float4*)(x + K + (base >> 5) * 8 + 16);
    c.d = float4(a.x, a.z, b.x, b.z);
    return c;
}
float dot_chunk(WChunk w, XChunk x, uint ch)
{
    float4 i = float4(float(dot16(w.q0, x.q0)), float(dot16(w.q1, x.q1)), float(dot16(w.q2, x.q2)), float(dot16(w.q3, x.q3)));
    return dot(w.s * x.d, i);
}
#else
#error "mmvq: unsupported WFMT"
#endif

#define CHUNKS (K / CHUNK_VALUES)
#ifndef MQ_COLS
#define MQ_COLS 8
#endif
#ifndef MQ_ROWS
#define MQ_ROWS 1
#endif
#ifndef MQ_SG
#define MQ_SG 4
#endif
#ifndef MQ_UNROLL
#define MQ_UNROLL 1
#endif
// MQ_LPR (lanes per row group) < SG: every LPR lanes own MQ_ROWS rows and split K among themselves, so each subgroup
// works on SG / LPR row groups at once and their activation loads hit the same addresses (broadcast). MQ_LPR == SG:
// whole subgroups split K, and the MQ_SG subgroups of a workgroup split it further (reduced through shared memory).
#ifndef MQ_LPR
#define MQ_LPR SG
#endif
#define MQ_THREADS (SG * MQ_SG)
#define MQ_OUT (MQ_ROWS * MQ_COLS)
#if MQ_LPR < SG
#define MQ_GROUP_ROWS (MQ_THREADS / MQ_LPR * MQ_ROWS)
#else
#define MQ_GROUP_ROWS MQ_ROWS
#endif

#if MQ_SG > 1 && MQ_LPR == SG && defined(KERNEL_MMVQ)
groupshared float mqPart[MQ_SG][MQ_OUT * (1 + DUAL)];
#endif

// Sum over the MQ_LPR lanes of a row group (lane groups are aligned within the subgroup).
float row_group_sum(float v, uint lane)
{
#if MQ_LPR < SG
    [ForceUnroll] for (uint m = 1; m < MQ_LPR; m <<= 1) v += WaveReadLaneAt(v, lane ^ m);
    return v;
#else
    return WaveActiveSum(v);
#endif
}

#ifdef KERNEL_MMVQ
struct Params
{
    uint8_t* xq;
    uint8_t* w;
#if DUAL
    uint8_t* w2;
#endif
    float* y;
    uint T;
};

[shader("compute")]
[numthreads(MQ_THREADS, 1, 1)]
void mmvq(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    uint tid = ltid.x;
    uint lane = tid % SG;
#if MQ_LPR < SG
    uint row0 = gid.x * MQ_GROUP_ROWS + (tid / MQ_LPR) * MQ_ROWS;
    uint chBegin = lane % MQ_LPR;
    const uint chStep = MQ_LPR;
#else
    uint row0 = gid.x * MQ_ROWS;
    uint chBegin = tid;
    const uint chStep = MQ_THREADS;
#endif
    float acc[MQ_OUT];
    [ForceUnroll] for (int i = 0; i < MQ_OUT; i++) acc[i] = 0;
#if DUAL
    float acc2[MQ_OUT];
    [ForceUnroll] for (int i = 0; i < MQ_OUT; i++) acc2[i] = 0;
#endif

    [unroll(MQ_UNROLL)]
    for (uint ch = chBegin; ch < CHUNKS; ch += chStep)
    {
        WChunk wc[MQ_ROWS];
#if DUAL
        WChunk wc2[MQ_ROWS];
#endif
        [ForceUnroll]
        for (int r = 0; r < MQ_ROWS; r++)
        {
            uint row = min(row0 + r, uint(N - 1));
            wc[r] = load_chunk(p.w, row, ch);
#if DUAL
            wc2[r] = load_chunk(p.w2, row, ch);
#endif
        }
        // Columns past T (bucket padding) recompute the last row: unconditional loads can all be in flight at once.
        [ForceUnroll]
        for (int c = 0; c < MQ_COLS; c++)
        {
            uint col = min(uint(c), p.T - 1);
            XChunk xc = load_xchunk(p.xq + uint64_t(col) * QROW, ch);
            [ForceUnroll]
            for (int r = 0; r < MQ_ROWS; r++)
            {
                acc[r * MQ_COLS + c] += dot_chunk(wc[r], xc, ch);
#if DUAL
                acc2[r * MQ_COLS + c] += dot_chunk(wc2[r], xc, ch);
#endif
            }
        }
    }

#if MQ_LPR < SG
    [ForceUnroll]
    for (int i = 0; i < MQ_OUT; i++)
    {
        uint r = i / MQ_COLS;
        uint c = i % MQ_COLS;
        float s = row_group_sum(acc[i], lane);
#if DUAL
        s = epi2(s, row_group_sum(acc2[i], lane));
#endif
        if (lane % MQ_LPR == 0 && c < p.T && row0 + r < N) p.y[uint64_t(c) * Y_STRIDE + row0 + r] = s;
    }
#elif MQ_SG > 1
    uint sg = tid / SG;
    [ForceUnroll]
    for (int i = 0; i < MQ_OUT; i++)
    {
        float s = WaveActiveSum(acc[i]);
        if (lane == 0) mqPart[sg][i] = s;
#if DUAL
        float s2 = WaveActiveSum(acc2[i]);
        if (lane == 0) mqPart[sg][MQ_OUT + i] = s2;
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint i = tid; i < MQ_OUT; i += MQ_THREADS)
    {
        uint r = i / MQ_COLS;
        uint c = i % MQ_COLS;
        if (c >= p.T || row0 + r >= N) continue;
        float s = 0;
        [ForceUnroll] for (int g = 0; g < MQ_SG; g++) s += mqPart[g][i];
#if DUAL
        float s2 = 0;
        [ForceUnroll] for (int g = 0; g < MQ_SG; g++) s2 += mqPart[g][MQ_OUT + i];
        s = epi2(s, s2);
#endif
        p.y[uint64_t(c) * Y_STRIDE + row0 + r] = s;
    }
#else
    [ForceUnroll]
    for (int i = 0; i < MQ_OUT; i++)
    {
        uint r = i / MQ_COLS;
        uint c = i % MQ_COLS;
        float s = WaveActiveSum(acc[i]);
#if DUAL
        s = epi2(s, WaveActiveSum(acc2[i]));
#endif
        if (lane == 0 && c < p.T && row0 + r < N) p.y[uint64_t(c) * Y_STRIDE + row0 + r] = s;
    }
#endif
}
#endif

#ifdef KERNEL_MOE_MMVQ
// Expert-indexed int-dot matvec (MatMulId with few rows): y[item, n] = W[ids[item], n, :] · x[row(item)], item = t * S + s,
// row(item) = PER_SLOT ? item : t. Grid: (N / MOE_GROUP_ROWS, items). Macros as above plus E, S, PER_SLOT, EXPERT_BYTES;
// MQ_LPR lanes per row group, no split across subgroups.
// Every MQ_LPR lanes own MQ_ROWS rows of one item: MQ_THREADS / MQ_LPR row groups per workgroup.
#define MOE_GROUP_ROWS (MQ_THREADS / MQ_LPR * MQ_ROWS)
struct MoeParams
{
    uint8_t* xq;
    uint8_t* w;
    uint* ids;
    float* y;
    uint T;
};

[shader("compute")]
[numthreads(MQ_THREADS, 1, 1)]
void moe_mmvq(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform MoeParams p)
{
    uint tid = ltid.x;
    uint lane = tid % SG;
    uint item = gid.y;
    uint row0 = gid.x * MOE_GROUP_ROWS + (tid / MQ_LPR) * MQ_ROWS;
    uint e = p.ids[item];
    if (e >= E)
    {
        [ForceUnroll]
        for (int r = 0; r < MQ_ROWS; r++)
        {
            if (lane % MQ_LPR == 0 && row0 + r < N) p.y[uint64_t(item) * N + row0 + r] = 0.0;
        }
        return;
    }
    uint8_t* wb = rowStart(p.w, EXPERT_BYTES, e);
    uint8_t* xr = p.xq + uint64_t(PER_SLOT ? item : item / S) * QROW;
    float acc[MQ_ROWS];
    [ForceUnroll] for (int r = 0; r < MQ_ROWS; r++) acc[r] = 0;
    [unroll(MQ_UNROLL)]
    for (uint ch = lane % MQ_LPR; ch < CHUNKS; ch += MQ_LPR)
    {
        XChunk xc = load_xchunk(xr, ch);
        [ForceUnroll]
        for (int r = 0; r < MQ_ROWS; r++)
        {
            acc[r] += dot_chunk(load_chunk(wb, min(row0 + r, uint(N - 1)), ch), xc, ch);
        }
    }
    [ForceUnroll]
    for (int r = 0; r < MQ_ROWS; r++)
    {
        float s = row_group_sum(acc[r], lane);
        if (lane % MQ_LPR == 0 && row0 + r < N) p.y[uint64_t(item) * N + row0 + r] = s;
    }
}
#endif
#endif
