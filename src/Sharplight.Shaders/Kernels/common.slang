#pragma once
// Sharplight common kernel helpers.
// Conventions shared by all kernels:
//  * Parameters are passed as an entry-point `uniform` struct: all pointers first, then 32-bit
//    scalars. On Vulkan this becomes a push-constant block (pointers = buffer device addresses),
//    on CUDA plain kernel parameters. No descriptor sets.
//  * Shapes and strides are compile-time constants injected as preprocessor macros; only the
//    token count T (and runtime scalars read through `rt`) are dynamic.
//  * SG = subgroup (wave/warp) size the kernel was specialized for.

#ifndef SG
#define SG 32
#endif

// Runtime scalar slot of the KV offset (n_past) in the `rt` buffer (uint array).
#ifndef RT_N_PAST
#define RT_N_PAST 0
#endif

// Multi-sequence (continuous batching) forms of the stateful kernels; SLOT_CTX = cache rows per sequence slot.
#ifndef MULTI_SEQ
#define MULTI_SEQ 0
#endif

float gelu_tanh(float x) { return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }
float erf_approx(float x)
{
    // Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)
    float s = sign(x); x = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * x);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return s * y;
}
float gelu_erf(float x) { return 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865476)); }

// ---- 16-bit float helpers ----------------------------------------------------------------
float h2f(uint16_t h) { return f16tof32(uint(h)); }
float bf2f(uint16_t b) { return asfloat(uint(b) << 16); }
uint16_t f2h(float f) { return uint16_t(f32tof16(f)); }

// ---- weight formats ------------------------------------------------------------------------
// Each format provides load8(base, row, g, out float v[8]): the 8 consecutive logical values
// [8g, 8g+8) of row `row`, where ROW_BYTES is the byte size of one row (K elements).
// Formats are selected with WFMT_* = 1 macros so different weight types specialize the same kernel.

#define FMT_F32 0
#define FMT_F16 1
#define FMT_BF16 2
#define FMT_Q8_0 8
#define FMT_Q8_0R 108
#define FMT_Q4_0 2001
#define FMT_Q4_K 12
#define FMT_Q5_K 13
#define FMT_Q6_K 14

void load8_f32(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    float4* p = (float4*)(base + rowBytes * row + uint64_t(g) * 32);
    float4 a = p[0], b = p[1];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

void load8_f16(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint4 u = *(uint4*)(base + rowBytes * row + uint64_t(g) * 16);
    v[0] = f16tof32(u.x & 0xFFFF); v[1] = f16tof32(u.x >> 16);
    v[2] = f16tof32(u.y & 0xFFFF); v[3] = f16tof32(u.y >> 16);
    v[4] = f16tof32(u.z & 0xFFFF); v[5] = f16tof32(u.z >> 16);
    v[6] = f16tof32(u.w & 0xFFFF); v[7] = f16tof32(u.w >> 16);
}

void load8_bf16(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint4 u = *(uint4*)(base + rowBytes * row + uint64_t(g) * 16);
    v[0] = asfloat(u.x << 16); v[1] = asfloat(u.x & 0xFFFF0000u);
    v[2] = asfloat(u.y << 16); v[3] = asfloat(u.y & 0xFFFF0000u);
    v[4] = asfloat(u.z << 16); v[5] = asfloat(u.z & 0xFFFF0000u);
    v[6] = asfloat(u.w << 16); v[7] = asfloat(u.w & 0xFFFF0000u);
}

// Q8_0: block of 32 = { f16 d; int8 q[32]; } (34 bytes, 2-byte aligned)
void load8_q8_0(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint16_t* blk = (uint16_t*)(base + rowBytes * row + uint64_t(g >> 2) * 34);
    float d = h2f(blk[0]);
    uint q = (g & 3) * 4 + 1; // u16 index of the first quant pair
    [ForceUnroll]
    for (int i = 0; i < 4; i++)
    {
        uint w = uint(blk[q + i]);
        v[2 * i] = d * float(int(w << 24) >> 24);
        v[2 * i + 1] = d * float(int(w << 16) >> 24);
    }
}

// Q8_0R (repacked): row = [K/32 f16 scales][K int8 quants]; rowBytes = 17K/16, scales take rowBytes/17.
void load8_q8_0r(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* r = base + rowBytes * row;
    float d = h2f(*(uint16_t*)(r + (g >> 2) * 2));
    uint2 q = *(uint2*)(r + rowBytes / 17 + uint64_t(g) * 8);
    [ForceUnroll]
    for (int i = 0; i < 4; i++)
    {
        v[i] = d * float(int(q.x << (24 - 8 * i)) >> 24);
        v[4 + i] = d * float(int(q.y << (24 - 8 * i)) >> 24);
    }
}

// Q4_0: block of 32 = { f16 d; uint8 qs[16]; } (18 bytes). value j<16: lo nibble of qs[j]; j>=16: hi nibble of qs[j-16]
void load8_q4_0(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 2) * 18;
    float d = h2f(*(uint16_t*)blk);
    uint sub = g & 3;               // which 8-group inside the 32 block
    uint off = (sub & 1) * 8;       // qs byte offset
    uint shift = (sub >> 1) * 4;    // lo or hi nibble
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = d * (float((uint(blk[2 + off + i]) >> shift) & 0xF) - 8.0);
}

// K-quant 6-bit scale/min unpacking (ggml get_scale_min_k4).
void scale_min_k4(uint8_t* q, uint j, out float sc, out float mn)
{
    if (j < 4) { sc = float(uint(q[j]) & 63); mn = float(uint(q[j + 4]) & 63); }
    else
    {
        sc = float((uint(q[j + 4]) & 0xF) | ((uint(q[j - 4]) >> 6) << 4));
        mn = float((uint(q[j + 4]) >> 4) | ((uint(q[j]) >> 6) << 4));
    }
}

// Q4_K: 256 values = { f16 d; f16 dmin; u8 scales[12]; u8 qs[128]; } (144 bytes)
void load8_q4_k(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 144;
    float d = h2f(*(uint16_t*)blk);
    float dmin = h2f(*(uint16_t*)(blk + 2));
    uint idx = (g & 31) * 8;        // value index inside the 256 block
    uint j = idx >> 6;              // 64-value chunk
    uint hi = (idx >> 5) & 1;       // second 32 of the chunk uses the high nibble
    uint l0 = idx & 31;
    float sc, mn;
    scale_min_k4(blk + 4, 2 * j + hi, sc, mn);
    float a = d * sc, b = dmin * mn;
    uint8_t* qs = blk + 16 + 32 * j + l0;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = a * float((uint(qs[i]) >> (4 * hi)) & 0xF) - b;
}

// Q5_K: 256 values = { f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]; } (176 bytes)
void load8_q5_k(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 176;
    float d = h2f(*(uint16_t*)blk);
    float dmin = h2f(*(uint16_t*)(blk + 2));
    uint idx = (g & 31) * 8;
    uint j = idx >> 6;
    uint hi = (idx >> 5) & 1;
    uint l0 = idx & 31;
    float sc, mn;
    scale_min_k4(blk + 4, 2 * j + hi, sc, mn);
    float a = d * sc, b = dmin * mn;
    uint8_t* qh = blk + 16;
    uint8_t* qs = blk + 48 + 32 * j + l0;
    uint hbit = 2 * j + hi;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
    {
        uint q = ((uint(qs[i]) >> (4 * hi)) & 0xF) | (((uint(qh[l0 + i]) >> hbit) & 1) << 4);
        v[i] = a * float(q) - b;
    }
}

// Q6_K: 256 values = { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d; } (210 bytes)
void load8_q6_k(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 210;
    float d = h2f(*(uint16_t*)(blk + 208));
    uint idx = (g & 31) * 8;
    uint n = idx >> 7;              // 128-value half
    uint r = idx & 127;
    uint qd = r >> 5;               // quarter 0..3
    uint l0 = r & 31;
    float s = d * float(int(uint(blk[192 + n * 8 + (l0 >> 4) + 2 * qd]) << 24) >> 24);
    uint8_t* ql = blk + n * 64 + (qd & 1) * 32 + l0;
    uint8_t* qh = blk + 128 + n * 32 + l0;
    uint lshift = (qd >> 1) * 4;
    uint hshift = 2 * qd;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
    {
        int q = int(((uint(ql[i]) >> lshift) & 0xF) | (((uint(qh[i]) >> hshift) & 3) << 4)) - 32;
        v[i] = s * float(q);
    }
}

#ifdef WFMT
void load8(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
#if WFMT == FMT_F32
    load8_f32(base, rowBytes, row, g, v);
#elif WFMT == FMT_F16
    load8_f16(base, rowBytes, row, g, v);
#elif WFMT == FMT_BF16
    load8_bf16(base, rowBytes, row, g, v);
#elif WFMT == FMT_Q8_0R
    load8_q8_0r(base, rowBytes, row, g, v);
#elif WFMT == FMT_Q8_0
    load8_q8_0(base, rowBytes, row, g, v);
#elif WFMT == FMT_Q4_0
    load8_q4_0(base, rowBytes, row, g, v);
#elif WFMT == FMT_Q4_K
    load8_q4_k(base, rowBytes, row, g, v);
#elif WFMT == FMT_Q5_K
    load8_q5_k(base, rowBytes, row, g, v);
#elif WFMT == FMT_Q6_K
    load8_q6_k(base, rowBytes, row, g, v);
#else
#error "unsupported WFMT"
#endif
}
#endif
