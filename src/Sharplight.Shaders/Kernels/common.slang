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

// CUDA C++ (SL_TARGET_CUDA): wave operations run over the lanes active when they execute (__activemask(), the semantics of
// subgroup operations on SPIR-V and MSL). Without this Slang's CUDA backend tracks the active mask through every branch and
// loop iteration of a kernel that has a wave operation (a __ballot_sync per branch), which made the decode attention 3.5x slower
// than its SPIR-V form. The kernels call wave operations in subgroup-uniform control flow or in branches whose lanes are
// meant to participate, as they must for SPIR-V.
#if SL_TARGET_CUDA
#define WaveReadLaneAt(v, l) WaveMaskReadLaneAt(WaveGetConvergedMask(), v, l)
#define WaveActiveSum(v) WaveMaskSum(WaveGetConvergedMask(), v)
#define WaveActiveMax(v) WaveMaskMax(WaveGetConvergedMask(), v)
#define WaveActiveMin(v) WaveMaskMin(WaveGetConvergedMask(), v)
#define WaveActiveProduct(v) WaveMaskProduct(WaveGetConvergedMask(), v)
#define WaveActiveAllTrue(c) WaveMaskAllTrue(WaveGetConvergedMask(), c)
#define WaveActiveAnyTrue(c) WaveMaskAnyTrue(WaveGetConvergedMask(), c)
#define WaveIsFirstLane() WaveMaskIsFirstLane(WaveGetConvergedMask())
#define WavePrefixSum(v) WaveMaskPrefixSum(WaveGetConvergedMask(), v)
#endif

// Runtime scalar slot of the KV offset (n_past) in the `rt` buffer (uint array).
#ifndef RT_N_PAST
#define RT_N_PAST 0
#endif

// Multi-sequence (continuous batching) forms of the stateful kernels; SLOT_CTX = cache rows per sequence slot.
#ifndef MULTI_SEQ
#define MULTI_SEQ 0
#endif
#ifndef SLOT_CTX
#define SLOT_CTX 0
#endif

// Paged KV caches (multi-sequence): the cache is a pool of PAGE_BS-row blocks and `pages` [slots, PAGE_MB] maps block
// pos / PAGE_BS of a slot to its physical block; otherwise slot s owns rows [s * SLOT_CTX, (s + 1) * SLOT_CTX).
#ifndef PAGED
#define PAGED 0
#endif

// The block table of a parameter struct (member `pages`, present only when PAGED).
#if PAGED
#define SEQ_PAGES(p) ((p).pages)
#else
#define SEQ_PAGES(p) ((int*)0)
#endif

// Cache row of position `pos` of sequence slot `slot`.
uint64_t seq_row(int* pages, uint slot, uint pos)
{
#if PAGED
    // Clamped so that positions past a sequence's end (tile tails whose loads a select discards) still address the cache.
    int block = pages[slot * PAGE_MB + min(pos / PAGE_BS, uint(PAGE_MB - 1))];
    return uint64_t(uint(max(block, 0))) * PAGE_BS + pos % PAGE_BS;
#else
    return uint64_t(slot) * SLOT_CTX + pos;
#endif
}

// Compressed caches (KV_COMP positions per row, see StateDefinition.Compression): row of entry `e` (positions [e * KV_COMP,
// (e + 1) * KV_COMP)) of slot `slot`, with COMP_SLOT_CTX rows per slot, or through the block table with ceil(PAGE_BS / KV_COMP) rows per
// block. Position `pos` closes the window of entry (pos + 1) / KV_COMP - 1 when (pos + 1) % KV_COMP == 0.
#ifndef KV_COMP
#define KV_COMP 1
#endif
#ifndef COMP_SLOT_CTX
#define COMP_SLOT_CTX 0
#endif
uint64_t seq_row_c(int* pages, uint slot, uint e)
{
#if PAGED
    uint first = e * KV_COMP;
    int block = pages[slot * PAGE_MB + min(first / PAGE_BS, uint(PAGE_MB - 1))];
    return uint64_t(uint(max(block, 0))) * ((PAGE_BS + KV_COMP - 1) / KV_COMP) + (first % PAGE_BS) / KV_COMP;
#else
    return uint64_t(slot) * COMP_SLOT_CTX + e;
#endif
}

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

// Address `bytes` past `base`, computed on integers: NVIDIA's driver truncates a 64-bit OpPtrAccessChain index to 32 bits, so
// element or byte offsets into a tensor of 4 GiB or more (large KV caches, embedding tables) must not go through pointer indexing.
uint8_t* byteAt(uint8_t* base, uint64_t bytes) { return (uint8_t*)(uint64_t(base) + bytes); }

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
#define FMT_Q4_1 3
#define FMT_Q5_0 6
#define FMT_Q5_1 7
#define FMT_IQ4_NL 20
#define FMT_IQ4_XS 23
#define FMT_Q1_0 41
#define FMT_Q2_0 42
#define FMT_Q2_0_G128 142
#define FMT_TERN1_G128 143
// Trellis-coded formats (TrellisBlocks.cs), decoded through the codebook grid.
#define FMT_TRELLIS_V4K8 144
#define FMT_TRELLIS_V4K6 145
#define FMT_TRELLIS_V4K7 146
#define FMT_TRELLIS_V2K5 147
#define FMT_TRELLIS_V2K6 148
#define FMT_TRELLIS_V2K7 149
// ggml type ids of the formats decoded in quant_k.slang / quant_iq.slang.
#define FMT_Q2_K 10
#define FMT_Q3_K 11
#define FMT_IQ2_XXS 16
#define FMT_IQ2_XS 17
#define FMT_IQ3_XXS 18
#define FMT_IQ1_S 19
#define FMT_IQ3_S 21
#define FMT_IQ2_S 22
#define FMT_IQ1_M 29
#define FMT_TQ1_0 34
#define FMT_TQ2_0 35
#define FMT_MXFP4 39
#define FMT_MXFP4R 139
#define FMT_NVFP4 40
// FP8 formats decoded in quant_fp8.slang.
#define FMT_F8E4M3 50
#define FMT_F8E5M2 51
#define FMT_F8E4M3_B32 52
#define FMT_F8E4M3_B128 53
// The same in the aligned split layout [K codes][K / B f32 scales] (WeightRepack.Fp8B32Split / Fp8B128Split).
#define FMT_F8E4M3_B32R 152
#define FMT_F8E4M3_B128R 153
// Group-scaled formats of quantized Hugging Face checkpoints, decoded in quant_group.slang.
#define FMT_NVFP4_GS 60
#define FMT_Q4Z_G32 61
#define FMT_Q4Z_G128 62
#define FMT_Q8Z_G32 63
#define FMT_Q8Z_G128 64
#define FMT_NF4_B64 65
#define FMT_FP4_B64 66
#define FMT_Q8A_G32 67
#define FMT_Q2S_G128 68

// IQ formats (and NF4, its 16 f32 code values) decode through a lookup grid that lives in a device buffer (GridTables.cs). The lowering prepends its address to the parameter block
// of every kernel that reads such a weight: those kernels declare `uint* grid` first under HAS_GRID and call GRID_BIND(p) at the start of each entry
// point, which stores it in g_grid for the decoders (quant_iq.slang).
#if WFMT >= FMT_TRELLIS_V4K8 && WFMT <= FMT_TRELLIS_V2K7
#define WFMT_TRELLIS 1
#else
#define WFMT_TRELLIS 0
#endif
#if WFMT == FMT_IQ2_XXS || WFMT == FMT_IQ2_XS || WFMT == FMT_IQ2_S || WFMT == FMT_IQ3_XXS || WFMT == FMT_IQ3_S || WFMT == FMT_IQ1_S || WFMT == FMT_IQ1_M || WFMT_TRELLIS || WFMT == FMT_NF4_B64 || WFMT == FMT_FP4_B64
#define HAS_GRID 1
static uint* g_grid;
#define GRID_BIND(p) g_grid = (p).grid;
#else
#define HAS_GRID 0
#define GRID_BIND(p)
#endif

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
    // The 8 code bytes as four 16-bit loads (2-byte aligned in the 18-byte blocks) instead of eight byte loads.
    uint16_t* q = (uint16_t*)(blk + 2 + off);
    uint w0 = uint(q[0]) | (uint(q[1]) << 16);
    uint w1 = uint(q[2]) | (uint(q[3]) << 16);
    [ForceUnroll]
    for (int i = 0; i < 4; i++)
    {
        v[i] = d * (float((w0 >> (8 * i + shift)) & 0xF) - 8.0);
        v[4 + i] = d * (float((w1 >> (8 * i + shift)) & 0xF) - 8.0);
    }
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

// scale_min_k4 on the 12 scale bytes held in three words (byte n in bits 8 (n % 4) of word n / 4).
// Byte selects instead of an indexed array, which would live in local memory.
uint kq_byte(uint3 words, uint n)
{
    uint w = n < 4 ? words.x : n < 8 ? words.y : words.z;
    return (w >> (8 * (n & 3))) & 0xFFu;
}
void scale_min_k4_words(uint3 words, uint j, out float sc, out float mn)
{
    if (j < 4) { sc = float(kq_byte(words, j) & 63); mn = float(kq_byte(words, j + 4) & 63); }
    else
    {
        sc = float((kq_byte(words, j + 4) & 0xF) | ((kq_byte(words, j - 4) >> 6) << 4));
        mn = float((kq_byte(words, j + 4) >> 4) | ((kq_byte(words, j) >> 6) << 4));
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

// Q4_1: block of 32 = { f16 d; f16 m; u8 qs[16] } (20 bytes): x = d * nibble + m (j<16 lo nibble of qs[j], j>=16 hi nibble of qs[j-16]).
void load8_q4_1(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 2) * 20;
    float d = h2f(*(uint16_t*)blk), m = h2f(*(uint16_t*)(blk + 2));
    uint sub = g & 3, off = (sub & 1) * 8, shift = (sub >> 1) * 4;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = d * float((uint(blk[4 + off + i]) >> shift) & 0xF) + m;
}

// Q5_0: block of 32 = { f16 d; u8 qh[4]; u8 qs[16] } (22 bytes): x = d * ((nibble | bit 4 from qh) - 16); Q5_1 adds an offset m instead: { f16 d; f16 m; u8 qh[4]; u8 qs[16] } (24 bytes), x = d * q5 + m.
void load8_q5(uint8_t* base, uint64_t rowBytes, uint row, uint g, bool withMin, out float v[8])
{
    uint bytes = withMin ? 24 : 22;
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 2) * bytes;
    float d = h2f(*(uint16_t*)blk);
    float m = withMin ? h2f(*(uint16_t*)(blk + 2)) : 0.0;
    uint8_t* hp = blk + (withMin ? 4 : 2);
    uint qh = uint(hp[0]) | (uint(hp[1]) << 8) | (uint(hp[2]) << 16) | (uint(hp[3]) << 24);
    uint sub = g & 3, off = (sub & 1) * 8, shift = (sub >> 1) * 4;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
    {
        uint j = sub * 8 + i;                       // element index inside the block
        uint q = ((uint(hp[4 + off + i]) >> shift) & 0xF) | (((qh >> j) & 1u) << 4);
        v[i] = withMin ? d * float(q) + m : d * (float(q) - 16.0);
    }
}

// IQ4_NL / IQ4_XS: the nibble indexes the fixed codebook kvalues = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}.
float iq4_value(uint nib)
{
    const int kv[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
    return float(kv[nib]);
}

// IQ4_NL: block of 32 = { f16 d; u8 qs[16] } (18 bytes); value j<16: lo nibble of qs[j]; j>=16: hi nibble of qs[j-16].
void load8_iq4_nl(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 2) * 18;
    float d = h2f(*(uint16_t*)blk);
    uint sub = g & 3;
    uint off = (sub & 1) * 8;
    uint shift = (sub >> 1) * 4;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = d * iq4_value((uint(blk[2 + off + i]) >> shift) & 0xF);
}

// IQ4_XS: 256 values = { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128] } (136 bytes); eight 32-value sub-blocks like IQ4_NL with a 6-bit scale (ls - 32).
void load8_iq4_xs(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 136;
    float d = h2f(*(uint16_t*)blk);
    uint sh = uint(blk[2]) | (uint(blk[3]) << 8);
    uint ib = (g & 31) >> 2;        // 32-value sub-block
    uint sub = g & 3;
    uint ls = ((uint(blk[4 + (ib >> 1)]) >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    float dl = d * float(int(ls) - 32);
    uint off = (sub & 1) * 8;
    uint shift = (sub >> 1) * 4;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = dl * iq4_value((uint(blk[8 + 16 * ib + off + i]) >> shift) & 0xF);
}

// Q1_0: 128 values = { f16 d; u8 bits[16] } (18 bytes); bit j (LSB first) set = +d, clear = -d.
void load8_q1_0(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 4) * 18;
    float d = h2f(*(uint16_t*)blk);
    uint b = uint(blk[2 + (g & 15)]);
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = ((b >> i) & 1) != 0 ? d : -d;
}

// Two-bit codes (Q2_0 group 64 and Q2_0_G128): { f16 d; u8 qs[GROUP/4] }; value j = ((qs[j/4] >> 2(j%4)) & 3) - 1, times d.
void load8_q2(uint8_t* base, uint64_t rowBytes, uint row, uint blockIdx, uint sub, uint blockBytes, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(blockIdx) * blockBytes;
    float d = h2f(*(uint16_t*)blk);
    uint w = uint(blk[2 + 2 * sub]) | (uint(blk[3 + 2 * sub]) << 8);
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = d * float(int((w >> (2 * i)) & 3) - 1);
}

// Dense ternary, group 128: { u8 qs[24]; u8 qh[2]; f16 d } (28 bytes), see DataType.Tern1_G128.
float tern_digit(uint b, uint n)
{
    uint pw = 1;
    for (uint i = 0; i < n; i++) pw *= 3;
    return float(int((((b * pw) & 255u) * 3u) >> 8) - 1);
}

void load8_tern1(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 4) * 28;
    float d = h2f(*(uint16_t*)(blk + 26));
    uint sub = g & 15;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
    {
        uint n, bi;
        if (sub < 10) { n = sub >> 1; bi = (sub & 1) * 8 + i; }
        else if (sub < 15) { n = sub - 10; bi = 16 + i; }
        else { n = uint(i) >> 1; bi = 24 + (uint(i) & 1); }
        v[i] = d * tern_digit(uint(blk[bi]), n);
    }
}

#if WFMT_TRELLIS
// Trellis-coded group-128 blocks { f16 d; u8 qs[steps * k / 8] } (TrellisBlocks.cs): the 16-bit state of step t, from the byte path
// (k = 8: qs[t - 1] << 8 | qs[t]) or the circular bit stream (16 bits from bit (steps - 1 - t) * k, LSB first).
uint trellis_state(uint8_t* qs, uint steps, uint k, uint t)
{
    if (k == 8)
    {
        return (uint(qs[(t + steps - 1) % steps]) << 8) | uint(qs[t]);
    }
    uint nbytes = steps * k / 8;
    uint off = (steps - 1 - t) * k;
    uint i0 = off >> 3;
    uint i1 = i0 + 1 >= nbytes ? i0 + 1 - nbytes : i0 + 1;
    uint i2 = i0 + 2 >= nbytes ? i0 + 2 - nbytes : i0 + 2;
    uint w = uint(qs[i0]) | (uint(qs[i1]) << 8) | (uint(qs[i2]) << 16);
    return (w >> (off & 7)) & 0xFFFFu;
}

// The codebook grid holds point p as one word (f16 value 0 | f16 value 1 << 16).
void trellis_point(uint p, float d, out float a, out float b)
{
    uint w = g_grid[p];
    a = d * f16tof32(w & 0xFFFFu);
    b = d * f16tof32(w >> 16);
}

// Four values per step (32 steps of k bits): group g covers steps 2 (g % 16) and 2 (g % 16) + 1 of block g / 16.
void load8_trellis4(uint8_t* base, uint64_t rowBytes, uint row, uint g, uint k, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 4) * (2 + 4 * k);
    float d = h2f(*(uint16_t*)blk);
    [ForceUnroll]
    for (uint i = 0; i < 2; i++)
    {
        uint x = (trellis_state(blk + 2, 32, k, (g & 15) * 2 + i) & 0x7FFFu) * 0x9E3779B1u;
        trellis_point(x >> 21, d, v[4 * i], v[4 * i + 1]);
        trellis_point((x >> 10) & 2047u, d, v[4 * i + 2], v[4 * i + 3]);
    }
}

// Two values per step (64 steps of k bits): group g covers steps 4 (g % 16) .. + 3 of block g / 16.
void load8_trellis2(uint8_t* base, uint64_t rowBytes, uint row, uint g, uint k, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 4) * (2 + 8 * k);
    float d = h2f(*(uint16_t*)blk);
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
    {
        uint x = (trellis_state(blk + 2, 64, k, (g & 15) * 4 + i) & 0x7FFFu) * 0x9E3779B1u;
        trellis_point(x >> 21, d, v[2 * i], v[2 * i + 1]);
    }
}
#endif

// 16 bytes at any byte address: 4 or 5 aligned words, funnel-shifted (the fifth word is read only when the address is not
// word aligned, as the bytes may end the buffer).
uint4 load16_any(uint64_t at)
{
    uint* wp = (uint*)(at & ~uint64_t(3));
    uint sh = uint(at & 3) * 8;
    uint4 q = uint4(wp[0], wp[1], wp[2], wp[3]);
    if (sh != 0)
    {
        q = (q >> sh) | (uint4(wp[1], wp[2], wp[3], wp[4]) << (32 - sh));
    }
    return q;
}

// MXFP4R (WeightRepack.Mxfp4Split): rows in groups of 16, each group [16 rows x K/2 code bytes][K/32 blocks x 16 rows E8M0
// scales], rowBytes = 17 K / 32 as in MXFP4. Block b of row `row` has its 16 code bytes (16-byte aligned) at
// mxfp4r_codes(..) + 16 b and its scale at mxfp4r_scale(.., b): the scales of one block of a row group are 16 adjacent bytes,
// so the rows of a tile read their scales from a few sectors. `table` is the start of the matrix (a multiple of 16 rows).
uint64_t mxfp4r_codes(uint8_t* table, uint64_t rowBytes, uint row)
{
    return uint64_t(table) + uint64_t(row >> 4) * 16 * rowBytes + uint64_t(row & 15) * (rowBytes / 17 * 16);
}

uint mxfp4r_scale(uint8_t* table, uint64_t rowBytes, uint row, uint block)
{
    return uint(*(uint8_t*)(uint64_t(table) + uint64_t(row >> 4) * 16 * rowBytes + 256 * (rowBytes / 17) + uint64_t(block) * 16 + (row & 15)));
}

// Start of row `row`. The address is computed on integers: NVIDIA's driver truncates a 64-bit OpPtrAccessChain index on a byte
// pointer to 32 bits, which breaks rows that start 4 GiB or more into a table.
uint8_t* rowStart(uint8_t* base, uint64_t rowBytes, uint row)
{
    return (uint8_t*)(uint64_t(base) + rowBytes * row);
}

#if defined(WFMT) && (WFMT == FMT_Q4_K || WFMT == FMT_Q5_K || WFMT == FMT_Q6_K || WFMT == FMT_Q4_0 || WFMT == FMT_Q5_0 || WFMT == FMT_Q4Z_G32 || WFMT == FMT_Q4Z_G128)
// Packed-half decoding of quantized codes for the tensor-core GEMMs (needs Float16).

// Bytes 0,1 (hi false) or 2,3 of u (codes 0..255) as half2 (code - offset) * scale + add.
uint codes_h2(uint u, bool hi, half offset, half2 scale, half2 add)
{
    uint v = hi ? (((u >> 16) & 0xFFu) | ((u >> 8) & 0xFF0000u)) : ((u & 0xFFu) | ((u << 8) & 0xFF0000u));
    return bit_cast<uint>((bit_cast<half2>(v | 0x64006400u) - half2(offset, offset)) * scale + add);
}

// 16 bytes from a 2-byte aligned address: four word loads when it is 4-byte aligned, else five word loads from the word
// boundary below it (the 2 bytes before and after the range must be readable, e.g. inside the same block) joined by
// shifts, instead of eight halfword loads. The address is computed on integers (pointer minus integer miscompiles).
uint4 load16_u16(uint8_t* p)
{
    uint64_t a = uint64_t(p);
    uint* w = (uint*)(a & ~3ull);
    if ((a & 2) == 0)
    {
        return uint4(w[0], w[1], w[2], w[3]);
    }
    uint w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4];
    return uint4((w0 >> 16) | (w1 << 16), (w1 >> 16) | (w2 << 16), (w2 >> 16) | (w3 << 16), (w3 >> 16) | (w4 << 16));
}
#endif

#if defined(WFMT) && (WFMT == FMT_Q4_K || WFMT == FMT_Q5_K || WFMT == FMT_Q6_K)

// 8 consecutive weights of row `row` from K offset 8g as packed halves: one 8-byte code load (Q6_K: u16 loads, its
// 210-byte blocks are only 2-byte aligned) and half2 math, as load16h in matmul.slang. Offsets are summed before
// they are added to the pointer.
uint4 load8h_kquant(uint8_t* w, uint64_t rowBytes, uint row, uint g)
{
    uint idx = (g & 31) * 8;
#if WFMT == FMT_Q6_K
    uint8_t* blk = rowStart(w, rowBytes, row) + uint64_t(g >> 5) * 210;
    uint n = idx >> 7, qd = (idx & 127) >> 5, l0 = idx & 31;
    uint16_t* pl = (uint16_t*)(blk + (n * 64 + (qd & 1) * 32 + l0));
    uint16_t* ph = (uint16_t*)(blk + (128 + n * 32 + l0));
    uint2 ql = uint2(uint(pl[0]) | (uint(pl[1]) << 16), uint(pl[2]) | (uint(pl[3]) << 16));
    uint2 qh = uint2(uint(ph[0]) | (uint(ph[1]) << 16), uint(ph[2]) | (uint(ph[3]) << 16));
    uint2 q = ((ql >> ((qd >> 1) * 4)) & 0x0F0F0F0Fu) | (((qh >> (2 * qd)) & 0x03030303u) << 4);
    float d = h2f(*(uint16_t*)(blk + 208));
    half2 a = half2(half(d * float(int(uint(blk[192 + n * 8 + (l0 >> 4) + 2 * qd]) << 24) >> 24)));
    half2 add = half2(0.0h, 0.0h);
    half offset = 1056.0h;
#else
#if WFMT == FMT_Q4_K
    uint8_t* blk = rowStart(w, rowBytes, row) + uint64_t(g >> 5) * 144;
    uint qsOffset = 16;
#else
    uint8_t* blk = rowStart(w, rowBytes, row) + uint64_t(g >> 5) * 176;
    uint qsOffset = 48;
#endif
    uint j = idx >> 6, upper = (idx >> 5) & 1, l0 = idx & 31;
    uint2 q = (*(uint2*)(blk + (qsOffset + 32 * j + l0)) >> (4 * upper)) & 0x0F0F0F0Fu;
#if WFMT == FMT_Q5_K
    q |= ((*(uint2*)(blk + (16 + l0)) >> (2 * j + upper)) & 0x01010101u) << 4;
#endif
    float sc, mn;
    scale_min_k4(blk + 4, 2 * j + upper, sc, mn);
    uint dm = *(uint*)blk;
    half2 a = half2(half(f16tof32(dm & 0xFFFF) * sc));
    half2 add = half2(half(-f16tof32(dm >> 16) * mn));
    half offset = 1024.0h;
#endif
    return uint4(codes_h2(q.x, false, offset, a, add), codes_h2(q.x, true, offset, a, add),
                 codes_h2(q.y, false, offset, a, add), codes_h2(q.y, true, offset, a, add));
}
#endif

#include "e4m3.slang"

#ifdef WFMT
#include "quant_k.slang"
#include "quant_iq.slang"
#include "quant_fp8.slang"
#include "quant_group.slang"

void load8(uint8_t* table, uint64_t rowBytes, uint tableRow, uint g, out float v[8])
{
    // Every decoder then addresses within one row only.
    uint8_t* base = rowStart(table, rowBytes, tableRow);
    const uint row = 0;
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
#elif WFMT == FMT_Q4_1
    load8_q4_1(base, rowBytes, row, g, v);
#elif WFMT == FMT_Q5_0
    load8_q5(base, rowBytes, row, g, false, v);
#elif WFMT == FMT_Q5_1
    load8_q5(base, rowBytes, row, g, true, v);
#elif WFMT == FMT_IQ4_NL
    load8_iq4_nl(base, rowBytes, row, g, v);
#elif WFMT == FMT_IQ4_XS
    load8_iq4_xs(base, rowBytes, row, g, v);
#elif WFMT == FMT_Q1_0
    load8_q1_0(base, rowBytes, row, g, v);
#elif WFMT == FMT_Q2_0
    load8_q2(base, rowBytes, row, g >> 3, g & 7, 18, v);
#elif WFMT == FMT_Q2_0_G128
    load8_q2(base, rowBytes, row, g >> 4, g & 15, 34, v);
#elif WFMT == FMT_TERN1_G128
    load8_tern1(base, rowBytes, row, g, v);
#elif WFMT == FMT_TRELLIS_V4K8
    load8_trellis4(base, rowBytes, row, g, 8, v);
#elif WFMT == FMT_TRELLIS_V4K6
    load8_trellis4(base, rowBytes, row, g, 6, v);
#elif WFMT == FMT_TRELLIS_V4K7
    load8_trellis4(base, rowBytes, row, g, 7, v);
#elif WFMT == FMT_TRELLIS_V2K5
    load8_trellis2(base, rowBytes, row, g, 5, v);
#elif WFMT == FMT_TRELLIS_V2K6
    load8_trellis2(base, rowBytes, row, g, 6, v);
#elif WFMT == FMT_TRELLIS_V2K7
    load8_trellis2(base, rowBytes, row, g, 7, v);
#elif WFMT == FMT_Q2_K
    load8_q2_k(base, rowBytes, row, g, v);
#elif WFMT == FMT_Q3_K
    load8_q3_k(base, rowBytes, row, g, v);
#elif WFMT == FMT_IQ2_XXS
    load8_iq2_xxs(base, rowBytes, row, g, v);
#elif WFMT == FMT_IQ2_XS
    load8_iq2_xs(base, rowBytes, row, g, v);
#elif WFMT == FMT_IQ2_S
    load8_iq2_s(base, rowBytes, row, g, v);
#elif WFMT == FMT_IQ3_XXS
    load8_iq3_xxs(base, rowBytes, row, g, v);
#elif WFMT == FMT_IQ3_S
    load8_iq3_s(base, rowBytes, row, g, v);
#elif WFMT == FMT_IQ1_S
    load8_iq1_s(base, rowBytes, row, g, v);
#elif WFMT == FMT_IQ1_M
    load8_iq1_m(base, rowBytes, row, g, v);
#elif WFMT == FMT_TQ1_0
    load8_tq1_0(base, rowBytes, row, g, v);
#elif WFMT == FMT_TQ2_0
    load8_tq2_0(base, rowBytes, row, g, v);
#elif WFMT == FMT_MXFP4
    load8_mxfp4(base, rowBytes, row, g, v);
#elif WFMT == FMT_MXFP4R
    // Group g: codes 8 (g & 1) .. + 8 of block g / 4, low nibbles for g & 2 == 0, else high.
    uint2 q = *(uint2*)(mxfp4r_codes(table, rowBytes, tableRow) + uint64_t(g >> 2) * 16 + (g & 1) * 8) >> ((g & 2) * 2);
    float d = e8m0_half(mxfp4r_scale(table, rowBytes, tableRow, g >> 2));
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = fp4_value(((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0xF) * d;
#elif WFMT == FMT_NVFP4
    load8_nvfp4(base, rowBytes, row, g, v);
#elif WFMT == FMT_NVFP4_GS
    load8_nvfp4_gs(base, rowBytes, row, g, v);
#elif WFMT == FMT_Q4Z_G32
    load8_q4z(base, rowBytes, row, g, 1, v);
#elif WFMT == FMT_Q4Z_G128
    load8_q4z(base, rowBytes, row, g, 4, v);
#elif WFMT == FMT_Q8Z_G32
    load8_q8z(base, rowBytes, row, g, 32, v);
#elif WFMT == FMT_Q8Z_G128
    load8_q8z(base, rowBytes, row, g, 128, v);
#elif WFMT == FMT_NF4_B64 || WFMT == FMT_FP4_B64
    load8_nf4_b64(base, rowBytes, row, g, v);
#elif WFMT == FMT_Q8A_G32
    load8_q8a(base, rowBytes, row, g, v);
#elif WFMT == FMT_Q2S_G128
    load8_q2s(base, rowBytes, row, g, v);
#elif WFMT == FMT_F8E4M3
    load8_fp8(base, rowBytes, row, g, 0, false, v);
#elif WFMT == FMT_F8E5M2
    load8_fp8(base, rowBytes, row, g, 0, true, v);
#elif WFMT == FMT_F8E4M3_B32 || WFMT == FMT_F8E4M3_B32R
    load8_fp8(base, rowBytes, row, g, 32, false, v);
#elif WFMT == FMT_F8E4M3_B128 || WFMT == FMT_F8E4M3_B128R
    load8_fp8(base, rowBytes, row, g, 128, false, v);
#else
#error "unsupported WFMT"
#endif
}
#endif

// ---------------------------------------------------------------------------------------------
// Split f16 form of f32 values (the split-f16 tensor-core GEMM): a value v becomes an f16 hi part (v rounded to 11
// significant bits on the bits, which compilers cannot fold away) and an f16 lo part (the rest), so hi + lo keeps about
// 22 bits when both are normal f16 numbers. split_scale gives the power of two that brings a row whose largest magnitude
// is m into [2^14, 2^15) (both parts in f16 range for every value of the row); 1 / split_scale(m) undoes it exactly.
float split_scale(float m)
{
    int e = int((asuint(m) >> 23) & 0xFFu);
    return asfloat(uint(clamp(268 - e, 1, 253)) << 23);
}

// split_scale for the split-f16 flash attention kernels: 1 while the largest magnitude m is in [2^-6, 2^15), else split_scale(m).
// Inside, the unscaled f16 parts hold the values (at most the lo parts of the smallest ones lose bits to f16's subnormals) and the
// kernels round exactly as their outputs were validated: SD3.5 Medium's 1024 px image follows its text encoders' attention so
// closely that rescaling every in-range row moved it from 38.4 to 27 dB against diffusers (docs/benchmarks.md section 42).
// Outside, the hi parts would overflow (from 2^15) or the largest values' parts lose their low bits (below 2^-6).
float split_scale_outside(float m)
{
    return m >= 0.015625 && m < 32768.0 ? 1.0 : split_scale(m);
}

// Range guard of the f16 activation copies the tensor-core GEMMs read ("row-ranged" copies): f16 ends at 65504, while
// activations that no norm bounds (a gated FFN's product feeding the down projection, ...) can exceed it, e.g. Qwen3-TTS's code
// predictor at 1.5e5. A row whose largest magnitude m reaches 2^15 is stored multiplied by split_scale(m) (into [2^14, 2^15)),
// every other row as is (bit-identical to the plain copy). Each row has a scale word after the rows: 0 for an unscaled row,
// else the bits of 1 / split_scale(m), which the GEMMs multiply into the row's outputs (exact: a power of two).
float f16_range_scale(float m)
{
    return m < 32768.0 ? 1.0 : split_scale(m);
}

// The factor undoing a row's range scale, from its scale word.
float f16_range_inverse(uint word)
{
    return word == 0 ? 1.0 : asfloat(word);
}

// Two values as packed f16 hi parts (returned) and lo parts.
uint split_h2(float a, float b, out uint lo)
{
    float ha = asfloat((asuint(a) + 0x1000u) & 0xFFFFE000u);
    float hb = asfloat((asuint(b) + 0x1000u) & 0xFFFFE000u);
    lo = f32tof16(a - ha) | (f32tof16(b - hb) << 16);
    return f32tof16(ha) | (f32tof16(hb) << 16);
}

// 8 values times s as packed hi parts (returned) and lo parts.
uint4 split8_scaled(float v[8], float s, out uint4 lo)
{
    uint4 hi;
    hi.x = split_h2(v[0] * s, v[1] * s, lo.x);
    hi.y = split_h2(v[2] * s, v[3] * s, lo.y);
    hi.z = split_h2(v[4] * s, v[5] * s, lo.z);
    hi.w = split_h2(v[6] * s, v[7] * s, lo.w);
    return hi;
}

// Grouped tile order of a 2D GEMM grid launched as (column tiles, token tiles): workgroup (x, y) is remapped so that consecutive
// workgroups walk `group` token tiles of one column tile before moving to the next column tile. The workgroups running at once then
// share a few weight tiles (read from DRAM about once per group) and the group's activation rows (which stay in L2), instead of
// all streaming every weight tile once per token tile. Returns (column tile, token tile); group 0 keeps the launch order.
uint2 grouped_tile(uint2 gid, uint colTiles, uint tokTiles, uint group)
{
    if (group == 0)
    {
        return gid;
    }
    uint linear = gid.y * colTiles + gid.x;
    uint per = group * colTiles;
    uint g = linear / per, r = linear % per;
    uint rows = min(group, tokTiles - g * group);
    return uint2(r / rows, g * group + r % rows);
}
