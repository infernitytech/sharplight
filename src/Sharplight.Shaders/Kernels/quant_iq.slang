#pragma once
// Weight-format decoders for ggml's grid based IQ formats. Included by common.slang (after the FMT_* ids and h2f). Each format's lattice
// grid lives in its own grid_*.slang file and is only compiled into kernels specialized for that format. All block sizes are even,
// so the 16-bit loads below are aligned; 32-bit words are assembled from two halves.
// The results equal the CPU dequantizers (Quant.Iq.cs) and ggml's dequantize_row_*.

uint rd16(uint8_t* p, uint o) { return uint(*(uint16_t*)(p + o)); }
uint rd32(uint8_t* p, uint o) { return rd16(p, o) | (rd16(p, o + 2) << 16); }

#if WFMT == FMT_IQ2_XXS || WFMT == FMT_IQ2_XS || WFMT == FMT_IQ2_S || WFMT == FMT_IQ3_XXS
#include "grid_iq_signs.slang"
// 7-bit sign index -> sign byte (bit j set = value j negative; the eighth bit is the parity of the other seven).
uint iq_sign_byte(uint i) { return (iq_signs[i >> 2] >> (8 * (i & 3))) & 255; }
#endif

#if WFMT == FMT_IQ2_XXS || WFMT == FMT_IQ2_XS || WFMT == FMT_IQ2_S
// Byte j (0..7) of the 8-byte grid entry `idx` of a table stored as two 32-bit words per entry.
#define IQ2_GRID_BYTE(table, idx, j) float((table[2 * (idx) + ((j) >> 2)] >> (8 * ((j) & 3))) & 255)
#endif

#if WFMT == FMT_IQ2_XXS
#include "grid_iq2_xxs.slang"
// IQ2_XXS: 256 values = { f16 d; u16 qs[32] } (66 bytes); per 32 values: 4 grid indices (8 bit), then 4 sign indices (7 bit) and a 4-bit scale.
void load8_iq2_xxs(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 66;
    float d = h2f(*(uint16_t*)blk);
    uint gg = g & 31;
    uint ib32 = gg >> 2;
    uint l = gg & 3;
    uint w0 = rd32(blk, 2 + 8 * ib32);
    uint w1 = rd32(blk, 6 + 8 * ib32);
    float db = d * (0.5 + float(w1 >> 28)) * 0.25;
    uint idx = (w0 >> (8 * l)) & 255;
    uint signs = iq_sign_byte((w1 >> (7 * l)) & 127);
    [ForceUnroll]
    for (int j = 0; j < 8; j++)
    {
        float x = db * IQ2_GRID_BYTE(iq2xxs_grid, idx, j);
        v[j] = ((signs >> j) & 1) != 0 ? -x : x;
    }
}
#endif

#if WFMT == FMT_IQ2_XS
#include "grid_iq2_xs.slang"
// IQ2_XS: 256 values = { f16 d; u16 qs[32]; u8 scales[8] } (74 bytes); each u16 = 9-bit grid index | 7-bit sign index << 9; two 4-bit scales per 32 values.
void load8_iq2_xs(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 74;
    float d = h2f(*(uint16_t*)blk);
    uint gg = g & 31;
    uint ib32 = gg >> 2;
    uint l = gg & 3;
    uint q = rd16(blk, 2 + 2 * gg);
    uint sc = uint(blk[66 + ib32]);
    float db = d * (0.5 + float(l < 2 ? (sc & 15) : (sc >> 4))) * 0.25;
    uint idx = q & 511;
    uint signs = iq_sign_byte(q >> 9);
    [ForceUnroll]
    for (int j = 0; j < 8; j++)
    {
        float x = db * IQ2_GRID_BYTE(iq2xs_grid, idx, j);
        v[j] = ((signs >> j) & 1) != 0 ? -x : x;
    }
}
#endif

#if WFMT == FMT_IQ2_S
#include "grid_iq2_s.slang"
// IQ2_S: 256 values = { f16 d; u8 qs[64]; u8 qh[8]; u8 scales[8] } (82 bytes); qs[0..32) low grid index bits, qs[32..64) sign bytes, qh the two high bits.
void load8_iq2_s(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 82;
    float d = h2f(*(uint16_t*)blk);
    uint gg = g & 31;
    uint ib32 = gg >> 2;
    uint l = gg & 3;
    uint sc = uint(blk[74 + ib32]);
    float db = d * (0.5 + float(l < 2 ? (sc & 15) : (sc >> 4))) * 0.25;
    uint idx = uint(blk[2 + gg]) | ((uint(blk[66 + ib32]) << (8 - 2 * l)) & 0x300);
    uint signs = uint(blk[34 + gg]);
    [ForceUnroll]
    for (int j = 0; j < 8; j++)
    {
        float x = db * IQ2_GRID_BYTE(iq2s_grid, idx, j);
        v[j] = ((signs >> j) & 1) != 0 ? -x : x;
    }
}
#endif

#if WFMT == FMT_IQ3_XXS
#include "grid_iq3_xxs.slang"
// IQ3_XXS: 256 values = { f16 d; u8 qs[96] } (98 bytes); qs[0..64) 8-bit indices of 4-value grid entries, then one word per 32 values: 4 sign indices (7 bit) and a 4-bit scale.
void load8_iq3_xxs(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 98;
    float d = h2f(*(uint16_t*)blk);
    uint gg = g & 31;
    uint ib32 = gg >> 2;
    uint l = gg & 3;
    uint word = rd32(blk, 66 + 4 * ib32);
    float db = d * (0.5 + float(word >> 28)) * 0.5;
    uint signs = iq_sign_byte((word >> (7 * l)) & 127);
    uint g1 = iq3xxs_grid[blk[2 + 8 * ib32 + 2 * l]];
    uint g2 = iq3xxs_grid[blk[2 + 8 * ib32 + 2 * l + 1]];
    [ForceUnroll]
    for (int j = 0; j < 4; j++)
    {
        float x1 = db * float((g1 >> (8 * j)) & 255);
        float x2 = db * float((g2 >> (8 * j)) & 255);
        v[j] = ((signs >> j) & 1) != 0 ? -x1 : x1;
        v[j + 4] = ((signs >> (j + 4)) & 1) != 0 ? -x2 : x2;
    }
}
#endif

#if WFMT == FMT_IQ3_S
#include "grid_iq3_s.slang"
// IQ3_S: 256 values = { f16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4] } (110 bytes); 9-bit indices of 4-value grid entries, explicit sign bytes, scale 1 + 2 * nibble.
void load8_iq3_s(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 110;
    float d = h2f(*(uint16_t*)blk);
    uint gg = g & 31;
    uint ib32 = gg >> 2;
    uint l = gg & 3;
    float db = d * float(1 + 2 * ((uint(blk[106 + (ib32 >> 1)]) >> (4 * (ib32 & 1))) & 15));
    uint qh = uint(blk[66 + ib32]);
    uint i1 = uint(blk[2 + 8 * ib32 + 2 * l]) | ((qh << (8 - 2 * l)) & 256);
    uint i2 = uint(blk[2 + 8 * ib32 + 2 * l + 1]) | ((qh << (7 - 2 * l)) & 256);
    uint signs = uint(blk[74 + 4 * ib32 + l]);
    uint g1 = iq3s_grid[i1];
    uint g2 = iq3s_grid[i2];
    [ForceUnroll]
    for (int j = 0; j < 4; j++)
    {
        float x1 = db * float((g1 >> (8 * j)) & 255);
        float x2 = db * float((g2 >> (8 * j)) & 255);
        v[j] = ((signs >> j) & 1) != 0 ? -x1 : x1;
        v[j + 4] = ((signs >> (j + 4)) & 1) != 0 ? -x2 : x2;
    }
}
#endif

#if WFMT == FMT_IQ1_S || WFMT == FMT_IQ1_M
#include "grid_iq1.slang"
#define IQ1_DELTA 0.125
// Value j (0..7) of grid entry idx: the table stores value + 1 per nibble (low nibbles = values 0..3, high nibbles = values 4..7).
float iq1_grid_value(uint entry, uint j)
{
    return float(int((entry >> (8 * (j & 3) + 4 * (j >> 2))) & 15) - 1);
}
#endif

#if WFMT == FMT_IQ1_S
// IQ1_S: 256 values = { f16 d; u8 qs[32]; u16 qh[8] } (50 bytes); qh per 32 values: 3 high index bits per 8 values, 3-bit scale at bits 12..14, delta sign at bit 15.
void load8_iq1_s(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 50;
    float d = h2f(*(uint16_t*)blk);
    uint gg = g & 31;
    uint ib = gg >> 2;
    uint l = gg & 3;
    uint qh = rd16(blk, 34 + 2 * ib);
    float dl = d * float(2 * ((qh >> 12) & 7) + 1);
    float delta = (qh & 0x8000) != 0 ? -IQ1_DELTA : IQ1_DELTA;
    uint entry = iq1s_grid_gpu[uint(blk[2 + gg]) | (((qh >> (3 * l)) & 7) << 8)];
    [ForceUnroll]
    for (int j = 0; j < 8; j++)
        v[j] = dl * (iq1_grid_value(entry, uint(j)) + delta);
}
#endif

#if WFMT == FMT_IQ1_M
// IQ1_M: 256 values = { u8 qs[32]; u8 qh[16]; u16 scales[4] } (56 bytes); the f16 super-scale is spread over the top nibbles of the scale words,
// each 16 values have a 3-bit scale, each 8 values a delta sign (qh bit 3 / 7) and 3 high index bits.
void load8_iq1_m(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 56;
    uint s0 = rd16(blk, 48), s1 = rd16(blk, 50), s2 = rd16(blk, 52), s3 = rd16(blk, 54);
    float d = h2f(uint16_t((s0 >> 12) | ((s1 >> 8) & 0x00F0) | ((s2 >> 4) & 0x0F00) | (s3 & 0xF000)));
    uint gg = g & 31;
    uint ib = gg >> 2;
    uint l = gg & 3;
    uint sc = rd16(blk, 48 + 2 * (ib >> 1));
    float dl = d * float(2 * ((sc >> (6 * (ib & 1) + (l >= 2 ? 3 : 0))) & 7) + 1);
    uint qhb = uint(blk[32 + 2 * ib + (l >> 1)]);
    uint high = (l & 1) == 0 ? ((qhb << 8) & 0x700) : ((qhb << 4) & 0x700);
    float delta = (qhb & ((l & 1) == 0 ? 0x08u : 0x80u)) != 0 ? -IQ1_DELTA : IQ1_DELTA;
    uint entry = iq1s_grid_gpu[uint(blk[4 * ib + l]) | high];
    [ForceUnroll]
    for (int j = 0; j < 8; j++)
        v[j] = dl * (iq1_grid_value(entry, uint(j)) + delta);
}
#endif
