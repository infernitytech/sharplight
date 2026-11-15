#pragma once
// Decoders of the group-scaled formats of quantized Hugging Face checkpoints. Included by common.slang (after the FMT_* ids and
// quant_k.slang, whose fp4_value / ue4m3_half it uses); the CPU reference is Quant.Grouped.cs / Quant.Fp4.cs.
//  * nvfp4_gs: { f32 g; u8 d[4]; u8 qs[32] } (40 bytes): an NVFP4 block after its global scale g.
//  * q4z_g32 / q4z_g128: { f16 d; f16 z; u8 qs[G / 2] } (20 / 68 bytes), 32-value runs of 16 bytes laid out like Q4_0 (value j
//    in the low nibble of byte j, j + 16 in the high one); value = (q - z) * d.
//  * q8z_g32 / q8z_g128: { f16 d; f16 z; u8 q[G] } (36 / 132 bytes); value = (q - z) * d.
//  * fp4_b64: nf4_b64 with the FP4 code book; q8a_g32: { f32 d; i8 q[32] } (36 bytes), value = q * d * (1/127);
//    q2s_g128: { f32 d; u8 qs[32] } (36 bytes), byte j holds values j, 32 + j, 64 + j, 96 + j in bits 6, 4, 2, 0, value = (code - 1) * d.
//  * nf4_b64: { f32 d; u8 qs[32] } (36 bytes), two Q4_0-style runs; value = d * NF4[q], the code book read from the grid
//    buffer (HAS_GRID, GridTables.cs): a table in kernel-local memory is indexed divergently and is several times slower.
// Every block is 4-byte aligned (when rows are), so the codes are read as 32-bit words.

// Eight consecutive bytes at a 4-byte aligned address.
uint2 load8_bytes(uint8_t* p)
{
    uint* w = (uint*)p;
    return uint2(w[0], w[1]);
}

// Eight 4-bit codes: the low (shift 0) or high (shift 4) nibbles of eight bytes, value i in bits 4i.. of the result pair.
uint code4(uint2 q, uint shift, uint i)
{
    return ((i < 4 ? q.x : q.y) >> (8 * (i & 3) + shift)) & 0xF;
}

void load8_nvfp4_gs(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 3) * 40;
    uint s = (g >> 1) & 3;
    uint shift = (g & 1) * 4;
    float d = ue4m3_half(uint(blk[4 + s])) * asfloat(*(uint*)blk);
    uint2 q = load8_bytes(blk + 8 + 8 * s);
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = fp4_value(code4(q, shift, i)) * d;
}

// Q4Z with `runs` 32-value runs per block (1 for group 32, 4 for group 128).
void load8_q4z(uint8_t* base, uint64_t rowBytes, uint row, uint g, uint runs, out float v[8])
{
    uint groupsPerBlock = runs * 4;
    uint8_t* blk = base + rowBytes * row + uint64_t(g / groupsPerBlock) * (4 + 16 * runs);
    uint inBlock = g % groupsPerBlock;
    uint run = inBlock >> 2, sub = inBlock & 3;
    uint dz = *(uint*)blk;
    float d = f16tof32(dz & 0xFFFF);
    float z = f16tof32(dz >> 16);
    uint2 q = load8_bytes(blk + 4 + 16 * run + (sub & 1) * 8);
    uint shift = (sub >> 1) * 4;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = (float(code4(q, shift, i)) - z) * d;
}

// Q8Z with groups of `group` values (32 or 128).
void load8_q8z(uint8_t* base, uint64_t rowBytes, uint row, uint g, uint group, out float v[8])
{
    uint groupsPerBlock = group / 8;
    uint8_t* blk = base + rowBytes * row + uint64_t(g / groupsPerBlock) * (4 + group);
    uint dz = *(uint*)blk;
    float d = f16tof32(dz & 0xFFFF);
    float z = f16tof32(dz >> 16);
    uint2 q = load8_bytes(blk + 4 + 8 * (g % groupsPerBlock));
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = (float(((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0xFF) - z) * d;
}

#if WFMT == FMT_NF4_B64 || WFMT == FMT_FP4_B64
void load8_nf4_b64(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 3) * 36;
    float d = asfloat(*(uint*)blk);
    uint run = (g >> 2) & 1, sub = g & 3;
    uint2 q = load8_bytes(blk + 4 + 16 * run + (sub & 1) * 8);
    uint shift = (sub >> 1) * 4;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = asfloat(g_grid[code4(q, shift, i)]) * d;
}
#endif

// Q8A: 32 signed codes with an f32 absmax; value = q * d * (1/127) in this order, as the reference computes it.
void load8_q8a(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 2) * 36;
    float d = asfloat(*(uint*)blk);
    uint2 q = load8_bytes(blk + 4 + 8 * (g & 3));
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
    {
        int c = int(((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0xFF);
        c = (c << 24) >> 24;
        v[i] = float(c) * d * 7.874015718698502e-3;
    }
}

// Q2S: ternary codes, group 128, f32 scale; group g of 8 values is bytes 8 * (g & 3).. of the run of codes selected by g >> 2 & 3.
void load8_q2s(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 4) * 36;
    float d = asfloat(*(uint*)blk);
    uint plane = (g >> 2) & 3, sub = g & 3;
    uint2 q = load8_bytes(blk + 4 + 8 * sub);
    uint shift = 6 - 2 * plane;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = float(int((((i < 4 ? q.x : q.y) >> (8 * (i & 3) + shift)) & 3)) - 1) * d;
}
