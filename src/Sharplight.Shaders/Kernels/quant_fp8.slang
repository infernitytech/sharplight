#pragma once
// FP8 (OCP E4M3FN / E5M2) weight decoders. Included by common.slang (after the FMT_* ids); the CPU reference is Quant.Fp8.cs.
//  * f8e4m3 / f8e5m2: one code per byte, rows of K bytes.
//  * f8e4m3_b32 / f8e4m3_b128: blocks of { f32 d; u8 q[B] } (36 / 132 bytes, 4-byte aligned) along the row, value = d * e4m3(q).
//  * f8e4m3_b32r / f8e4m3_b128r (FP8_SPLIT): the same rows repacked on upload to [K codes][K / B f32 scales].
//
// Decoding goes through half bits, exact for every code: an E4M3 code's sign moved to bit 15 and its 7 exponent/mantissa bits
// to bits 7..13 is the half code * 2^-8 (the exponent biases differ by 8; E4M3 subnormals m * 2^-9 become the half subnormals
// m * 2^-17). E5M2 is the upper byte of a half. One multiply by 256 * d then applies the block scale. NaN codes stay NaN.

#if defined(WFMT) && (WFMT == FMT_F8E4M3 || WFMT == FMT_F8E5M2 || WFMT == FMT_F8E4M3_B32 || WFMT == FMT_F8E4M3_B128 \
                      || WFMT == FMT_F8E4M3_B32R || WFMT == FMT_F8E4M3_B128R)
#define WFMT_FP8 1
#if WFMT == FMT_F8E4M3_B32 || WFMT == FMT_F8E4M3_B32R
#define FP8_BLOCK 32
#elif WFMT == FMT_F8E4M3_B128 || WFMT == FMT_F8E4M3_B128R
#define FP8_BLOCK 128
#else
#define FP8_BLOCK 0
#endif
#endif
// FP8_SPLIT: rows in the aligned split layout [K codes][K / B f32 scales] (16-byte aligned codes).
#if defined(WFMT) && (WFMT == FMT_F8E4M3_B32R || WFMT == FMT_F8E4M3_B128R)
#define FP8_SPLIT 1
#else
#define FP8_SPLIT 0
#endif

// Two FP8 codes (bytes 0,1 of u, or bytes 2,3 when hi) as packed half bits: E4M3 as value * 2^-8, E5M2 exactly.
uint fp8x2_bits(uint u, bool hi, bool e5m2)
{
    uint v = hi ? (u >> 16) : (u & 0xFFFFu);
    if (e5m2)
    {
        return ((v & 0xFFu) << 8) | ((v & 0xFF00u) << 16);
    }
    return ((v & 0x7Fu) << 7) | ((v & 0x80u) << 8) | ((v & 0x7F00u) << 15) | ((v & 0x8000u) << 16);
}

// Address of the 8 codes [8g, 8g + 8) of row `row` and the multiplier of their half bits (256 * block scale for E4M3, 1 for
// E5M2). blockValues: 0 for unscaled codes (rows of K bytes), else 32 or 128 (blocks of 4 + blockValues bytes).
uint64_t fp8_group(uint8_t* base, uint64_t rowBytes, uint row, uint g, uint blockValues, bool e5m2, out float multiplier)
{
    uint64_t at = uint64_t(base) + rowBytes * row;
    multiplier = e5m2 ? 1.0 : 256.0;
    if (blockValues == 0)
    {
        return at + uint64_t(g) * 8;
    }

    uint groupsPerBlock = blockValues / 8;
#if FP8_SPLIT
    // Codes first (rowBytes * B / (B + 4) of them), then one f32 scale per block.
    multiplier *= asfloat(*(uint*)(at + rowBytes / (blockValues + 4) * blockValues + (g / groupsPerBlock) * 4));
    return at + uint64_t(g) * 8;
#else
    uint64_t block = at + uint64_t(g / groupsPerBlock) * (blockValues + 4);
    multiplier *= asfloat(*(uint*)block);
    return block + 4 + (g % groupsPerBlock) * 8;
#endif
}

// The 8 values [8g, 8g + 8) of row `row` as f32.
void load8_fp8(uint8_t* base, uint64_t rowBytes, uint row, uint g, uint blockValues, bool e5m2, out float v[8])
{
    float s;
    uint* p = (uint*)fp8_group(base, rowBytes, row, g, blockValues, e5m2, s);
    uint c0 = p[0], c1 = p[1];
    uint h[4] = { fp8x2_bits(c0, false, e5m2), fp8x2_bits(c0, true, e5m2), fp8x2_bits(c1, false, e5m2), fp8x2_bits(c1, true, e5m2) };
    [ForceUnroll]
    for (int i = 0; i < 4; i++)
    {
        v[2 * i] = f16tof32(h[i] & 0xFFFFu) * s;
        v[2 * i + 1] = f16tof32(h[i] >> 16) * s;
    }
}

#ifdef WFMT_FP8
// Packed-half decoding for the tensor-core GEMMs (needs Float16).

// Packed half bits times s: one half2 multiply while s is a normal half (one rounding beyond s's own), else in f32.
uint fp8_scale_h2(uint bits, float s)
{
    if (abs(s) >= 6.103515625e-05 && abs(s) <= 65504.0)
    {
        return bit_cast<uint>(bit_cast<half2>(bits) * half2(half(s), half(s)));
    }
    return f32tof16(f16tof32(bits & 0xFFFFu) * s) | (f32tof16(f16tof32(bits >> 16) * s) << 16);
}

// Packed halves of 4 codes (one word).
uint2 fp8x4_h(uint c, float s)
{
    bool e5m2 = WFMT == FMT_F8E5M2;
    uint lo = fp8x2_bits(c, false, e5m2);
    uint hi = fp8x2_bits(c, true, e5m2);
    if (e5m2)
    {
        return uint2(lo, hi);   // exact halves, unscaled
    }
    return uint2(fp8_scale_h2(lo, s), fp8_scale_h2(hi, s));
}

// The 8 weights [8g, 8g + 8) of row `row` as packed halves.
uint4 load8h_fp8(uint8_t* w, uint64_t rowBytes, uint row, uint g)
{
    float s;
    uint* p = (uint*)fp8_group(w, rowBytes, row, g, FP8_BLOCK, WFMT == FMT_F8E5M2, s);
    uint2 a = fp8x4_h(p[0], s), b = fp8x4_h(p[1], s);
    return uint4(a, b);
}

// The 16 weights [16 g16, 16 g16 + 16) as packed halves (lo: first 8). Both groups lie in one block (blocks hold 32 or 128 values).
void load16h_fp8(uint8_t* w, uint64_t rowBytes, uint row, uint g16, out uint4 lo, out uint4 hi)
{
    float s;
    uint* p = (uint*)fp8_group(w, rowBytes, row, g16 * 2, FP8_BLOCK, WFMT == FMT_F8E5M2, s);
#if FP8_SPLIT || FP8_BLOCK == 0
    uint4 q = *(uint4*)p;   // 16-byte aligned codes
#else
    uint4 q = uint4(p[0], p[1], p[2], p[3]);
#endif
    lo = uint4(fp8x4_h(q.x, s), fp8x4_h(q.y, s));
    hi = uint4(fp8x4_h(q.z, s), fp8x4_h(q.w, s));
}
#endif
