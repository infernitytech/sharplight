#pragma once
// Weight-format decoders for the 2/3-bit k-quants, the ternary BitNet formats and the 4-bit floating point formats.
// Included by common.slang (after the FMT_* ids and h2f); each load8_* returns the 8 consecutive values [8g, 8g+8) of row `row`
// exactly as the CPU dequantizer (Quant.KQuants.cs / Quant.Ternary.cs / Quant.Fp4.cs) and ggml's dequantize_row_* do.
// Block sizes are all even, so 16-bit loads are aligned; MXFP4 (17-byte blocks) is read bytewise.

// Q2_K: 256 values = { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin } (84 bytes); value = d*(sc & 15)*q2 - dmin*(sc >> 4) per 16 values.
void load8_q2_k(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 84;
    float d = h2f(*(uint16_t*)(blk + 80));
    float dmin = h2f(*(uint16_t*)(blk + 82));
    uint idx = (g & 31) * 8;
    uint n = idx >> 7;              // 128-value half
    uint r = idx & 127;
    uint j = r >> 5;                // 2-bit plane inside the half
    uint h = (r >> 4) & 1;          // 16-value sub-block inside the 32
    uint l0 = r & 15;
    uint sc = uint(blk[n * 8 + 2 * j + h]);
    float dl = d * float(sc & 15);
    float ml = dmin * float(sc >> 4);
    uint8_t* q = blk + 16 + 32 * n + 16 * h + l0;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = dl * float((uint(q[i]) >> (2 * j)) & 3) - ml;
}

// Q3_K: 256 values = { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d } (110 bytes); sixteen 6-bit scales (minus 32); value = q2 - (hmask bit ? 0 : 4).
void load8_q3_k(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 110;
    float dAll = h2f(*(uint16_t*)(blk + 108));
    uint idx = (g & 31) * 8;
    uint n = idx >> 7;
    uint r = idx & 127;
    uint j = r >> 5;
    uint h = (r >> 4) & 1;
    uint l0 = r & 15;
    uint isc = n * 8 + 2 * j + h;   // scale index 0..15
    uint lo4 = (uint(blk[96 + (isc & 7)]) >> (4 * (isc >> 3))) & 15;
    uint hi2 = (uint(blk[104 + (isc & 3)]) >> (2 * (isc >> 2))) & 3;
    float dl = dAll * float(int(lo4 | (hi2 << 4)) - 32);
    uint8_t* q = blk + 32 + 32 * n + 16 * h + l0;
    uint8_t* hm = blk + 16 * h + l0;
    uint hbit = 4 * n + j;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
    {
        int q3 = int((uint(q[i]) >> (2 * j)) & 3) - (((uint(hm[i]) >> hbit) & 1) != 0 ? 0 : 4);
        v[i] = dl * float(q3);
    }
}

// TQ2_0: 256 values = { u8 qs[64]; f16 d } (66 bytes); value = ((qs[32j + m] >> 2l) & 3) - 1 at index 128j + 32l + m.
void load8_tq2_0(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 66;
    float d = h2f(*(uint16_t*)(blk + 64));
    uint idx = (g & 31) * 8;
    uint j = idx >> 7;
    uint l = (idx >> 5) & 3;
    uint m0 = idx & 31;
    uint8_t* q = blk + 32 * j + m0;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = float(int((uint(q[i]) >> (2 * l)) & 3) - 1) * d;
}

// TQ1_0: 256 values = { u8 qs[48]; u8 qh[4]; f16 d } (54 bytes); base-3 digit n of a byte b is ((b * 3^n mod 256) * 3 >> 8) - 1.
// Values 0..159 are digits n = idx/32 of qs[idx%32], 160..239 digits n = (idx-160)/16 of qs[32 + (idx-160)%16], 240..255 digits n = (idx-240)/4 of qh[(idx-240)%4].
float tq1_trit(uint b, uint n)
{
    uint pw = n == 0 ? 1u : (n == 1 ? 3u : (n == 2 ? 9u : (n == 3 ? 27u : 81u)));
    return float(int((((b * pw) & 255u) * 3u) >> 8) - 1);
}

void load8_tq1_0(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 5) * 54;
    float d = h2f(*(uint16_t*)(blk + 52));
    uint idx = (g & 31) * 8;
    if (idx < 240)
    {
        // The eight values share one digit position n and read eight consecutive bytes.
        uint n = idx < 160 ? (idx >> 5) : ((idx - 160) >> 4);
        uint src = idx < 160 ? (idx & 31) : (32 + ((idx - 160) & 15));
        [ForceUnroll]
        for (int i = 0; i < 8; i++)
            v[i] = d * tq1_trit(uint(blk[src + i]), n);
    }
    else
    {
        // The last 16 values: digit n = k / 4 of qh byte k % 4 (k = idx - 240 + i).
        [ForceUnroll]
        for (int i = 0; i < 8; i++)
        {
            uint k = idx - 240 + uint(i);
            v[i] = d * tq1_trit(uint(blk[48 + (k & 3)]), k >> 2);
        }
    }
}

// FP4 (E2M1) magnitudes doubled: code 0..7 -> 0, 1, 2, 3, 4, 6, 8, 12; bit 3 is the sign.
float fp4_value(uint nib)
{
    uint m = nib & 7;
    float mag = m < 4 ? float(m) : float((2 + (m & 1)) << ((m >> 1) - 1));
    return (nib & 8) != 0 ? -mag : mag;
}

// MXFP4: 32 values = { u8 e; u8 qs[16] } (17 bytes); scale 2^(e-127)/2 (E8M0 halved, denormal for e < 2); layout like Q4_0.
float e8m0_half(uint e)
{
    return asfloat(e < 2 ? (0x00200000u << e) : ((e - 1) << 23));
}

void load8_mxfp4(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 2) * 17;
    float d = e8m0_half(uint(blk[0]));
    uint sub = g & 3;
    uint off = (sub & 1) * 8;
    uint shift = (sub >> 1) * 4;
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = fp4_value((uint(blk[1 + off + i]) >> shift) & 0xF) * d;
}

// NVFP4: 64 values = { u8 d[4] (UE4M3); u8 qs[32] } (36 bytes); four 16-value sub-blocks, each laid out like one MXFP4 half pair (low nibbles first).
float ue4m3_half(uint x)
{
    if (x == 0 || x == 0x7F) return 0.0;
    uint e = (x >> 3) & 15;
    uint man = x & 7;
    float raw = e == 0 ? float(man) * (1.0 / 512.0) : asfloat(((e + 120) << 23) | (man << 20));
    return raw * 0.5;
}

void load8_nvfp4(uint8_t* base, uint64_t rowBytes, uint row, uint g, out float v[8])
{
    uint8_t* blk = base + rowBytes * row + uint64_t(g >> 3) * 36;
    uint s = (g >> 1) & 3;
    uint shift = (g & 1) * 4;
    float d = ue4m3_half(uint(blk[s]));
    [ForceUnroll]
    for (int i = 0; i < 8; i++)
        v[i] = fp4_value((uint(blk[4 + 8 * s + i]) >> shift) & 0xF) * d;
}
