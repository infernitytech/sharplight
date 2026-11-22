#pragma once
// E4M3 activation quantization: the fp8 GEMM's activation rows (quantize_fp8 in gemm_fp8.slang, row norms writing them
// directly). Included by common.slang.

// E4M3 code of |v| <= 448, rounded to nearest even (subnormals below 2^-6 included).
uint e4m3_code(float v)
{
    uint sign = v < 0 ? 0x80u : 0u;
    float a = abs(v);
    uint code;
    if (a < 0.015625)
    {
        // Subnormals and zero: multiples of 2^-9 (a rounding up to 8 * 2^-9 is the smallest normal, code 8).
        code = uint(round(a * 512.0));
    }
    else
    {
        uint bits = asuint(a);
        bits += 0x7FFFFu + ((bits >> 20) & 1u);
        code = min((bits >> 20) - (120u << 3), 0x7Eu);
    }
    return sign | code;
}

// Scale d = amax / 448 of the 128-value group whose 4 values v this lane holds (a 32-lane subgroup covers the group; every
// lane of it must call this). E4M3_UE8M0: d rounded up to a power of two, 2^ceil(log2(amax / 448)) (exponent clamped to 2^-126 ..
// 2^127; an all-zero group gets 1), the UE8M0 scales of the block-scaled FP8 GEMM (gemm_fp8_bs.cu).
float e4m3_group_scale(float4 v)
{
    float amax = WaveActiveMax(max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w))));
#ifdef E4M3_UE8M0
    if (!(amax > 0))
    {
        return 1.0;
    }
    uint bits = asuint(amax / 448.0);
    uint e = ((bits >> 23) & 0xFFu) + ((bits & 0x7FFFFFu) != 0 ? 1u : 0u);
    return asfloat(clamp(e, 1u, 254u) << 23);
#else
    return amax / 448.0;
#endif
}

// The 4 codes of v / d, packed little-endian.
uint e4m3_pack4(float4 v, float d)
{
    float id = d > 0 ? 1.0 / d : 0.0;
    return e4m3_code(v.x * id) | (e4m3_code(v.y * id) << 8) | (e4m3_code(v.z * id) << 16) | (e4m3_code(v.w * id) << 24);
}
