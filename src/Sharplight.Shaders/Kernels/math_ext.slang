#pragma once
// Extra math helpers for generated elementwise kernels (accurate erf, ONNX-compatible pow).
#include "common.slang"

// erf: Maclaurin series for |x| < 1 (alternating but fast converging), Numerical Recipes erfc
// Chebyshev fit (relative error < 1.2e-7 on erfc) otherwise. Absolute error < ~2e-8 everywhere.
float erf_accurate(float x)
{
    float ax = abs(x);
    float r;
    if (ax < 1.0)
    {
        float t = ax * ax;
        float s = -1.0892221037148573e-09;
        s = s * t + 1.3122532963802806e-08;
        s = s * t + -1.4503852223150468e-07;
        s = s * t + 1.4589169000933706e-06;
        s = s * t + -1.3227513227513228e-05;
        s = s * t + 0.00010683760683760684;
        s = s * t + -0.0007575757575757576;
        s = s * t + 0.004629629629629629;
        s = s * t + -0.023809523809523808;
        s = s * t + 0.1;
        s = s * t + -0.3333333333333333;
        s = s * t + 1.0;
        r = 1.1283791670955126 * ax * s;
    }
    else
    {
        float t = 1.0 / (1.0 + 0.5 * ax);
        float poly = -1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
                     t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
        float erfc = t * exp(-ax * ax + poly);
        r = 1.0 - erfc;
    }
    return x < 0.0 ? -r : r;
}

// ONNX Pow: integer exponents work for negative bases; pow(x, 0) = 1.
float pow_onnx(float a, float b)
{
    if (b == 0.0) return 1.0;
    if (a < 0.0)
    {
        float rb = round(b);
        if (rb == b)
        {
            float m = pow(-a, b);
            return (fmod(abs(b), 2.0) == 1.0) ? -m : m;
        }
        return asfloat(0x7FC00000u);
    }
    if (a == 0.0) return b > 0.0 ? 0.0 : asfloat(0x7F800000u);
    return pow(a, b);
}

// float -> bfloat16 bits, round to nearest even.
uint16_t f2bf(float f)
{
    uint b = asuint(f);
    if ((b & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t((b >> 16) | 0x40u);
    b += 0x7FFFu + ((b >> 16) & 1u);
    return uint16_t(b >> 16);
}

// round half to even
float round_even(float x)
{
    if (abs(x) >= 8388608.0) return x;
    float r = floor(x + 0.5);
    if (r - x == 0.5 && fmod(r, 2.0) != 0.0) r -= 1.0;
    return r;
}
float sign_f(float x) { return x != x ? x : (x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0)); }
float asinh_f(float x) { float a = abs(x); float r = log(a + sqrt(a * a + 1.0)); return x < 0.0 ? -r : r; }
float mod_py(float a, float b) { float r = fmod(a, b); return (r != 0.0 && ((r < 0.0) != (b < 0.0))) ? r + b : r; }
