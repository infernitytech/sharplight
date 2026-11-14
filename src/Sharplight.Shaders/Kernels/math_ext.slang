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

// Correctly rounded a / b (GPU division is only accurate to a few ulp): one exact-residual correction step. Quantization relies on it,
// since round(x / scale) must see the same ties as the CPU.
float div_ieee(float a, float b)
{
    float q = a / b;
    float r = fma(-q, b, a);
    return fma(r, 1.0 / b, q);
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
// Python-style remainder (sign of the divisor). An infinite divisor leaves a finite dividend of its sign unchanged and yields the divisor
// for the other sign (as numpy does); fmod alone would compute a - b·trunc(a/b) = NaN on GPUs.
float mod_py(float a, float b)
{
    if (isinf(b) && !isinf(a) && a == a)
    {
        return (a == 0.0 || ((a < 0.0) == (b < 0.0))) ? a : b;
    }

    float r = fmod(a, b);
    return (r != 0.0 && ((r < 0.0) != (b < 0.0))) ? r + b : r;
}

// ---- narrow formats and exact integer arithmetic (graph ops Narrow, integer Add/Div/Mod/..., bitwise ops) ----

// Rounds x to a float format with `mbits` mantissa bits and minimum normal exponent `emin`, to nearest even. Magnitudes above `maxv`
// saturate (overflow 0), become infinity (1) or NaN (2); `fnuz` formats have no negative zero; `nanNegZero` maps NaN to -0 (FP4 E2M1).
float narrow_float(float x, int mbits, int emin, float maxv, int overflow, bool fnuz, bool nanNegZero)
{
    if (x != x) return nanNegZero ? asfloat(0x80000000u) : x;
    float a = abs(x);
    float r = a;
    if (!isinf(a))
    {
        int e = max(int((asuint(a) >> 23) & 255u) - 127, emin);
        int k = mbits - e;
        r = round_even(a * asfloat(uint(127 + k) << 23)) * asfloat(uint(127 - k) << 23);
    }
    if (r > maxv) r = overflow == 0 ? maxv : asfloat(overflow == 1 ? 0x7F800000u : 0x7FC00000u);
    if (fnuz && r == 0.0) return 0.0;
    return asfloat(asuint(r) | (asuint(x) & 0x80000000u));
}
float narrow_bf16(float x) { return bf2f(f2bf(x)); }

int idiv(int a, int b) { return b == 0 ? 0 : (b == -1 ? -a : a / b); }
int imod(int a, int b, bool cStyle)
{
    if (b == 0 || b == -1) return 0;
    // Remainder by the truncating division: `%` on int may lower to an unsigned modulo.
    int r = a - (a / b) * b;
    return (!cStyle && r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}
int ipow(int a, int b)
{
    if (b < 0) return a == 1 ? 1 : (a == -1 ? ((b & 1) == 0 ? 1 : -1) : 0);
    int r = 1;
    int p = a;
    while (b > 0)
    {
        if ((b & 1) != 0) r *= p;
        p *= p;
        b >>= 1;
    }
    return r;
}
int ishl(int a, int b) { return (b < 0 || b >= 32) ? 0 : a << b; }
int ishr(int a, int b) { return (b < 0 || b >= 32) ? (a < 0 ? -1 : 0) : a >> b; }
int ushr(int a, int b) { return (b < 0 || b >= 32) ? 0 : int(uint(a) >> uint(b)); }
// Low `bits` bits of v (1..31), sign-extended when isSigned.
int wrap_int(int v, int bits, bool isSigned)
{
    int s = 32 - bits;
    return isSigned ? (v << s) >> s : int(uint(v) & ((1u << uint(bits)) - 1u));
}
