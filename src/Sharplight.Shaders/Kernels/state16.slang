#pragma once
// Recurrent state element formats. STATE_FMT: 0 = f32 (default), 1 = f16, 2 = bf16. A 16-bit state is widened to f32 on load,
// updated in f32 and rounded to the state format (round to nearest even) after every update, so the state a kernel carries in
// registers is always exactly what it would store; state_store is then exact.
//   STATE_T            element type of the state buffer
//   state_load(x)      element -> f32
//   state_store(x)     f32 (already rounded) -> element
//   state_round(x)     f32 -> f32 rounded to the state format
// The roundings use integer operations: drivers may fold an f32 -> f16 -> f32 conversion round trip into a no-op, and the
// rounding mode of the f32 -> f16 conversion instruction is implementation-defined.
#include "common.slang"

#ifndef STATE_FMT
#define STATE_FMT 0
#endif

#if STATE_FMT == 2
typealias STATE_T = uint16_t;
float state_load(uint16_t x) { return bf2f(x); }
float state_round(float x)
{
    uint b = asuint(x);
    if ((b & 0x7FFFFFFFu) > 0x7F800000u) return x;   // NaN stays NaN
    return asfloat((b + 0x7FFFu + ((b >> 16) & 1u)) & 0xFFFF0000u);
}
uint16_t state_store(float x) { return uint16_t(asuint(x) >> 16); }
#elif STATE_FMT == 1
typealias STATE_T = uint16_t;
// f32 -> f16 bits, round to nearest even (overflow to infinity, subnormals kept), in integer operations: the conversion
// instruction's rounding mode is implementation-defined (some drivers truncate).
uint16_t f2h_rne(float x)
{
    uint b = asuint(x);
    uint sign = (b >> 16) & 0x8000u;
    uint a = b & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return uint16_t(sign | 0x7E00u);          // NaN
    if (a >= 0x477FF000u) return uint16_t(sign | 0x7C00u);         // rounds to or is infinity
    if (a >= 0x38800000u)                                          // f16 normal range
    {
        a += 0xFFFu + ((a >> 13) & 1u);
        return uint16_t(sign | ((a - 0x38000000u) >> 13));
    }
    if (a <= 0x33000000u) return uint16_t(sign);                   // at most half the smallest subnormal: zero
    uint m = (a & 0x7FFFFFu) | 0x800000u;
    uint shift = 126u - (a >> 23);                                 // 14..24: to units of 2^-24
    uint r = (m + (1u << (shift - 1u)) - 1u + ((m >> shift) & 1u)) >> shift;
    return uint16_t(sign | r);
}
float state_load(uint16_t x) { return h2f(x); }
float state_round(float x) { return h2f(f2h_rne(x)); }
uint16_t state_store(float x) { return f2h_rne(x); }
#else
typealias STATE_T = float;
float state_load(float x) { return x; }
float state_round(float x) { return x; }
float state_store(float x) { return x; }
#endif
