#pragma once
// Recurrent state element formats. STATE_FMT: 0 = f32 (default), 1 = f16, 2 = bf16. A 16-bit state is widened to f32 on load,
// updated in f32 and rounded to the state format (round to nearest even) after every update, so the state a kernel carries in
// registers is always exactly what it would store; state_store is then exact.
//   STATE_T            element type of the state buffer
//   state_load(x)      element -> f32
//   state_store(x)     f32 (already rounded) -> element
//   state_round(x)     f32 -> f32 rounded to the state format
// The roundings use integer operations: drivers may fold an f32 -> f16 -> f32 conversion round trip into a no-op.
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
float state_load(uint16_t x) { return h2f(x); }
float state_round(float x)
{
    uint b = asuint(x);
    // f16 normal range: keep 10 mantissa bits (round to nearest even); below it the conversion handles the subnormal steps.
    if (abs(x) >= 6.103515625e-5 && (b & 0x7FFFFFFFu) < 0x7F800000u) return asfloat((b + 0xFFFu + ((b >> 13) & 1u)) & 0xFFFFE000u);
    return h2f(f2h(x));
}
uint16_t state_store(float x) { return f2h(x); }
#else
typealias STATE_T = float;
float state_load(float x) { return x; }
float state_round(float x) { return x; }
float state_store(float x) { return x; }
#endif
