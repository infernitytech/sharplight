#pragma once
// Fused elementwise epilogue of the convolution kernels (conv.slang, conv_cm.slang).
// EPI = 1: the including generated source defines `float conv_epi(float v, float r)` (the fused group applied to the
//          conv result v, with r the residual element); the final store applies it.
// EPI_RES = 1: the params carry `float* res`, a dense tensor of y's shape read at the output index.
#ifndef EPI
#define EPI 0
#endif
#ifndef EPI_RES
#define EPI_RES 0
#endif
#if EPI && EPI_RES
#define CONV_EPI(v, idx) v = conv_epi(v, p.res[idx])
#elif EPI
#define CONV_EPI(v, idx) v = conv_epi(v, 0.0)
#else
#define CONV_EPI(v, idx)
#endif
