// Row normalization over the innermost axis. Rows are addressed as [T, R1] with strides
// (S0, S1) (elements); the output is dense [T, R1, D].
// MODE 0 = RMS: x / sqrt(mean(x^2) + EPS)   MODE 1 = L2: x / sqrt(sum(x^2) + EPS)
// HAS_W: multiply by weight[D].  One subgroup per row.
// Macros: D, R1, S0, S1, EPS, MODE, HAS_W, SG, ROWS_PER_WG
#include "common.slang"

struct Params
{
    float* x;
    float* w;
    float* y;
    uint T;
};

#ifndef ROWS_PER_WG
#define ROWS_PER_WG 4
#endif

[shader("compute")]
[numthreads(SG * ROWS_PER_WG, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    uint lane = ltid.x % SG;
    uint row = gid.x * ROWS_PER_WG + ltid.x / SG;
    if (row >= p.T * R1) return;
    uint t = row / R1, r1 = row % R1;
    float* xr = p.x + uint64_t(t) * S0 + uint64_t(r1) * S1;
    float* yr = p.y + uint64_t(row) * D;

    float ss = 0;
    for (uint i = lane; i < D; i += SG) { float v = xr[i]; ss += v * v; }
    ss = WaveActiveSum(ss);
#if MODE == 0
    float scale = rsqrt(ss / float(D) + EPS);
#else
    float scale = rsqrt(ss + EPS);
#endif
    for (uint i = lane; i < D; i += SG)
    {
#if HAS_W
        yr[i] = xr[i] * scale * p.w[i];
#else
        yr[i] = xr[i] * scale;
#endif
    }
}
