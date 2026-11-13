// Recurrent-state kernels: causal depthwise conv with carried state, gated delta rule.
#include "common.slang"

#ifdef KERNEL_CONV1D
// ---------------------------------------------------------------------------------------------
// CausalConv1d: x [T, C] (row stride XS), weight [C, KS] f32, state [KS-1, C] f32 (oldest first),
// y dense [T, C]. One thread per channel; walks tokens sequentially and updates the state.
// Macros: C, KS, XS, ACT_SILU
struct ConvParams
{
    float* x;
    float* w;
    float* state;
    float* y;
    uint T;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void conv1d(uint3 dtid : SV_DispatchThreadID, uniform ConvParams p)
{
    uint c = dtid.x;
    if (c >= C) return;
    float win[KS];
    [ForceUnroll] for (int j = 0; j < KS - 1; j++) win[j] = p.state[uint64_t(j) * C + c];
    float wk[KS];
    [ForceUnroll] for (int j = 0; j < KS; j++) wk[j] = p.w[uint64_t(c) * KS + j];

    for (uint t = 0; t < p.T; t++)
    {
        win[KS - 1] = p.x[uint64_t(t) * XS + c];
        float s = 0;
        [ForceUnroll] for (int j = 0; j < KS; j++) s += win[j] * wk[j];
#if ACT_SILU
        s = s / (1.0 + exp(-s));
#endif
        p.y[uint64_t(t) * C + c] = s;
        [ForceUnroll] for (int j = 0; j < KS - 1; j++) win[j] = win[j + 1];
    }
    [ForceUnroll] for (int j = 0; j < KS - 1; j++) p.state[uint64_t(j) * C + c] = win[j];
}

#endif

#ifdef KERNEL_DELTA_RULE
// ---------------------------------------------------------------------------------------------
// GatedDeltaRule. One workgroup per value head h; row j (0..DV) of the state matrix M[h][j][i]
// (i over DK) is owned by LPR lanes, each holding DK/LPR elements in registers.
//   M <- exp(g) M;  delta_j = (v_j - M_j . k) * beta;  M_j += delta_j k;  o_j = (M_j . q) * SCALE
// q,k: [T, HK, DK] strides (QS0,QS1)/(KS0,KS1); v: [T, HV, DV] strides (VS0, VS1);
// g, beta: [T, HV] dense; state [HV, DV, DK] f32; out dense [T, HV, DV].
// Macros: HK, HV, DK, DV, QS0, QS1, KS0, KS1, VS0, VS1, SCALE, LPR (lanes per row, power of 2 <= SG),
// DVB (state rows per workgroup; rows are independent, so a head is split over DV/DVB workgroups).
struct DeltaParams
{
    float* q;
    float* k;
    float* v;
    float* g;
    float* beta;
    float* state;
    float* o;
    uint T;
};

#ifndef LPR
#define LPR 4
#endif
#define EPL (DK / LPR)
#ifndef DVB
#define DVB DV
#endif

// Tokens are processed in blocks of TB: q, k, v, g, beta of a block are staged in shared memory with one
// load phase (and, when NORM_Q/NORM_K, the L2 norms computed once per token), then the sequential update
// runs from shared memory.
// NORM_Q / NORM_K: q / k arrive unnormalized and are L2-normalized here (x * rsqrt(sum x^2 + EPS_*)).
#ifndef NORM_Q
#define NORM_Q 0
#endif
#ifndef NORM_K
#define NORM_K 0
#endif
#ifndef TB
#define TB 16
#endif
#define NT (DVB * LPR)

groupshared float sq[TB * DK];
groupshared float sk[TB * DK];
groupshared float sv[TB * DVB];
groupshared float sgd[TB];
groupshared float sbeta[TB];
groupshared float sqn[TB];
groupshared float skn[TB];

float row_sum(float v)
{
    [ForceUnroll]
    for (uint off = 1; off < LPR; off <<= 1)
        v += WaveReadLaneAt(v, WaveGetLaneIndex() ^ off);
    return v;
}

[shader("compute")]
[numthreads(NT, 1, 1)]
void delta_rule(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform DeltaParams p)
{
    uint h = gid.x;
    uint hk = h % HK;
    uint jl = ltid.x / LPR;                 // state row within this workgroup
    uint j = gid.y * DVB + jl;              // state row (value dim)
    uint part = ltid.x % LPR;               // which slice of DK this lane owns
    uint i0 = part * EPL;

    float* st = p.state + (uint64_t(h) * DV + j) * DK + i0;
    float M[EPL];
    [ForceUnroll] for (int e = 0; e < EPL; e++) M[e] = st[e];

    for (uint t0 = 0; t0 < p.T; t0 += TB)
    {
        uint nt = min(uint(TB), p.T - t0);
        GroupMemoryBarrierWithGroupSync();
        for (uint i = ltid.x; i < nt * DK; i += NT)
        {
            uint tt = i / DK, c = i % DK;
            sq[i] = p.q[uint64_t(t0 + tt) * QS0 + uint64_t(hk) * QS1 + c];
            sk[i] = p.k[uint64_t(t0 + tt) * KS0 + uint64_t(hk) * KS1 + c];
        }
        for (uint i = ltid.x; i < nt * DVB; i += NT)
        {
            uint tt = i / DVB, c = i % DVB;
            sv[i] = p.v[uint64_t(t0 + tt) * VS0 + uint64_t(h) * VS1 + gid.y * DVB + c];
        }
        for (uint i = ltid.x; i < nt; i += NT)
        {
            sgd[i] = exp(p.g[uint64_t(t0 + i) * HV + h]);
            sbeta[i] = p.beta[uint64_t(t0 + i) * HV + h];
        }
        GroupMemoryBarrierWithGroupSync();
#if NORM_Q || NORM_K
        {
            uint lanes = WaveGetLaneCount(), lane = WaveGetLaneIndex();
            uint nsg = (NT + lanes - 1) / lanes, sgi = ltid.x / lanes;
            for (uint tt = sgi; tt < nt; tt += nsg)
            {
                float a = 0, c = 0;
                for (uint i = lane; i < DK; i += lanes) { float x = sq[tt * DK + i], y = sk[tt * DK + i]; a += x * x; c += y * y; }
                a = WaveActiveSum(a); c = WaveActiveSum(c);
                if (lane == 0)
                {
                    sqn[tt] = NORM_Q ? rsqrt(a + EPS_Q) : 1.0;
                    skn[tt] = NORM_K ? rsqrt(c + EPS_K) : 1.0;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
#endif
        for (uint tt = 0; tt < nt; tt++)
        {
#if NORM_Q
            float qn = sqn[tt];
#else
            float qn = 1.0;
#endif
#if NORM_K
            float kn = skn[tt];
#else
            float kn = 1.0;
#endif
            float decay = sgd[tt];
            float b = sbeta[tt];
            float vj = sv[tt * DVB + jl];
            uint base = tt * DK + i0;

            float kd = 0;
            [ForceUnroll] for (int e = 0; e < EPL; e++) { M[e] *= decay; kd += M[e] * sk[base + e]; }
            kd = row_sum(kd) * kn;
            float delta = (vj - kd) * b;
            float od = 0;
            float dk = delta * kn;
            [ForceUnroll] for (int e = 0; e < EPL; e++) { M[e] += sk[base + e] * dk; od += M[e] * sq[base + e]; }
            od = row_sum(od) * qn;
            if (part == 0) p.o[(uint64_t(t0 + tt) * HV + h) * DV + j] = od * SCALE;
        }
    }
    [ForceUnroll] for (int e = 0; e < EPL; e++) st[e] = M[e];
}
#endif
