// Recurrent-state kernels: causal depthwise conv with carried state, gated delta rule.
//
// ROLLBACK (multi-sequence rollback form, see RecurrentRollback): two more bindings after the runs table, commits [U, 4] i32
// (replay rows, read bank, tentative flag, 0) and records [S, 2, KR, RW] f32. A run first replays the committed rows of the
// slot's last tentative run from the read bank (state update only, no outputs); a tentative run then stores the state as it is
// after the replay, records its own rows into the other bank and leaves the state unchanged by them.
#include "common.slang"

#ifndef ROLLBACK
#define ROLLBACK 0
#endif

#ifdef KERNEL_CONV1D
// ---------------------------------------------------------------------------------------------
// CausalConv1d: x [T, C] (row stride XS), weight [C, KS] f32, state [KS-1, C] f32 (oldest first),
// y dense [T, C]. One thread per channel; walks tokens sequentially and updates the state.
// Macros: C, KS, XS, ACT_SILU, HAS_BIAS (optional bias [C] added before the activation)
struct ConvParams
{
    float* x;
    float* w;
    float* state;
    float* y;
    float* bias;
#if MULTI_SEQ
    int* runs;   // [U, 4] (slot, first row, row count, first position); grid z = run; state [S, KS-1, C]
#endif
#if ROLLBACK
    int* commits;
    float* records;   // record row: x [C]
#endif
    uint T;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void conv1d(uint3 dtid : SV_DispatchThreadID, uniform ConvParams p)
{
    uint c = dtid.x;
    if (c >= C) return;
#if MULTI_SEQ
    int4 run = int4(p.runs[(dtid.z) * 4], p.runs[(dtid.z) * 4 + 1], p.runs[(dtid.z) * 4 + 2], p.runs[(dtid.z) * 4 + 3]);
    float* state = p.state + uint64_t(run.x) * (KS - 1) * C;
    uint tBeg = uint(run.y), tEnd = uint(run.y + run.z);
    bool fresh = run.w == 0;   // a sequence starting at position 0 starts from zero state
#else
    float* state = p.state;
    uint tBeg = 0, tEnd = p.T;
    bool fresh = false;
#endif
    float win[KS];
    [ForceUnroll] for (int j = 0; j < KS - 1; j++) win[j] = fresh ? 0.0 : state[uint64_t(j) * C + c];
    float wk[KS];
    [ForceUnroll] for (int j = 0; j < KS; j++) wk[j] = p.w[uint64_t(c) * KS + j];
#if ROLLBACK
    int4 cm = int4(p.commits[dtid.z * 4], p.commits[dtid.z * 4 + 1], p.commits[dtid.z * 4 + 2], p.commits[dtid.z * 4 + 3]);
    bool tentative = cm.z != 0;
    float* recIn = p.records + (uint64_t(run.x) * 2 + uint(cm.y)) * KR * RW + c;
    float* recOut = p.records + (uint64_t(run.x) * 2 + uint(1 - cm.y)) * KR * RW + c;
    uint replay = fresh ? 0u : uint(cm.x);
    for (uint r = 0; r < replay; r++)
    {
        win[KS - 1] = recIn[uint64_t(r) * RW];
        [ForceUnroll] for (int j = 0; j < KS - 1; j++) win[j] = win[j + 1];
    }
    if (tentative)
    {
        [ForceUnroll] for (int j = 0; j < KS - 1; j++) state[uint64_t(j) * C + c] = win[j];
    }
#else
    bool tentative = false;
#endif

    for (uint t = tBeg; t < tEnd; t++)
    {
        win[KS - 1] = p.x[uint64_t(t) * XS + c];
#if ROLLBACK
        if (tentative) recOut[uint64_t(t - tBeg) * RW] = win[KS - 1];
#endif
        float s = 0;
        [ForceUnroll] for (int j = 0; j < KS; j++) s += win[j] * wk[j];
#if HAS_BIAS
        s += p.bias[c];
#endif
#if ACT_SILU
        s = s / (1.0 + exp(-s));
#endif
        p.y[uint64_t(t) * C + c] = s;
        [ForceUnroll] for (int j = 0; j < KS - 1; j++) win[j] = win[j + 1];
    }
    if (!tentative)
    {
        [ForceUnroll] for (int j = 0; j < KS - 1; j++) state[uint64_t(j) * C + c] = win[j];
    }
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
#if MULTI_SEQ
    int* runs;   // [U, 4] (slot, first row, row count, first position); grid z = run; state [S, HV, DV, DK]
#endif
#if ROLLBACK
    int* commits;
    float* records;   // record row: per value head h at h * (DK + DV + 2): k [DK] (as read, before an absorbed L2 norm), v [DV], g, beta
#endif
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
    uint part = ltid.x % LPR;               // this lane owns DK elements part + e * LPR (interleaved: conflict-free shared reads)

#if MULTI_SEQ
    int4 run = int4(p.runs[(gid.z) * 4], p.runs[(gid.z) * 4 + 1], p.runs[(gid.z) * 4 + 2], p.runs[(gid.z) * 4 + 3]);
    float* st = p.state + ((uint64_t(run.x) * HV + h) * DV + j) * DK + part;
    uint tBeg = uint(run.y), tEnd = uint(run.y + run.z);
    bool fresh = run.w == 0;   // a sequence starting at position 0 starts from zero state
#else
    float* st = p.state + (uint64_t(h) * DV + j) * DK + part;
    uint tBeg = 0, tEnd = p.T;
    bool fresh = false;
#endif
    float M[EPL];
    [ForceUnroll] for (int e = 0; e < EPL; e++) M[e] = fresh ? 0.0 : st[e * LPR];
#if ROLLBACK
    // Rows r < replay come from the read bank (replayed, no outputs); row r >= replay is token tBeg + r - replay.
    int4 cm = int4(p.commits[gid.z * 4], p.commits[gid.z * 4 + 1], p.commits[gid.z * 4 + 2], p.commits[gid.z * 4 + 3]);
    uint replay = fresh ? 0u : uint(cm.x);
    bool tentative = cm.z != 0;
    bool recordShared = tentative && gid.y == 0;
    float* recIn = p.records + (uint64_t(run.x) * 2 + uint(cm.y)) * KR * RW + uint64_t(h) * (DK + DV + 2);
    float* recOut = p.records + (uint64_t(run.x) * 2 + uint(1 - cm.y)) * KR * RW + uint64_t(h) * (DK + DV + 2);
#else
    const uint replay = 0;
    const bool tentative = false;
#endif
    uint rEnd = replay + (tEnd - tBeg);

    for (uint r0 = 0; r0 < rEnd; r0 += TB)
    {
        uint nt = min(uint(TB), rEnd - r0);
        GroupMemoryBarrierWithGroupSync();
        for (uint i = ltid.x; i < nt * DK; i += NT)
        {
            uint tt = i / DK, c = i % DK, r = r0 + tt;
#if ROLLBACK
            if (r < replay) { sq[i] = 0.0; sk[i] = recIn[uint64_t(r) * RW + c]; continue; }
#endif
            uint64_t t = tBeg + r - replay;
            sq[i] = p.q[t * QS0 + uint64_t(hk) * QS1 + c];
            sk[i] = p.k[t * KS0 + uint64_t(hk) * KS1 + c];
#if ROLLBACK
            if (recordShared) recOut[uint64_t(r - replay) * RW + c] = sk[i];
#endif
        }
        for (uint i = ltid.x; i < nt * DVB; i += NT)
        {
            uint tt = i / DVB, c = i % DVB, r = r0 + tt;
#if ROLLBACK
            if (r < replay) { sv[i] = recIn[uint64_t(r) * RW + DK + gid.y * DVB + c]; continue; }
#endif
            sv[i] = p.v[(tBeg + r - replay) * uint64_t(VS0) + uint64_t(h) * VS1 + gid.y * DVB + c];
#if ROLLBACK
            if (tentative) recOut[uint64_t(r - replay) * RW + DK + gid.y * DVB + c] = sv[i];
#endif
        }
        for (uint i = ltid.x; i < nt; i += NT)
        {
            uint r = r0 + i;
            float gr, br;
#if ROLLBACK
            if (r < replay)
            {
                gr = recIn[uint64_t(r) * RW + DK + DV];
                br = recIn[uint64_t(r) * RW + DK + DV + 1];
            }
            else
#endif
            {
                uint64_t t = tBeg + r - replay;
                gr = p.g[t * HV + h];
                br = p.beta[t * HV + h];
#if ROLLBACK
                if (recordShared)
                {
                    recOut[uint64_t(r - replay) * RW + DK + DV] = gr;
                    recOut[uint64_t(r - replay) * RW + DK + DV + 1] = br;
                }
#endif
            }
            sgd[i] = exp(gr);
            sbeta[i] = br;
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
#if ROLLBACK
            // A tentative run leaves the slot's state as it is after the replay.
            if (tentative && r0 + tt == replay)
            {
                [ForceUnroll] for (int e = 0; e < EPL; e++) st[e * LPR] = M[e];
            }
#endif
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
            uint base = tt * DK + part;

            // Independent partial sums keep the dependency chains short (the loop is latency-bound).
            float kv[EPL];
            float ka[4] = { 0, 0, 0, 0 };
            [ForceUnroll] for (int e = 0; e < EPL; e++) { kv[e] = sk[base + e * LPR]; M[e] *= decay; ka[e % 4] += M[e] * kv[e]; }
            float kd = row_sum((ka[0] + ka[1]) + (ka[2] + ka[3])) * kn;
            float delta = (vj - kd) * b;
            float dk = delta * kn;
            float oa[4] = { 0, 0, 0, 0 };
            [ForceUnroll] for (int e = 0; e < EPL; e++) { M[e] += kv[e] * dk; oa[e % 4] += M[e] * sq[base + e * LPR]; }
            float od = row_sum((oa[0] + oa[1]) + (oa[2] + oa[3])) * qn;
            if (part == 0 && r0 + tt >= replay) p.o[(uint64_t(tBeg + r0 + tt - replay) * HV + h) * DV + j] = od * SCALE;
        }
    }
    if (!tentative)
    {
        [ForceUnroll] for (int e = 0; e < EPL; e++) st[e * LPR] = M[e];
    }
}
#endif
