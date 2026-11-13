// Causal attention of T queries against a KV cache (online softmax, one workgroup per (head, query)).
// q: [T, HQ, D] with strides (QS0, QS1); caches: [ctx, HKV, D]/[ctx, HKV, DV] of type KV_F16 ? f16 : f32;
// out: dense [T, HQ, DV]. Query t sees cache rows [0, n_past + t].
// Macros: HQ, HKV, D, DV, QS0, QS1, SCALE, KV_F16, WG (threads, = chunk of keys),
//         WINDOW (> 0: a query sees only its last WINDOW keys; sliding-window layers), HAS_SOFTCAP + SOFTCAP
//         (cap value: scores = SOFTCAP * tanh(scores / SOFTCAP)). The flash-attention variant does not support
//         WINDOW / SOFTCAP (the lowering does not select it for them).
// SPLITS > 1 (flash decoding, few queries): the keys of each (head, query) are split into SPLITS contiguous ranges,
// one workgroup each (grid z); every split writes an unnormalized partial (acc[DV], m, l) to part[split][t][h][DV+2]
// and the `combine` entry merges them. The split ranges follow n_past at run time, so a recorded command buffer
// stays valid as the context grows.
#include "common.slang"

#ifndef SPLITS
#define SPLITS 1
#endif

#ifdef KERNEL_COMBINE

struct CombineParams
{
    float* part;
    float* o;
    uint T;
};

#ifndef WG
#define WG 128
#endif

[shader("compute")]
[numthreads(WG, 1, 1)]
void combine(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform CombineParams p)
{
    uint h = gid.x, t = gid.y;
    float wgt[SPLITS];
    float M = -1e30;
    [ForceUnroll] for (uint s = 0; s < SPLITS; s++) M = max(M, p.part[((uint64_t(s) * p.T + t) * HQ + h) * (DV + 2) + DV]);
    float den = 0;
    [ForceUnroll] for (uint s = 0; s < SPLITS; s++)
    {
        float* ps = p.part + ((uint64_t(s) * p.T + t) * HQ + h) * (DV + 2);
        wgt[s] = exp(ps[DV] - M);
        den += wgt[s] * ps[DV + 1];
    }
    float inv = 1.0 / den;
    float* orow = p.o + (uint64_t(t) * HQ + h) * DV;
    for (uint d = ltid.x; d < DV; d += WG)
    {
        float num = 0;
        [ForceUnroll] for (uint s = 0; s < SPLITS; s++) num += wgt[s] * p.part[((uint64_t(s) * p.T + t) * HQ + h) * (DV + 2) + d];
        orow[d] = num * inv;
    }
}

#else

struct Params
{
    float* q;
    uint8_t* kc;
    uint8_t* vc;
    float* o;
    float* part;
    uint* rt;
    uint T;
};

#ifndef WG
#define WG 128
#endif
#define DPT ((DV + WG - 1) / WG)   // output dims per thread

groupshared float sQ[D];
groupshared float sP[WG];
groupshared float sRed[WG / SG];

float reduce_max(float v, uint tid)
{
    v = WaveActiveMax(v);
    if (tid % SG == 0) sRed[tid / SG] = v;
    GroupMemoryBarrierWithGroupSync();
    float r = sRed[0];
    [ForceUnroll] for (uint i = 1; i < WG / SG; i++) r = max(r, sRed[i]);
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float reduce_sum(float v, uint tid)
{
    v = WaveActiveSum(v);
    if (tid % SG == 0) sRed[tid / SG] = v;
    GroupMemoryBarrierWithGroupSync();
    float r = 0;
    [ForceUnroll] for (uint i = 0; i < WG / SG; i++) r += sRed[i];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float load_kv(uint8_t* base, uint64_t elem)
{
#if KV_F16
    return h2f(((uint16_t*)base)[elem]);
#else
    return ((float*)base)[elem];
#endif
}

float dot_k(uint8_t* kc, uint64_t rowElem)
{
    float s = 0;
#if KV_F16
    uint4* kp = (uint4*)(kc + rowElem * 2);
    [ForceUnroll]
    for (uint i = 0; i < D / 8; i++)
    {
        uint4 u = kp[i];
        s += sQ[8 * i + 0] * f16tof32(u.x & 0xFFFF) + sQ[8 * i + 1] * f16tof32(u.x >> 16)
           + sQ[8 * i + 2] * f16tof32(u.y & 0xFFFF) + sQ[8 * i + 3] * f16tof32(u.y >> 16)
           + sQ[8 * i + 4] * f16tof32(u.z & 0xFFFF) + sQ[8 * i + 5] * f16tof32(u.z >> 16)
           + sQ[8 * i + 6] * f16tof32(u.w & 0xFFFF) + sQ[8 * i + 7] * f16tof32(u.w >> 16);
    }
#else
    float4* kp = (float4*)((float*)kc + rowElem);
    [ForceUnroll]
    for (uint i = 0; i < D / 4; i++)
    {
        float4 k = kp[i];
        s += sQ[4 * i] * k.x + sQ[4 * i + 1] * k.y + sQ[4 * i + 2] * k.z + sQ[4 * i + 3] * k.w;
    }
#endif
    return s;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    uint h = gid.x;
    uint t = gid.y;
    uint tid = ltid.x;
    uint hk = h / (HQ / HKV);
    uint L = p.rt[RT_N_PAST] + t + 1;
#if WINDOW
    uint jLo = L > WINDOW ? L - WINDOW : 0;
#else
    uint jLo = 0;
#endif
#if SPLITS > 1
    uint split = gid.z;
    uint span = L - jLo;
    uint per = (span + SPLITS - 1) / SPLITS;
    uint jBeg = jLo + min(span, split * per), jEnd = jLo + min(span, split * per + per);
#else
    uint jBeg = jLo, jEnd = L;
#endif

    float* qr = p.q + uint64_t(t) * QS0 + uint64_t(h) * QS1;
    for (uint i = tid; i < D; i += WG) sQ[i] = qr[i] * SCALE;
    GroupMemoryBarrierWithGroupSync();

    float m = -1e30, l = 0;
    float acc[DPT];
    [ForceUnroll] for (int i = 0; i < DPT; i++) acc[i] = 0;

    for (uint c0 = jBeg; c0 < jEnd; c0 += WG)
    {
        uint j = c0 + tid;
        float s = -1e30;
        if (j < jEnd) s = dot_k(p.kc, (uint64_t(j) * HKV + hk) * D);
#if HAS_SOFTCAP
        if (j < jEnd) s = SOFTCAP * tanh(s / SOFTCAP);
#endif
        float mc = reduce_max(s, tid);
        float mNew = max(m, mc);
        float pj = j < jEnd ? exp(s - mNew) : 0;
        sP[tid] = pj;
        float corr = exp(m - mNew);
        l = l * corr + reduce_sum(pj, tid);   // reduce_sum syncs, so sP is visible afterwards
        m = mNew;
        uint cnt = min(uint(WG), jEnd - c0);
        [ForceUnroll]
        for (int di = 0; di < DPT; di++)
        {
            uint d = tid + di * WG;
            float a = acc[di] * corr;
            if (d < DV)
                for (uint jj = 0; jj < cnt; jj++)
                    a += sP[jj] * load_kv(p.vc, (uint64_t(c0 + jj) * HKV + hk) * DV + d);
            acc[di] = a;
        }
        GroupMemoryBarrierWithGroupSync();
    }

#if SPLITS > 1
    float* ps = p.part + ((uint64_t(split) * p.T + t) * HQ + h) * (DV + 2);
    [ForceUnroll]
    for (int di = 0; di < DPT; di++)
    {
        uint d = tid + di * WG;
        if (d < DV) ps[d] = acc[di];
    }
    if (tid == 0) { ps[DV] = m; ps[DV + 1] = l; }
#else
    float* orow = p.o + (uint64_t(t) * HQ + h) * DV;
    float inv = 1.0 / l;
    [ForceUnroll]
    for (int di = 0; di < DPT; di++)
    {
        uint d = tid + di * WG;
        if (d < DV) orow[d] = acc[di] * inv;
    }
#endif
}

#ifdef KERNEL_FA
// ---------------------------------------------------------------------------------------------
// Flash attention for many queries (prefill) on tensor cores. A workgroup handles FA_BR consecutive queries of one
// head with 4 subgroups. Per FA_BC-key block: S = Q·Kᵀ with cooperative matrices (stored to shared memory), online
// softmax in scalar code — each thread owns the statistics of one query row and a quarter of its output row in
// registers, so rescaling rows is plain arithmetic — then P·V with cooperative matrices, staged through shared memory
// in FA_DC-wide chunks and accumulated into the registers. Causal: query t sees keys [0, n_past + t].
// Requires f16 caches, D % 16 == 0, DV % 64 == 0, D, DV <= 256, 32-wide subgroups with quad operations.
using namespace linalg;
#define FA_BR 32
#define FA_BC 32
#define FA_T 128
#define FA_KVD (D > DV ? D : DV)
#define FA_QS (D + 8)
#define FA_KS (FA_KVD + 8)
#define FA_SS FA_BC
#define FA_PS (FA_BC + 8)
#define FA_DC 64
#define FA_NC (DV / FA_DC)
typealias FaA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias FaB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias FaC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

groupshared half faQ[FA_BR * FA_QS];
groupshared half faKV[FA_BC * FA_KS];
groupshared float faS[FA_BR * FA_SS];
groupshared half faP[FA_BR * FA_PS];
groupshared float faO[FA_BR * FA_DC];

void fa_load_kv(uint8_t* cache, uint j0, uint L, uint hk, uint dim, uint tid)
{
    uint per = dim / 8;
    for (uint i = tid; i < FA_BC * per; i += FA_T)
    {
        uint r = i / per, c = (i % per) * 8;
        uint j = j0 + r;
        uint4 u = 0;
        if (j < L) u = *(uint4*)(cache + ((uint64_t(j) * HKV + hk) * dim + c) * 2);
        uint o = r * FA_KS + c;
        faKV[o + 0] = half(f16tof32(u.x & 0xFFFF)); faKV[o + 1] = half(f16tof32(u.x >> 16));
        faKV[o + 2] = half(f16tof32(u.y & 0xFFFF)); faKV[o + 3] = half(f16tof32(u.y >> 16));
        faKV[o + 4] = half(f16tof32(u.z & 0xFFFF)); faKV[o + 5] = half(f16tof32(u.z >> 16));
        faKV[o + 6] = half(f16tof32(u.w & 0xFFFF)); faKV[o + 7] = half(f16tof32(u.w >> 16));
    }
}

[shader("compute")]
[numthreads(FA_T, 1, 1)]
void fa(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    uint h = gid.x, t0 = gid.y * FA_BR;
    uint tid = ltid.x, sg = tid / 32;
    uint hk = h / (HQ / HKV);
    uint nPast = p.rt[RT_N_PAST];

    for (uint i = tid; i < FA_BR * D; i += FA_T)
    {
        uint r = i / D, d = i % D, t = t0 + r;
        faQ[r * FA_QS + d] = half(t < p.T ? p.q[uint64_t(t) * QS0 + uint64_t(h) * QS1 + d] * SCALE : 0.0);
    }

    uint row = tid / 4, sub = tid % 4;
    uint tq = t0 + row;
    bool rowValid = tq < p.T;
    float m = -1e30, l = 0;
    float o[FA_NC * 16];   // output dims c * FA_DC + sub * 16 + e
    [ForceUnroll] for (uint i = 0; i < FA_NC * 16; i++) o[i] = 0;
    uint L = nPast + min(p.T, t0 + FA_BR);   // keys needed by the block's last query
    GroupMemoryBarrierWithGroupSync();

    for (uint j0 = 0; j0 < L; j0 += FA_BC)
    {
        fa_load_kv(p.kc, j0, L, hk, D, tid);
        GroupMemoryBarrierWithGroupSync();

        // S: subgroup sg computes rows (sg / 2) * 16, keys (sg % 2) * 16.
        {
            FaC sacc = FaC(0.0);
            uint qa = (sg / 2) * 16 * FA_QS, kb = (sg % 2) * 16 * FA_KS;
            [ForceUnroll]
            for (uint k = 0; k < D; k += 16)
            {
                FaA a = FaA.Load<CoopMatMatrixLayout.RowMajor>(faQ, qa + k, FA_QS);
                FaB b = FaB.Load<CoopMatMatrixLayout.ColumnMajor>(faKV, kb + k, FA_KS);
                sacc = coopMatMulAdd<float, false>(a, b, sacc);
            }
            sacc.Store<CoopMatMatrixLayout.RowMajor>(faS, (sg / 2) * 16 * FA_SS + (sg % 2) * 16, FA_SS);
        }
        GroupMemoryBarrierWithGroupSync();   // S visible; K no longer needed

        fa_load_kv(p.vc, j0, L, hk, DV, tid);

        // Online softmax of row `row`, columns sub * 8 .. + 8 (the row's 4 threads are adjacent lanes).
        float sv[8];
        float mx = -1e30;
        [ForceUnroll]
        for (uint e = 0; e < 8; e++)
        {
            uint c = sub * 8 + e, j = j0 + c;
            sv[e] = rowValid && j <= nPast + tq ? faS[row * FA_SS + c] : -1e30;
            mx = max(mx, sv[e]);
        }
        mx = max(mx, QuadReadAcrossX(mx));
        mx = max(mx, QuadReadAcrossY(mx));
        float mNew = max(m, mx);
        float corr = exp(m - mNew);
        float ps = 0;
        [ForceUnroll]
        for (uint e = 0; e < 8; e++)
        {
            float pe = sv[e] > -1e29 ? exp(sv[e] - mNew) : 0.0;
            ps += pe;
            faP[row * FA_PS + sub * 8 + e] = half(pe);
        }
        ps += QuadReadAcrossX(ps);
        ps += QuadReadAcrossY(ps);
        l = l * corr + ps;
        m = mNew;
        [ForceUnroll] for (uint i = 0; i < FA_NC * 16; i++) o[i] *= corr;
        GroupMemoryBarrierWithGroupSync();   // P and V visible

        // P·V in chunks of FA_DC output dims: 2 x 4 tiles of 16x16, two per subgroup.
        [ForceUnroll]
        for (uint c = 0; c < FA_NC; c++)
        {
            [ForceUnroll]
            for (uint tt = 0; tt < 2; tt++)
            {
                uint tile = sg * 2 + tt, rb = tile / 4, cb = tile % 4;
                FaC acc = FaC(0.0);
                [ForceUnroll]
                for (uint k = 0; k < FA_BC; k += 16)
                {
                    FaA a = FaA.Load<CoopMatMatrixLayout.RowMajor>(faP, rb * 16 * FA_PS + k, FA_PS);
                    FaB b = FaB.Load<CoopMatMatrixLayout.RowMajor>(faKV, k * FA_KS + c * FA_DC + cb * 16, FA_KS);
                    acc = coopMatMulAdd<float, false>(a, b, acc);
                }
                acc.Store<CoopMatMatrixLayout.RowMajor>(faO, rb * 16 * FA_DC + cb * 16, FA_DC);
            }
            GroupMemoryBarrierWithGroupSync();
            [ForceUnroll] for (uint e = 0; e < 16; e++) o[c * 16 + e] += faO[row * FA_DC + sub * 16 + e];
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (rowValid)
    {
        float inv = l > 0 ? 1.0 / l : 0.0;
        float* orow = p.o + (uint64_t(tq) * HQ + h) * DV;
        [ForceUnroll]
        for (uint c = 0; c < FA_NC; c++)
            [ForceUnroll] for (uint e = 0; e < 16; e++) orow[c * FA_DC + sub * 16 + e] = o[c * 16 + e] * inv;
    }
}
#endif

#endif
