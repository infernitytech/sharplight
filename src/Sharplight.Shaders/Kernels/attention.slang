// Causal attention of T queries against a KV cache (online softmax, one workgroup per (head, query)).
// q: [T, HQ, D] with strides (QS0, QS1); caches: [ctx, HKV, D]/[ctx, HKV, DV] of type KV_F16 ? f16 : f32;
// out: dense [T, HQ, DV]. Query t sees cache rows [0, n_past + t].
// Macros: HQ, HKV, D, DV, QS0, QS1, SCALE, KV_F16, WG (threads, = chunk of keys),
//         optional WINDOW (> 0: a query sees only its last WINDOW keys), HAS_SOFTCAP + SOFTCAP (cap value: s = SOFTCAP * tanh(s / SOFTCAP))
#include "common.slang"

struct Params
{
    float* q;
    uint8_t* kc;
    uint8_t* vc;
    float* o;
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

    float* qr = p.q + uint64_t(t) * QS0 + uint64_t(h) * QS1;
    for (uint i = tid; i < D; i += WG) sQ[i] = qr[i] * SCALE;
    GroupMemoryBarrierWithGroupSync();

    float m = -1e30, l = 0;
    float acc[DPT];
    [ForceUnroll] for (int i = 0; i < DPT; i++) acc[i] = 0;

#if WINDOW
    uint jStart = L > WINDOW ? L - WINDOW : 0;
#else
    uint jStart = 0;
#endif
    for (uint c0 = jStart; c0 < L; c0 += WG)
    {
        uint j = c0 + tid;
        float s = -1e30;
        if (j < L) s = dot_k(p.kc, (uint64_t(j) * HKV + hk) * D);
#if HAS_SOFTCAP
        if (j < L) s = SOFTCAP * tanh(s / SOFTCAP);
#endif
        float mc = reduce_max(s, tid);
        float mNew = max(m, mc);
        float pj = j < L ? exp(s - mNew) : 0;
        sP[tid] = pj;
        float corr = exp(m - mNew);
        l = l * corr + reduce_sum(pj, tid);   // reduce_sum syncs, so sP is visible afterwards
        m = mNew;
        uint cnt = min(uint(WG), L - c0);
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

    float* orow = p.o + (uint64_t(t) * HQ + h) * DV;
    float inv = 1.0 / l;
    [ForceUnroll]
    for (int di = 0; di < DPT; di++)
    {
        uint d = tid + di * WG;
        if (d < DV) orow[d] = acc[di] * inv;
    }
}
