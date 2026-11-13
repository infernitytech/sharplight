// Mamba-2 selective scan (SelectiveScan op). One workgroup per (head h, block of PB rows of the head's state matrix S[p][n]);
// row p is owned by LPR lanes of one subgroup, each holding N/LPR state elements in registers (lanes interleaved: conflict-free
// shared reads); the per-row dot with C is a subgroup butterfly sum (LPR = 1 needs no subgroup ops at all).
//   dt' = max(softplus(dt + bias), DT_MIN);  S = exp(dt' A) S + dt' x B;  y = S . C + D x
// x [T, H, P] strides (XS0, XS1); dt [T, H] row stride DTS0; B, C [T, G, N] strides (BS0, BS1)/(CS0, CS1); state [H, P, N] f32 (MULTI_SEQ: [S, H, P, N], grid z = run of the
// [U, 4] runs table, a run at position 0 starts from zero state); y dense.
// Macros: H, P, G, N, XS0, XS1, DTS0, BS0, BS1, CS0, CS1, LPR, PB (P % PB == 0), TB (tokens staged per block), DT_MIN, MULTI_SEQ, A_STATE (A is [H, N]: a decay per state element, Mamba-1 with P = 1)
#include "common.slang"

#ifdef KERNEL_SSM_SCAN
struct SsmParams
{
    float* x;
    float* dt;
    float* dtb;
    float* a;
    float* b;
    float* c;
    float* d;
    float* state;
    float* y;
#if MULTI_SEQ
    int* runs;
#endif
    uint T;
};

#define EPL (N / LPR)
#define NT (PB * LPR)

groupshared float sB[TB * N];
groupshared float sC[TB * N];
groupshared float sx[TB * PB];
groupshared float sdt[TB];
groupshared float sdec[TB];

float row_sum(float v)
{
    [ForceUnroll]
    for (uint off = 1; off < LPR; off <<= 1)
        v += WaveReadLaneAt(v, WaveGetLaneIndex() ^ off);
    return v;
}

[shader("compute")]
[numthreads(NT, 1, 1)]
void ssm_scan(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform SsmParams p)
{
    uint h = gid.x;
    uint g = h / (H / G);
    uint pl = ltid.x / LPR;
    uint row = gid.y * PB + pl;
    uint part = ltid.x % LPR;
#if A_STATE
    float bias = p.dtb[h], A = 0.0, Dh = p.d[h];
#else
    float bias = p.dtb[h], A = p.a[h], Dh = p.d[h];
#endif

#if MULTI_SEQ
    int4 run = int4(p.runs[gid.z * 4], p.runs[gid.z * 4 + 1], p.runs[gid.z * 4 + 2], p.runs[gid.z * 4 + 3]);
    float* st = p.state + ((uint64_t(run.x) * H + h) * P + row) * N + part;
    uint tBeg = uint(run.y), tEnd = uint(run.y + run.z);
    bool fresh = run.w == 0;
#else
    float* st = p.state + (uint64_t(h) * P + row) * N + part;
    uint tBeg = 0, tEnd = p.T;
    bool fresh = false;
#endif
    float M[EPL];
    [ForceUnroll] for (int e = 0; e < EPL; e++) M[e] = fresh ? 0.0 : st[e * LPR];

    for (uint t0 = tBeg; t0 < tEnd; t0 += TB)
    {
        uint nt = min(uint(TB), tEnd - t0);
        GroupMemoryBarrierWithGroupSync();
        for (uint i = ltid.x; i < nt * N; i += NT)
        {
            uint tt = i / N, c = i % N;
            sB[i] = p.b[uint64_t(t0 + tt) * BS0 + uint64_t(g) * BS1 + c];
            sC[i] = p.c[uint64_t(t0 + tt) * CS0 + uint64_t(g) * CS1 + c];
        }
        for (uint i = ltid.x; i < nt * PB; i += NT)
        {
            uint tt = i / PB, c = i % PB;
            sx[i] = p.x[uint64_t(t0 + tt) * XS0 + uint64_t(h) * XS1 + gid.y * PB + c];
        }
        for (uint i = ltid.x; i < nt; i += NT)
        {
            float v = p.dt[uint64_t(t0 + i) * DTS0 + h] + bias;
            float s = v > 20.0 ? v : log(1.0 + exp(v));
            s = max(s, DT_MIN);
            sdt[i] = s;
            sdec[i] = exp(s * A);
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint tt = 0; tt < nt; tt++)
        {
#if A_STATE
            float dtt = sdt[tt];
#else
            float decay = sdec[tt];
#endif
            float xp = sx[tt * PB + pl];
            float coef = sdt[tt] * xp;
            uint base = tt * N + part;
            float ya[4] = { 0, 0, 0, 0 };
            [ForceUnroll] for (int e = 0; e < EPL; e++)
            {
#if A_STATE
                M[e] = M[e] * exp(dtt * p.a[uint64_t(h) * N + part + e * LPR]) + sB[base + e * LPR] * coef;
#else
                M[e] = M[e] * decay + sB[base + e * LPR] * coef;
#endif
                ya[e % 4] += M[e] * sC[base + e * LPR];
            }
            float yd = row_sum((ya[0] + ya[1]) + (ya[2] + ya[3]));
            if (part == 0) p.y[(uint64_t(t0 + tt) * H + h) * P + row] = yd + Dh * xp;
        }
    }
    [ForceUnroll] for (int e = 0; e < EPL; e++) st[e * LPR] = M[e];
}
#endif
