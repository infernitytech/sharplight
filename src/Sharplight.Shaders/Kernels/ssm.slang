// Mamba-2 selective scan (SelectiveScan op). One workgroup per (head h, block of PB rows of the head's state matrix S[p][n]);
// row p is owned by LPR lanes of one subgroup, each holding N/LPR state elements in registers (lanes interleaved: conflict-free
// shared reads); the per-row dot with C is a subgroup butterfly sum (LPR = 1 needs no subgroup ops at all).
//   dt' = max(softplus(dt + bias), DT_MIN);  S = exp(dt' A) S + dt' x B;  y = S . C + D x
// x [T, H, P] strides (XS0, XS1); dt [T, H] row stride DTS0; B, C [T, G, N] strides (BS0, BS1)/(CS0, CS1); state [H, P, N] f32 (MULTI_SEQ: [S, H, P, N], grid z = run of the
// [U, 4] runs table, a run at position 0 starts from zero state); y dense.
// Macros: H, P, G, N, XS0, XS1, DTS0, BS0, BS1, CS0, CS1, LPR, PB (P % PB == 0), TB (tokens staged per block), DT_MIN, MULTI_SEQ, A_STATE (A is [H, N]: a decay per state element, Mamba-1 with P = 1)
// ROLLBACK, KR, RW: rollback form (see recurrent.slang); record row: per head h at h * (P + 1) x [P] and the raw dt, then B [G, N] at H * (P + 1).
#include "common.slang"

#ifndef ROLLBACK
#define ROLLBACK 0
#endif

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
#if ROLLBACK
    int* commits;
    float* records;
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
#if ROLLBACK
    // Rows r < replay come from the read bank (replayed, no outputs); row r >= replay is token tBeg + r - replay.
    int4 cm = int4(p.commits[gid.z * 4], p.commits[gid.z * 4 + 1], p.commits[gid.z * 4 + 2], p.commits[gid.z * 4 + 3]);
    uint replay = fresh ? 0u : uint(cm.x);
    bool tentative = cm.z != 0;
    float* recIn = p.records + (uint64_t(run.x) * 2 + uint(cm.y)) * KR * RW;
    float* recOut = p.records + (uint64_t(run.x) * 2 + uint(1 - cm.y)) * KR * RW;
    bool recordHead = tentative && gid.y == 0;
    bool recordB = recordHead && h % (H / G) == 0;
#else
    const uint replay = 0;
    const bool tentative = false;
#endif
    uint rEnd = replay + (tEnd - tBeg);

    for (uint r0 = 0; r0 < rEnd; r0 += TB)
    {
        uint nt = min(uint(TB), rEnd - r0);
        GroupMemoryBarrierWithGroupSync();
        for (uint i = ltid.x; i < nt * N; i += NT)
        {
            uint tt = i / N, c = i % N, r = r0 + tt;
#if ROLLBACK
            if (r < replay) { sB[i] = recIn[uint64_t(r) * RW + H * (P + 1) + g * N + c]; sC[i] = 0.0; continue; }
#endif
            uint64_t t = tBeg + r - replay;
            sB[i] = p.b[t * BS0 + uint64_t(g) * BS1 + c];
            sC[i] = p.c[t * CS0 + uint64_t(g) * CS1 + c];
#if ROLLBACK
            if (recordB) recOut[uint64_t(r - replay) * RW + H * (P + 1) + g * N + c] = sB[i];
#endif
        }
        for (uint i = ltid.x; i < nt * PB; i += NT)
        {
            uint tt = i / PB, c = i % PB, r = r0 + tt;
#if ROLLBACK
            if (r < replay) { sx[i] = recIn[uint64_t(r) * RW + h * (P + 1) + gid.y * PB + c]; continue; }
#endif
            sx[i] = p.x[(tBeg + r - replay) * uint64_t(XS0) + uint64_t(h) * XS1 + gid.y * PB + c];
#if ROLLBACK
            if (tentative) recOut[uint64_t(r - replay) * RW + h * (P + 1) + gid.y * PB + c] = sx[i];
#endif
        }
        for (uint i = ltid.x; i < nt; i += NT)
        {
            uint r = r0 + i;
            float raw;
#if ROLLBACK
            if (r < replay)
            {
                raw = recIn[uint64_t(r) * RW + h * (P + 1) + P];
            }
            else
#endif
            {
                raw = p.dt[(tBeg + r - replay) * uint64_t(DTS0) + h];
#if ROLLBACK
                if (recordHead) recOut[uint64_t(r - replay) * RW + h * (P + 1) + P] = raw;
#endif
            }
            float v = raw + bias;
            float s = v > 20.0 ? v : log(1.0 + exp(v));
            s = max(s, DT_MIN);
            sdt[i] = s;
            sdec[i] = exp(s * A);
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint tt = 0; tt < nt; tt++)
        {
#if ROLLBACK
            // A tentative run leaves the slot's state as it is after the replay.
            if (tentative && r0 + tt == replay)
            {
                [ForceUnroll] for (int e = 0; e < EPL; e++) st[e * LPR] = M[e];
            }
#endif
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
            if (part == 0 && r0 + tt >= replay) p.y[(uint64_t(tBeg + r0 + tt - replay) * H + h) * P + row] = yd + Dh * xp;
        }
    }
    if (!tentative)
    {
        [ForceUnroll] for (int e = 0; e < EPL; e++) st[e * LPR] = M[e];
    }
}
#endif
