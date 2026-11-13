// RWKV-6 / RWKV-7 recurrences (Wkv6 / Wkv7 ops). One workgroup per (head h, block of RB rows of the head's N x N state matrix S[i][j], run);
// row i (a value channel) is owned by LPR lanes of one subgroup, each holding N/LPR key channels in registers (lanes interleaved: conflict-free shared
// reads); the per-row dot products are subgroup butterfly sums (LPR = 1 needs no subgroup ops at all). Tokens are staged TB at a time in shared memory.
//   Wkv7: sa = a . S_i;  S_i = S_i * w + v_i k + sa b;  y_i = S_i . r
//   Wkv6: y_i = sum_j r_j (u_j k_j v_i + S_ij);  S_i = S_i * w + k v_i
// All token inputs are dense [T, H, N] f32; u [H, N]; state [H, N, N] (MULTI_SEQ: [S, H, N, N], grid z = run of the [U, 4] runs table,
// a run at position 0 starts from zero state); y dense [T, H, N].
// Macros: H, N, LPR, RB (N % RB == 0), TB, MULTI_SEQ
#include "common.slang"

#if defined(KERNEL_WKV7) || defined(KERNEL_WKV6)
struct WkvParams
{
    float* r;
    float* w;
    float* k;
    float* v;
    float* a;      // Wkv7: a; Wkv6: u [H, N]
    float* b;      // Wkv7 only
    float* state;
    float* y;
#if MULTI_SEQ
    int* runs;
#endif
    uint T;
};

#define EPL (N / LPR)
#define NT (RB * LPR)

groupshared float sR[TB * N];
groupshared float sW[TB * N];
groupshared float sK[TB * N];
groupshared float sA[TB * N];
groupshared float sB[TB * N];
groupshared float sV[TB * RB];

float row_sum(float v)
{
    [ForceUnroll]
    for (uint off = 1; off < LPR; off <<= 1)
        v += WaveReadLaneAt(v, WaveGetLaneIndex() ^ off);
    return v;
}

// Stages tokens [t0, t0 + nt) of head h (all N channels of r, w, k and, for Wkv7, a and b; RB value channels of v) in shared memory.
void stage(WkvParams p, uint h, uint rowBlock, uint t0, uint nt, uint tid)
{
    for (uint i = tid; i < nt * N; i += NT)
    {
        uint tt = i / N, c = i % N;
        uint64_t at = (uint64_t(t0 + tt) * H + h) * N + c;
        sR[i] = p.r[at];
        sW[i] = p.w[at];
        sK[i] = p.k[at];
#ifdef KERNEL_WKV7
        sA[i] = p.a[at];
        sB[i] = p.b[at];
#endif
    }
    for (uint i = tid; i < nt * RB; i += NT)
    {
        uint tt = i / RB, c = i % RB;
        sV[i] = p.v[(uint64_t(t0 + tt) * H + h) * N + rowBlock * RB + c];
    }
}

#ifdef KERNEL_WKV7
[shader("compute")]
[numthreads(NT, 1, 1)]
void wkv7(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform WkvParams p)
#else
[shader("compute")]
[numthreads(NT, 1, 1)]
void wkv6(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform WkvParams p)
#endif
{
    uint h = gid.x;
    uint rl = ltid.x / LPR;
    uint row = gid.y * RB + rl;
    uint part = ltid.x % LPR;
#if MULTI_SEQ
    int4 run = int4(p.runs[gid.z * 4], p.runs[gid.z * 4 + 1], p.runs[gid.z * 4 + 2], p.runs[gid.z * 4 + 3]);
    float* st = p.state + (((uint64_t(run.x) * H + h) * N) + row) * N + part;
    uint tBeg = uint(run.y), tEnd = uint(run.y + run.z);
    bool fresh = run.w == 0;
#else
    float* st = p.state + (uint64_t(h) * N + row) * N + part;
    uint tBeg = 0, tEnd = p.T;
    bool fresh = false;
#endif
    float M[EPL];
    [ForceUnroll] for (int e = 0; e < EPL; e++) M[e] = fresh ? 0.0 : st[e * LPR];
#ifdef KERNEL_WKV6
    float U[EPL];
    [ForceUnroll] for (int e = 0; e < EPL; e++) U[e] = p.a[uint64_t(h) * N + part + e * LPR];
#endif

    for (uint t0 = tBeg; t0 < tEnd; t0 += TB)
    {
        uint nt = min(uint(TB), tEnd - t0);
        GroupMemoryBarrierWithGroupSync();
        stage(p, h, gid.y, t0, nt, ltid.x);
        GroupMemoryBarrierWithGroupSync();
        for (uint tt = 0; tt < nt; tt++)
        {
            float vi = sV[tt * RB + rl];
            uint base = tt * N + part;
            float ya[2] = { 0, 0 };
#ifdef KERNEL_WKV7
            float sp[2] = { 0, 0 };
            [ForceUnroll] for (int e = 0; e < EPL; e++) sp[e % 2] += sA[base + e * LPR] * M[e];
            float sa = row_sum(sp[0] + sp[1]);
            [ForceUnroll] for (int e = 0; e < EPL; e++)
            {
                M[e] = M[e] * sW[base + e * LPR] + vi * sK[base + e * LPR] + sa * sB[base + e * LPR];
                ya[e % 2] += M[e] * sR[base + e * LPR];
            }
#else
            [ForceUnroll] for (int e = 0; e < EPL; e++)
            {
                float kv = sK[base + e * LPR] * vi;
                ya[e % 2] += sR[base + e * LPR] * (U[e] * kv + M[e]);
                M[e] = M[e] * sW[base + e * LPR] + kv;
            }
#endif
            float y = row_sum(ya[0] + ya[1]);
            if (part == 0) p.y[(uint64_t(t0 + tt) * H + h) * N + row] = y;
        }
    }
    [ForceUnroll] for (int e = 0; e < EPL; e++) st[e * LPR] = M[e];
}
#endif
