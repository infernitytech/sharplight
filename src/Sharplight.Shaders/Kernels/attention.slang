// Causal attention of T queries against a KV cache (online softmax, one workgroup per (head, query)).
// q: [T, HQ, D] with strides (QS0, QS1); caches: [ctx, HKV, D]/[ctx, HKV, DV] of type KV_F16 ? f16 : f32;
// out: dense [T, HQ, DV]. Query t sees cache rows [0, n_past + t].
// Macros: HQ, HKV, D, DV, QS0, QS1, SCALE, KV_F16, WG (threads, = chunk of keys),
//         WINDOW (> 0: a query sees only its last WINDOW keys; sliding-window layers), HAS_SOFTCAP + SOFTCAP
//         (cap value: scores = SOFTCAP * tanh(scores / SOFTCAP)), HAS_SINKS (p.sinks[h]: per-head logit that only joins the softmax denominator).
// SPLITS > 1 (flash decoding, few queries): the keys of each (head, query) are split into SPLITS contiguous ranges,
// one workgroup each (grid z); every split writes an unnormalized partial (acc[DV], m, l) to part[split][t][h][DV+2]
// and the `combine` entry merges them. The split ranges follow n_past at run time, so a recorded command buffer
// stays valid as the context grows. The `decode` / `decode_combine` entries (split-K flash decoding with GQA groups and a
// run-time split count, below) are the preferred decode path; `main` with SPLITS is its fallback.
#include "common.slang"

#ifndef SPLITS
#define SPLITS 1
#endif
// SPANS (FULL_ATTN only): spans [T, 2] (first key row, count) per query row — packed independent sequences; with CAUSAL a
// query row t additionally sees only key rows <= t.
#ifndef SPANS
#define SPANS 0
#endif
#ifndef CAUSAL
#define CAUSAL 0
#endif

// ---------------------------------------------------------------------------------------------
// Split-K flash decoding (`decode` + `decode_combine`, few queries against a KV cache). A workgroup handles HPW query
// heads that share one KV head (GQA group), so K/V rows are read once per group, and one contiguous range of keys of one
// query row. The split count follows the key count at run time: a row is split into ranges of at least DEC_MIN_KEYS
// keys, and into at most decode_max_splits(T) ranges (the grid z of the launch), so short contexts run a single range
// and write the output directly, while long ones spread over many workgroups. Ranges beyond a row's split count exit.
// Multi-sequence: every row has its own context length. Partials: part[split][t][h][DEC_PS] (acc, m, l).
// Macros: HPW (divisor of HQ/HKV), KLANES (lanes sharing a key/value row, power of two <= SG), DEC_WG (threads), DEC_R
// (rows per lane group and batch), DEC_BC (keys per batch = DEC_R * DEC_WG / KLANES), DEC_MAX_SPLITS, DEC_MIN_KEYS
// (multiple of DEC_BC), DEC_TARGET_WG (workgroups the split count aims for).
#if defined(KERNEL_DECODE) || defined(KERNEL_DECODE_COMBINE)
#define DEC_GROUPS (HKV * (HQ / HKV / HPW))   // workgroups per (row, split)
#define DEC_PS (DV + 4)                        // floats per partial: acc[DV], m, l, padding (16-byte rows)

uint decode_max_splits(uint T)
{
    return clamp(uint(DEC_TARGET_WG) / max(T * DEC_GROUPS, 1u), 1u, uint(DEC_MAX_SPLITS));
}

// Keys per split range of a row with `span` keys (a multiple of DEC_BC).
uint decode_split_keys(uint span, uint T)
{
    uint ms = decode_max_splits(T);
    uint per = (span + ms - 1) / ms;
    per = (per + DEC_BC - 1) / DEC_BC * DEC_BC;
    return max(per, uint(DEC_MIN_KEYS));
}

// Key range [jLo, L) of query row t and the first cache row (x HKV) of its sequence.
void decode_range(uint t, uint* rt, int* seq, out uint jLo, out uint L, out uint64_t kvRow)
{
    kvRow = 0;
#if MULTI_SEQ
    L = uint(seq[t * 4 + 1]) + 1;
    kvRow = uint64_t(seq[t * 4]) * SLOT_CTX * HKV;
#else
    L = rt[RT_N_PAST] + t + 1;
#endif
#if WINDOW
    jLo = L > WINDOW ? L - WINDOW : 0;
#else
    jLo = 0;
#endif
}
#endif

#ifdef KERNEL_DECODE_COMBINE
struct DecodeCombineParams
{
    float* part;
    float* o;
    float* sinks;
    uint* rt;
#if MULTI_SEQ
    int* seq;
#endif
    uint T;
};

#define DC_WG 128
#define DC_D4 (DV / 4)                                  // float4 columns of a partial row
#define DC_SL (DC_WG / DC_D4 > 0 ? DC_WG / DC_D4 : 1)   // split lanes: thread (sl, c) reads splits sl, sl + DC_SL, ...
#define DC_K 16                                         // splits per thread and round (loads in flight)
groupshared float dcW[DEC_MAX_SPLITS];
groupshared float dcRed[DC_WG / SG];
groupshared float4 dcAcc[DC_SL * DC_D4];

float4 load4(float* ptr)
{
    return *(float4*)ptr;
}

// One workgroup per (head, row): merges the row's split partials; rows that ran a single range were written by `decode`.
// The split statistics and the partial rows are loaded in the same round trip; weights go through shared memory.
[shader("compute")]
[numthreads(DC_WG, 1, 1)]
void decode_combine(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform DecodeCombineParams p)
{
    uint h = gid.x, t = gid.y, tid = ltid.x;
    uint jLo, L;
    uint64_t kvRow;
#if MULTI_SEQ
    decode_range(t, p.rt, p.seq, jLo, L, kvRow);
#else
    decode_range(t, p.rt, nullptr, jLo, L, kvRow);
#endif
    uint span = L - jLo;
    uint per = decode_split_keys(span, p.T);
    uint ns = (span + per - 1) / per;
    if (ns <= 1)
    {
        return;
    }

    uint64_t rowStride = uint64_t(p.T) * HQ * DEC_PS;   // floats between the partials of consecutive splits
    float* base = p.part + (uint64_t(t) * HQ + h) * DEC_PS;
    uint sl = tid / DC_D4, c = tid % DC_D4;
    bool active = sl < DC_SL;

    // Statistics of split tid (tid < ns <= DEC_MAX_SPLITS, several per thread when DC_WG < DEC_MAX_SPLITS).
    float ms[(DEC_MAX_SPLITS + DC_WG - 1) / DC_WG], ls[(DEC_MAX_SPLITS + DC_WG - 1) / DC_WG];
    float M = -1e30;
    [ForceUnroll]
    for (uint k = 0; k < (DEC_MAX_SPLITS + DC_WG - 1) / DC_WG; k++)
    {
        uint s = tid + k * DC_WG;
        ms[k] = s < ns ? base[s * rowStride + DV] : -1e30;
        ls[k] = s < ns ? base[s * rowStride + DV + 1] : 0.0;
        M = max(M, ms[k]);
    }
    // First round of partial rows, issued before the statistics are reduced.
    float4 a[DC_K];
    [ForceUnroll]
    for (uint k = 0; k < DC_K; k++)
    {
        uint s = sl + k * DC_SL;
        a[k] = active && s < ns ? load4(base + s * rowStride + c * 4) : float4(0);
    }

    M = WaveActiveMax(M);
    if (tid % SG == 0) dcRed[tid / SG] = M;
    GroupMemoryBarrierWithGroupSync();
    M = dcRed[0];
    [ForceUnroll] for (uint i = 1; i < DC_WG / SG; i++) M = max(M, dcRed[i]);
#if HAS_SINKS
    M = max(M, p.sinks[h]);
#endif
    float den = 0;
    [ForceUnroll]
    for (uint k = 0; k < (DEC_MAX_SPLITS + DC_WG - 1) / DC_WG; k++)
    {
        uint s = tid + k * DC_WG;
        float w = s < ns ? exp(ms[k] - M) : 0.0;
        if (s < DEC_MAX_SPLITS) dcW[s] = w;
        den += w * ls[k];
    }
    den = WaveActiveSum(den);
    GroupMemoryBarrierWithGroupSync();   // dcW visible; everyone has read dcRed
    if (tid % SG == 0) dcRed[tid / SG] = den;
    GroupMemoryBarrierWithGroupSync();
    den = 0;
    [ForceUnroll] for (uint i = 0; i < DC_WG / SG; i++) den += dcRed[i];
#if HAS_SINKS
    den += exp(p.sinks[h] - M);
#endif

    float4 num = float4(0);
    [ForceUnroll] for (uint k = 0; k < DC_K; k++) num += (sl + k * DC_SL < ns ? dcW[sl + k * DC_SL] : 0.0) * a[k];
    for (uint s0 = DC_K * DC_SL; s0 < ns; s0 += DC_K * DC_SL)   // only for very wide rows (few split lanes)
    {
        [ForceUnroll]
        for (uint k = 0; k < DC_K; k++)
        {
            uint s = s0 + sl + k * DC_SL;
            if (active && s < ns) num += dcW[s] * load4(base + s * rowStride + c * 4);
        }
    }
    if (active)
    {
        dcAcc[sl * DC_D4 + c] = num;
    }
    GroupMemoryBarrierWithGroupSync();
    float inv = 1.0 / den;
    float* orow = p.o + (uint64_t(t) * HQ + h) * DV;
    for (uint i = tid; i < DC_D4; i += DC_WG)
    {
        float4 sum = float4(0);
        [ForceUnroll] for (uint k = 0; k < DC_SL; k++) sum += dcAcc[k * DC_D4 + i];
        *(float4*)(orow + i * 4) = sum * inv;
    }
}
#endif

#ifdef KERNEL_COMBINE

struct CombineParams
{
    float* part;
    float* o;
    float* sinks;
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
#if HAS_SINKS
    float sk = p.sinks[h];
    M = max(M, sk);
    den = exp(sk - M);
#endif
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
    float* sinks;
    uint* rt;
#if MULTI_SEQ
    int* seq;    // [T, 4] (slot, position, run length, run index) per query row
    int* runs;   // [U, 4] (slot, first row, row count, first position)
#endif
#if SPANS
    int* spans;  // [T, 2] (first key row, key count) per query row
#endif
    uint T;
    uint row0;   // first query row of this launch (long prefills are sliced into several launches)
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
    uint t = gid.y + p.row0;
    uint tid = ltid.x;
    uint hk = h / (HQ / HKV);
    uint64_t kvRow = 0;   // multi-sequence: first cache row (x HKV) of the query's slot
#if FULL_ATTN && SPANS
    uint spBeg = uint(p.spans[t * 2]);
    uint L = spBeg + uint(p.spans[t * 2 + 1]);
#if CAUSAL
    L = min(L, t + 1);
#endif
    L = max(L, spBeg);
#elif FULL_ATTN
    uint L = p.T;   // bidirectional: every query sees all T keys (K/V are the dense [T, HKV, D] tensors of this launch)
#elif MULTI_SEQ
    uint L = uint(p.seq[t * 4 + 1]) + 1;
    kvRow = uint64_t(p.seq[t * 4]) * SLOT_CTX * HKV;
#else
    uint L = p.rt[RT_N_PAST] + t + 1;
#endif
#if WINDOW && !FULL_ATTN
    uint jLo = L > WINDOW ? L - WINDOW : 0;
#elif FULL_ATTN && SPANS
    uint jLo = spBeg;
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
        if (j < jEnd) s = dot_k(p.kc, (kvRow + uint64_t(j) * HKV + hk) * D);
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
                    a += sP[jj] * load_kv(p.vc, (kvRow + uint64_t(c0 + jj) * HKV + hk) * DV + d);
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
#if HAS_SINKS
    {
        float sk = p.sinks[h], mf = max(m, sk), cf = exp(m - mf);
        l = l * cf + exp(sk - mf);
        [ForceUnroll] for (int di = 0; di < DPT; di++) acc[di] *= cf;
    }
#endif
    float inv = l > 0 ? 1.0 / l : 0.0;
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
// in FA_DC-wide chunks and accumulated into the registers. Causal: query t sees keys [0, n_past + t]; FULL_ATTN:
// every key of the launch (bidirectional); WINDOW: only the last WINDOW keys; HAS_SOFTCAP: capped scores.
// Head sizes are zero-padded in shared memory (D to 16, DV to 64), so D % 8 == DV % 8 == 0 suffices.
// K/V: f16 (KV_F16) or f32. Requires padded D, DV <= 256 and 32-wide subgroups with quad operations.
using namespace linalg;
#define FA_BR 32
#define FA_BC 32
#define FA_T 128
#define FA_DP ((D + 15) / 16 * 16)
#define FA_DVP ((DV + 63) / 64 * 64)
#define FA_KVD (FA_DP > FA_DVP ? FA_DP : FA_DVP)
#define FA_QS (FA_DP + 8)
#define FA_KS (FA_KVD + 8)
#define FA_SS FA_BC
#define FA_PS (FA_BC + 8)
#define FA_DC 64
#define FA_NC (FA_DVP / FA_DC)
typealias FaA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias FaB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias FaC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

groupshared half faQ[FA_BR * FA_QS];
groupshared half faKV[FA_BC * FA_KS];
groupshared float faS[FA_BR * FA_SS];
groupshared half faP[FA_BR * FA_PS];
groupshared float faO[FA_BR * FA_DC];

// Rows j0.. of K or V (dim columns, zero-padded to dimP) into faKV as f16; rows >= L are zero.
void fa_load_kv(uint8_t* cache, uint64_t kvRow, uint j0, uint L, uint hk, uint dim, uint dimP, uint tid)
{
    uint per = dimP / 8;
    for (uint i = tid; i < FA_BC * per; i += FA_T)
    {
        uint r = i / per, c = (i % per) * 8;
        uint j = j0 + r;
        uint o = r * FA_KS + c;
        if (j < L && c < dim)
        {
#if KV_F16
            uint4 u = *(uint4*)(cache + ((kvRow + uint64_t(j) * HKV + hk) * dim + c) * 2);
            faKV[o + 0] = half(f16tof32(u.x & 0xFFFF)); faKV[o + 1] = half(f16tof32(u.x >> 16));
            faKV[o + 2] = half(f16tof32(u.y & 0xFFFF)); faKV[o + 3] = half(f16tof32(u.y >> 16));
            faKV[o + 4] = half(f16tof32(u.z & 0xFFFF)); faKV[o + 5] = half(f16tof32(u.z >> 16));
            faKV[o + 6] = half(f16tof32(u.w & 0xFFFF)); faKV[o + 7] = half(f16tof32(u.w >> 16));
#else
            float4* f = (float4*)(cache + ((kvRow + uint64_t(j) * HKV + hk) * dim + c) * 4);
            float4 a = f[0], b = f[1];
            faKV[o + 0] = half(a.x); faKV[o + 1] = half(a.y); faKV[o + 2] = half(a.z); faKV[o + 3] = half(a.w);
            faKV[o + 4] = half(b.x); faKV[o + 5] = half(b.y); faKV[o + 6] = half(b.z); faKV[o + 7] = half(b.w);
#endif
        }
        else
        {
            [ForceUnroll] for (uint e = 0; e < 8; e++) faKV[o + e] = 0.0h;
        }
    }
}

[shader("compute")]
[numthreads(FA_T, 1, 1)]
void fa(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    uint h = gid.x, t0 = gid.y * FA_BR;
    uint tid = ltid.x, sg = tid / 32;
    uint hk = h / (HQ / HKV);
    uint64_t kvRow = 0;
    uint qT = p.T, qRow0 = 0;   // query rows of this sequence; first global row
#if FULL_ATTN
    uint nPast = 0;
#elif MULTI_SEQ
    // Blocks never straddle sequences: workgroup (h, blk, u) handles query tile blk of run u (grid z = runs,
    // y = tiles of the longest possible run); tiles past a run's end exit.
    uint u = gid.z, blk = gid.y;
    int4 run = int4(p.runs[u * 4], p.runs[u * 4 + 1], p.runs[u * 4 + 2], p.runs[u * 4 + 3]);
    if (blk * FA_BR >= uint(run.z)) return;
    t0 = blk * FA_BR; qT = uint(run.z); qRow0 = uint(run.y);
    uint nPast = uint(run.w);
    kvRow = uint64_t(run.x) * SLOT_CTX * HKV;
#else
    uint nPast = p.rt[RT_N_PAST];
#endif

    for (uint i = tid; i < FA_BR * FA_DP; i += FA_T)
    {
        uint r = i / FA_DP, d = i % FA_DP, t = t0 + r;
        faQ[r * FA_QS + d] = half(t < qT && d < D ? p.q[uint64_t(qRow0 + t) * QS0 + uint64_t(h) * QS1 + d] * SCALE : 0.0);
    }

    uint row = tid / 4, sub = tid % 4;
    uint tq = t0 + row;
    bool rowValid = tq < qT;
    float m = -1e30, l = 0;
    float o[FA_NC * 16];   // output dims c * FA_DC + sub * 16 + e
    [ForceUnroll] for (uint i = 0; i < FA_NC * 16; i++) o[i] = 0;
#if FULL_ATTN && SPANS
    // Packed sequences: the block's key range is the union of its rows' spans; each row masks to its own span.
    uint rowBeg = 0, rowEnd = 0, jMin = 0xFFFFFFFFu, jMax = 0;
    for (uint r = 0; r < FA_BR; r++)
    {
        uint t = t0 + r;
        if (t >= qT) break;
        uint b = uint(p.spans[t * 2]), e = b + uint(p.spans[t * 2 + 1]);
#if CAUSAL
        e = min(e, t + 1);
#endif
        if (r == row) { rowBeg = b; rowEnd = e; }
        if (e > b) { jMin = min(jMin, b); jMax = max(jMax, e); }
    }
    if (jMin >= jMax) { jMin = 0; jMax = 0; }
    uint L = jMax, jFirst = jMin / FA_BC * FA_BC;
#else
#if FULL_ATTN
    uint L = p.T, rowEnd = p.T;
#else
    uint L = nPast + min(qT, t0 + FA_BR);   // keys needed by the block's last query
    uint rowEnd = nPast + tq + 1;            // this row's causal end
#endif
#if WINDOW && !FULL_ATTN
    uint rowBeg = rowEnd > WINDOW ? rowEnd - WINDOW : 0;
    uint jFirst = nPast + t0 + 1 > WINDOW ? (nPast + t0 + 1 - WINDOW) / FA_BC * FA_BC : 0;   // first query's window
#else
    uint rowBeg = 0, jFirst = 0;
#endif
#endif
    GroupMemoryBarrierWithGroupSync();

    for (uint j0 = jFirst; j0 < L; j0 += FA_BC)
    {
        fa_load_kv(p.kc, kvRow, j0, L, hk, D, FA_DP, tid);
        GroupMemoryBarrierWithGroupSync();

        // S: subgroup sg computes rows (sg / 2) * 16, keys (sg % 2) * 16.
        {
            FaC sacc = FaC(0.0);
            uint qa = (sg / 2) * 16 * FA_QS, kb = (sg % 2) * 16 * FA_KS;
            [ForceUnroll]
            for (uint k = 0; k < FA_DP; k += 16)
            {
                FaA a = FaA.Load<CoopMatMatrixLayout.RowMajor>(faQ, qa + k, FA_QS);
                FaB b = FaB.Load<CoopMatMatrixLayout.ColumnMajor>(faKV, kb + k, FA_KS);
                sacc = coopMatMulAdd<float, false>(a, b, sacc);
            }
            sacc.Store<CoopMatMatrixLayout.RowMajor>(faS, (sg / 2) * 16 * FA_SS + (sg % 2) * 16, FA_SS);
        }
        GroupMemoryBarrierWithGroupSync();   // S visible; K no longer needed

        fa_load_kv(p.vc, kvRow, j0, L, hk, DV, FA_DVP, tid);

        // Online softmax of row `row`, columns sub * 8 .. + 8 (the row's 4 threads are adjacent lanes).
        float sv[8];
        float mx = -1e30;
        [ForceUnroll]
        for (uint e = 0; e < 8; e++)
        {
            uint c = sub * 8 + e, j = j0 + c;
            float sc = faS[row * FA_SS + c];
#if HAS_SOFTCAP
            sc = SOFTCAP * tanh(sc / SOFTCAP);
#endif
            sv[e] = rowValid && j < rowEnd && j >= rowBeg ? sc : -1e30;
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

#if HAS_SINKS
    {
        float sk = p.sinks[h], mf = max(m, sk), cf = exp(m - mf);
        l = l * cf + exp(sk - mf);
        [ForceUnroll] for (uint i = 0; i < FA_NC * 16; i++) o[i] *= cf;
    }
#endif
    if (rowValid)
    {
        float inv = l > 0 ? 1.0 / l : 0.0;
        float* orow = p.o + (uint64_t(qRow0 + tq) * HQ + h) * DV;
        [ForceUnroll]
        for (uint c = 0; c < FA_NC; c++)
            [ForceUnroll] for (uint e = 0; e < 16; e++) if (c * FA_DC + sub * 16 + e < DV) orow[c * FA_DC + sub * 16 + e] = o[c * 16 + e] * inv;
    }
}
#endif

#ifdef KERNEL_DECODE
// ---------------------------------------------------------------------------------------------
// Split-K flash decoding, see decode_range above. KLANES adjacent lanes own one key row at a time (16-byte loads of
// 8 dims each, butterfly sum of the HPW scores) and the same lanes the matching value row, so a lane group runs its own
// online softmax over its stream of keys with no shared memory or barriers in the key loop: per batch it takes DEC_R
// rows (all loads issued first), rescales once and accumulates P·V for all HPW heads in registers. The lane groups'
// (m, l, acc) states are merged at the end, inside the subgroup by shuffles and across subgroups in shared memory.
#define DEC_KC ((D / 8 + KLANES - 1) / KLANES)    // 8-dim slices of a key row per lane
#define DEC_VC ((DV / 8 + KLANES - 1) / KLANES)   // 8-dim slices of a value row per lane
#define DEC_NRG (DEC_WG / KLANES)                 // lane groups (key streams) per workgroup
#define DEC_NSG (DEC_WG / SG)                     // subgroups per workgroup

groupshared float dM[DEC_NSG * HPW];
groupshared float dL[DEC_NSG * HPW];
groupshared float dO[DEC_NSG * HPW * DEC_VC * KLANES * 8];

// 8 consecutive cache elements, kept packed in registers until used.
struct Kv8
{
#if KV_F16
    uint4 u;
#else
    float4 a;
    float4 b;
#endif
};

Kv8 load_kv8(uint8_t* cache, uint64_t elem)
{
    Kv8 r;
#if KV_F16
    r.u = *(uint4*)(cache + elem * 2);
#else
    float4* f = (float4*)(cache + elem * 4);
    r.a = f[0];
    r.b = f[1];
#endif
    return r;
}

Kv8 zero_kv8()
{
    Kv8 r;
#if KV_F16
    r.u = uint4(0);
#else
    r.a = float4(0);
    r.b = float4(0);
#endif
    return r;
}

void unpack_kv8(Kv8 k, out float v[8])
{
#if KV_F16
    v[0] = f16tof32(k.u.x & 0xFFFF); v[1] = f16tof32(k.u.x >> 16);
    v[2] = f16tof32(k.u.y & 0xFFFF); v[3] = f16tof32(k.u.y >> 16);
    v[4] = f16tof32(k.u.z & 0xFFFF); v[5] = f16tof32(k.u.z >> 16);
    v[6] = f16tof32(k.u.w & 0xFFFF); v[7] = f16tof32(k.u.w >> 16);
#else
    v[0] = k.a.x; v[1] = k.a.y; v[2] = k.a.z; v[3] = k.a.w;
    v[4] = k.b.x; v[5] = k.b.y; v[6] = k.b.z; v[7] = k.b.w;
#endif
}

[shader("compute")]
[numthreads(DEC_WG, 1, 1)]
void decode(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    uint hk = gid.x / (HQ / HKV / HPW);
    uint h0 = hk * (HQ / HKV) + gid.x % (HQ / HKV / HPW) * HPW;
    uint t = gid.y;
    uint tid = ltid.x;
    uint kl = tid % KLANES, rg = tid / KLANES;

    // Queries first: their loads do not depend on the key range.
    float q[HPW][DEC_KC * 8];
    [ForceUnroll]
    for (uint g = 0; g < HPW; g++)
    {
        float* qr = p.q + uint64_t(t) * QS0 + uint64_t(h0 + g) * QS1;
        [ForceUnroll]
        for (uint c = 0; c < DEC_KC; c++)
        {
            uint d0 = (c * KLANES + kl) * 8;
            [ForceUnroll] for (uint e = 0; e < 8; e++) q[g][c * 8 + e] = d0 < D ? qr[d0 + e] * SCALE : 0.0;
        }
    }

    uint jLo, L;
    uint64_t kvRow;
#if MULTI_SEQ
    decode_range(t, p.rt, p.seq, jLo, L, kvRow);
#else
    decode_range(t, p.rt, nullptr, jLo, L, kvRow);
#endif
    uint span = L - jLo;
    uint per = decode_split_keys(span, p.T);
    uint ns = (span + per - 1) / per;
    uint split = gid.z;
    if (split >= ns)
    {
        return;
    }

    uint jBeg = jLo + split * per;
    uint jEnd = min(L, jBeg + per);

    float m[HPW], l[HPW];
    float acc[HPW][DEC_VC * 8];
    [ForceUnroll]
    for (uint g = 0; g < HPW; g++)
    {
        m[g] = -1e30;
        l[g] = 0;
        [ForceUnroll] for (uint i = 0; i < DEC_VC * 8; i++) acc[g][i] = 0;
    }

    // Batch of DEC_BC keys: lane group rg takes rows c0 + i * DEC_NRG + rg (i < DEC_R), so neighbouring groups read
    // neighbouring rows.
    for (uint c0 = jBeg; c0 < jEnd; c0 += DEC_BC)
    {
        Kv8 kb[DEC_R][DEC_KC];
        Kv8 vb[DEC_R][DEC_VC];
        [ForceUnroll]
        for (uint i = 0; i < DEC_R; i++)
        {
            uint j = c0 + i * DEC_NRG + rg;
            uint64_t row = kvRow + uint64_t(j) * HKV + hk;
            [ForceUnroll]
            for (uint c = 0; c < DEC_KC; c++)
            {
                uint d0 = (c * KLANES + kl) * 8;
                kb[i][c] = j < jEnd && d0 < D ? load_kv8(p.kc, row * D + d0) : zero_kv8();
            }
            [ForceUnroll]
            for (uint c = 0; c < DEC_VC; c++)
            {
                uint d0 = (c * KLANES + kl) * 8;
                vb[i][c] = j < jEnd && d0 < DV ? load_kv8(p.vc, row * DV + d0) : zero_kv8();
            }
        }

        float s[DEC_R][HPW];
        [ForceUnroll]
        for (uint i = 0; i < DEC_R; i++)
        {
            [ForceUnroll] for (uint g = 0; g < HPW; g++) s[i][g] = 0;
            [ForceUnroll]
            for (uint c = 0; c < DEC_KC; c++)
            {
                float k[8];
                unpack_kv8(kb[i][c], k);
                [ForceUnroll]
                for (uint g = 0; g < HPW; g++)
                    [ForceUnroll] for (uint e = 0; e < 8; e++) s[i][g] += q[g][c * 8 + e] * k[e];
            }
        }
        [ForceUnroll]
        for (uint off = KLANES / 2; off > 0; off >>= 1)
            [ForceUnroll] for (uint i = 0; i < DEC_R; i++) [ForceUnroll] for (uint g = 0; g < HPW; g++)
                s[i][g] += WaveReadLaneAt(s[i][g], WaveGetLaneIndex() ^ off);

        [ForceUnroll]
        for (uint g = 0; g < HPW; g++)
        {
            float mNew = m[g];
            [ForceUnroll]
            for (uint i = 0; i < DEC_R; i++)
            {
#if HAS_SOFTCAP
                s[i][g] = SOFTCAP * tanh(s[i][g] / SOFTCAP);
#endif
                if (c0 + i * DEC_NRG + rg < jEnd) mNew = max(mNew, s[i][g]);
            }
            float corr = exp(m[g] - mNew);
            m[g] = mNew;
            l[g] *= corr;
            [ForceUnroll] for (uint e = 0; e < DEC_VC * 8; e++) acc[g][e] *= corr;
            [ForceUnroll]
            for (uint i = 0; i < DEC_R; i++)
            {
                float pv = c0 + i * DEC_NRG + rg < jEnd ? exp(s[i][g] - mNew) : 0.0;
                l[g] += pv;
                [ForceUnroll]
                for (uint c = 0; c < DEC_VC; c++)
                {
                    float v[8];
                    unpack_kv8(vb[i][c], v);
                    [ForceUnroll] for (uint e = 0; e < 8; e++) acc[g][c * 8 + e] += pv * v[e];
                }
            }
        }
    }

    // Merge the lane groups of the subgroup (lanes with equal kl), then the subgroups through shared memory.
    [ForceUnroll]
    for (uint off = KLANES; off < SG; off <<= 1)
    {
        [ForceUnroll]
        for (uint g = 0; g < HPW; g++)
        {
            float mo = WaveReadLaneAt(m[g], WaveGetLaneIndex() ^ off);
            float lo = WaveReadLaneAt(l[g], WaveGetLaneIndex() ^ off);
            float mNew = max(m[g], mo);
            float ca = exp(m[g] - mNew), cb = exp(mo - mNew);
            m[g] = mNew;
            l[g] = l[g] * ca + lo * cb;
            [ForceUnroll]
            for (uint e = 0; e < DEC_VC * 8; e++)
            {
                float ao = WaveReadLaneAt(acc[g][e], WaveGetLaneIndex() ^ off);
                acc[g][e] = acc[g][e] * ca + ao * cb;
            }
        }
    }
    uint sg = tid / SG;
    if (tid % SG < KLANES)
    {
        [ForceUnroll]
        for (uint g = 0; g < HPW; g++)
        {
            if (kl == 0)
            {
                dM[sg * HPW + g] = m[g];
                dL[sg * HPW + g] = l[g];
            }
            [ForceUnroll]
            for (uint c = 0; c < DEC_VC; c++)
                [ForceUnroll] for (uint e = 0; e < 8; e++) dO[((sg * HPW + g) * DEC_VC * KLANES + c * KLANES + kl) * 8 + e] = acc[g][c * 8 + e];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Output element i = (head g, dim d): weights of the subgroups' states, then the merged (normalized) value.
    bool direct = ns == 1;
    for (uint i = tid; i < HPW * DV; i += DEC_WG)
    {
        uint g = i / DV, d = i % DV;
        uint h = h0 + g;
        float M = -1e30;
        [ForceUnroll] for (uint w = 0; w < DEC_NSG; w++) M = max(M, dM[w * HPW + g]);
        float lsum = 0, num = 0;
        [ForceUnroll]
        for (uint w = 0; w < DEC_NSG; w++)
        {
            float wt = exp(dM[w * HPW + g] - M);
            lsum += wt * dL[w * HPW + g];
            num += wt * dO[(w * HPW + g) * DEC_VC * KLANES * 8 + d];
        }
        if (direct)
        {
#if HAS_SINKS
            float sk = p.sinks[h], mf = max(M, sk), cf = exp(M - mf);
            lsum = lsum * cf + exp(sk - mf);
            num *= cf;
#endif
            p.o[(uint64_t(t) * HQ + h) * DV + d] = lsum > 0 ? num / lsum : 0.0;
        }
        else
        {
            float* dst = p.part + ((uint64_t(split) * p.T + t) * HQ + h) * DEC_PS;
            dst[d] = num;
            if (d == 0)
            {
                dst[DV] = M;
                dst[DV + 1] = lsum;
            }
        }
    }
}
#endif

#endif
