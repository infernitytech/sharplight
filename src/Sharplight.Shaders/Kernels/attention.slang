// Causal attention of T queries against a KV cache (online softmax, one workgroup per (head, query)).
// q: [T, HQ, D] with strides (QS0, QS1); caches: [ctx, HKV, D]/[ctx, HKV, DV] in format KV_FMT (FMT_F32, FMT_F16, or the
// GGUF block formats FMT_Q8_0 / FMT_Q4_0 with 32-value blocks; quantized rows are a multiple of 32 values);
// out: dense [T, HQ, DV]. Query t sees cache rows [0, n_past + t].
// Macros: HQ, HKV, D, DV, VRS (value cache row stride in elements, >= DV: the values are the first DV entries of each row,
//         e.g. the latent prefix of a multi-head latent attention cache whose K and V are one buffer), QS0, QS1, SCALE, KV_FMT
//         (KV_F16: KV_FMT == FMT_F16), WG (threads, = chunk of keys),
//         WINDOW (> 0: a query sees only keys less than WINDOW positions away: its last WINDOW keys when causal, the symmetric
//         window of bidirectional encoders with FULL_ATTN), HAS_ALIBI + ALIBI_SLOPES (initializer list of HQ floats: scores get
//         -slope[h] * |key position - query position|),
//         CHUNK (> 0: a query at position p sees only keys [p - p % CHUNK, p]; Llama 4 chunked attention), HAS_SOFTCAP + SOFTCAP
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
#ifndef VRS
#define VRS DV
#endif
// SPANS (FULL_ATTN only): spans [T, 2] (first key row, count) per query row — packed independent sequences; with CAUSAL a
// query row t additionally sees only key rows <= t.
#ifndef SPANS
#define SPANS 0
#endif
#ifndef CAUSAL
#define CAUSAL 0
#endif
#ifndef KV_FMT
#define KV_FMT (KV_F16 ? FMT_F16 : FMT_F32)
#endif
#ifndef HAS_ALIBI
#define HAS_ALIBI 0
#endif
#if HAS_ALIBI
static const float kAlibiSlopes[HQ] = ALIBI_SLOPES;

// ALiBi score bias of head h for key position j and query position qp.
float alibi_bias(uint h, uint j, uint qp)
{
    return -kAlibiSlopes[h] * abs(float(j) - float(qp));
}
#endif
#define KV_QUANT (KV_FMT == FMT_Q8_0 || KV_FMT == FMT_Q4_0)
#define KV_BLOCK_BYTES (KV_FMT == FMT_Q8_0 ? 34 : 18)

// ---- KV cache element access (elements are addressed linearly in the cache; quantized caches in 32-value blocks) -------
// Block of cache element `elem`, its f16 scale, and the 16-bit word holding the byte at offset `b` of its quants (blocks are
// 2-byte aligned: 34 / 18 bytes).
uint8_t* kv_block(uint8_t* base, uint64_t elem) { return byteAt(base, (elem >> 5) * KV_BLOCK_BYTES); }
uint kv_word(uint8_t* blk, uint b) { return uint(*(uint16_t*)(blk + 2 + b)); }

// 8 cache values starting at element `elem` (a multiple of 8) as raw data: kept packed in registers until used, so a kernel
// can issue many loads before the first one is consumed.
struct Kv8
{
#if KV_FMT == FMT_F16
    uint4 u;
#elif KV_FMT == FMT_F32
    float4 a;
    float4 b;
#else
    float d;     // block scale
    uint2 q;     // q8_0: the 8 int8 quants; q4_0: the 8 bytes holding them (nibble at `shift`)
    uint shift;
#endif
};

Kv8 load_kv8(uint8_t* cache, uint64_t elem)
{
    Kv8 r;
#if KV_FMT == FMT_F16
    r.u = *(uint4*)byteAt(cache, elem * 2);
#elif KV_FMT == FMT_F32
    float4* f = (float4*)byteAt(cache, elem * 4);
    r.a = f[0];
    r.b = f[1];
#else
    uint8_t* blk = kv_block(cache, elem);
    uint within = uint(elem & 31);
    uint b = KV_FMT == FMT_Q8_0 ? within : within & 15;
    r.d = h2f(*(uint16_t*)blk);
    r.q = uint2(kv_word(blk, b) | (kv_word(blk, b + 2) << 16), kv_word(blk, b + 4) | (kv_word(blk, b + 6) << 16));
    r.shift = KV_FMT == FMT_Q4_0 && within >= 16 ? 4 : 0;
#endif
    return r;
}

Kv8 zero_kv8()
{
    Kv8 r;
#if KV_FMT == FMT_F16
    r.u = uint4(0);
#elif KV_FMT == FMT_F32
    r.a = float4(0);
    r.b = float4(0);
#else
    r.d = 0;
    r.q = uint2(0);
    r.shift = 0;
#endif
    return r;
}

#if KV_QUANT
// Quant e (0..7) of a packed Kv8 as a signed integer value (before the scale).
float kv_quant(Kv8 k, uint e)
{
    uint w = e < 4 ? k.q.x : k.q.y;
    uint byte = (w >> ((e & 3) * 8)) & 0xFF;
#if KV_FMT == FMT_Q8_0
    return float(int(byte << 24) >> 24);
#else
    return float((byte >> k.shift) & 0xF) - 8.0;
#endif
}
#endif

void unpack_kv8(Kv8 k, out float v[8])
{
#if KV_FMT == FMT_F16
    v[0] = f16tof32(k.u.x & 0xFFFF); v[1] = f16tof32(k.u.x >> 16);
    v[2] = f16tof32(k.u.y & 0xFFFF); v[3] = f16tof32(k.u.y >> 16);
    v[4] = f16tof32(k.u.z & 0xFFFF); v[5] = f16tof32(k.u.z >> 16);
    v[6] = f16tof32(k.u.w & 0xFFFF); v[7] = f16tof32(k.u.w >> 16);
#elif KV_FMT == FMT_F32
    v[0] = k.a.x; v[1] = k.a.y; v[2] = k.a.z; v[3] = k.a.w;
    v[4] = k.b.x; v[5] = k.b.y; v[6] = k.b.z; v[7] = k.b.w;
#else
    [ForceUnroll] for (uint e = 0; e < 8; e++) v[e] = k.d * kv_quant(k, e);
#endif
}

// 8 values as packed f16 (shared-memory tiles of the tensor-core kernels).
uint4 pack_half8(float v[8])
{
    return uint4(f32tof16(v[0]) | (f32tof16(v[1]) << 16), f32tof16(v[2]) | (f32tof16(v[3]) << 16),
                 f32tof16(v[4]) | (f32tof16(v[5]) << 16), f32tof16(v[6]) | (f32tof16(v[7]) << 16));
}

uint4 kv8_to_half(Kv8 k)
{
#if KV_FMT == FMT_F16
    return k.u;
#else
    float v[8];
    unpack_kv8(k, v);
    return pack_half8(v);
#endif
}

// One cache element.
float load_kv(uint8_t* base, uint64_t elem)
{
#if KV_FMT == FMT_F16
    return h2f(*(uint16_t*)byteAt(base, elem * 2));
#elif KV_FMT == FMT_F32
    return *(float*)byteAt(base, elem * 4);
#else
    uint8_t* blk = kv_block(base, elem);
    uint within = uint(elem & 31);
    uint b = KV_FMT == FMT_Q8_0 ? within : within & 15;
    uint byte = (kv_word(blk, b & ~1u) >> ((b & 1) * 8)) & 0xFF;
#if KV_FMT == FMT_Q8_0
    return h2f(*(uint16_t*)blk) * float(int(byte << 24) >> 24);
#else
    return h2f(*(uint16_t*)blk) * (float((byte >> (within >= 16 ? 4 : 0)) & 0xF) - 8.0);
#endif
#endif
}

// ---- Cache rows of a sequence --------------------------------------------------------------------------------------
// `kv_base(slot)` once per query row, then `kv_row(base, j)` = cache row of key j times HKV (add the KV head). Contiguous
// caches: base = first row (x HKV) of the slot (0 for single-sequence caches). Paged caches: base = the slot, and every key
// goes through the block table (g_pages, set from the parameters at kernel entry).
#if PAGED
static int* g_pages;
#endif

uint64_t kv_base(uint slot)
{
#if PAGED
    return slot;
#else
    return uint64_t(slot) * SLOT_CTX * HKV;
#endif
}

uint64_t kv_row(uint64_t base, uint j)
{
#if PAGED
    return seq_row(g_pages, uint(base), j) * HKV;
#else
    return base + uint64_t(j) * HKV;
#endif
}

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
#if defined(KERNEL_DECODE) || defined(KERNEL_DECODE_COMBINE) || defined(KERNEL_FA_MQA)
#ifndef DEC_GROUPS
#define DEC_GROUPS (HKV * (HQ / HKV / HPW))   // workgroups per (row, split)
#endif
#define DEC_PS (DV + 4)                        // floats per partial: acc[DV], m, l, padding (16-byte rows)

uint decode_max_splits(uint T)
{
#ifdef DEC_SPLIT_ROWS
    if (T > DEC_SPLIT_ROWS)
    {
        return 1;
    }
#endif
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

// Key range [jLo, L) of query row t and the cache base of its sequence (see kv_base).
void decode_range(uint t, uint* rt, int* seq, out uint jLo, out uint L, out uint64_t kvRow)
{
    kvRow = 0;
#if MULTI_SEQ
    L = uint(seq[t * 4 + 1]) + 1;
    kvRow = kv_base(uint(seq[t * 4]));
#else
    L = rt[RT_N_PAST] + t + 1;
#endif
#if WINDOW
    jLo = L > WINDOW ? L - WINDOW : 0;
#else
    jLo = 0;
#endif
#if CHUNK && !FULL_ATTN
    jLo = max(jLo, (L - 1) / CHUNK * CHUNK);   // chunked attention: only the keys of the query's own chunk
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
#if PAGED
    int* pages;  // [slots, PAGE_MB] block table of the paged cache
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

float dot_k(uint8_t* kc, uint64_t rowElem)
{
    float s = 0;
    [ForceUnroll]
    for (uint i = 0; i < D / 8; i++)
    {
        float k[8];
        unpack_kv8(load_kv8(kc, rowElem + i * 8), k);
        [ForceUnroll] for (uint e = 0; e < 8; e++) s += sQ[8 * i + e] * k[e];
    }
    return s;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
#if PAGED
    g_pages = p.pages;
#endif
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
    kvRow = kv_base(uint(p.seq[t * 4]));
#else
    uint L = p.rt[RT_N_PAST] + t + 1;
#endif
#if FULL_ATTN
    uint qPos = t;       // self-attention: query row t is key row t
#else
    uint qPos = L - 1;   // the query sits at the last visible key
#endif
#if WINDOW && !FULL_ATTN
    uint jLo = L > WINDOW ? L - WINDOW : 0;
#elif FULL_ATTN && SPANS
    uint jLo = spBeg;
#else
    uint jLo = 0;
#endif
#if WINDOW && FULL_ATTN
    jLo = max(jLo, qPos + 1 > WINDOW ? qPos + 1 - WINDOW : 0);
    L = max(jLo, min(L, qPos + WINDOW));
#endif
#if CHUNK && !FULL_ATTN
    jLo = max(jLo, (L - 1) / CHUNK * CHUNK);   // chunked attention: only the keys of the query's own chunk
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
        if (j < jEnd) s = dot_k(p.kc, (kv_row(kvRow, j) + hk) * D);
#if HAS_SOFTCAP
        if (j < jEnd) s = SOFTCAP * tanh(s / SOFTCAP);
#endif
#if HAS_ALIBI
        if (j < jEnd) s += alibi_bias(h, j, qPos);
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
                    a += sP[jj] * load_kv(p.vc, (kv_row(kvRow, c0 + jj) + hk) * VRS + d);
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
// every key of the launch (bidirectional); WINDOW: only the last WINDOW keys (FULL_ATTN: the symmetric window); HAS_SOFTCAP: capped
// scores; HAS_ALIBI: distance biases.
// Head sizes are zero-padded in shared memory (D to 16, DV to 64), so D % 8 == DV % 8 == 0 suffices.
// K/V: any KV_FMT (converted to f16 in shared memory). Requires padded D, DV <= 256 and 32-wide subgroups with quad operations.
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

// Rows j0.. of K or V (dim columns of rows rowStride wide, zero-padded to dimP) into faKV as f16; rows >= L are zero.
void fa_load_kv(uint8_t* cache, uint64_t kvRow, uint j0, uint L, uint hk, uint dim, uint rowStride, uint dimP, uint tid)
{
    uint per = dimP / 8;
    for (uint i = tid; i < FA_BC * per; i += FA_T)
    {
        uint r = i / per, c = (i % per) * 8;
        uint j = j0 + r;
        uint o = r * FA_KS + c;
        if (j < L && c < dim)
        {
            float v[8];
            unpack_kv8(load_kv8(cache, (kv_row(kvRow, j) + hk) * rowStride + c), v);
            [ForceUnroll] for (uint e = 0; e < 8; e++) faKV[o + e] = half(v[e]);
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
#if PAGED
    g_pages = p.pages;
#endif
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
    kvRow = kv_base(uint(run.x));
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
#if HAS_ALIBI
    uint faQueryPos = nPast + tq;   // FULL_ATTN: nPast = 0, the row itself
#endif
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
#if WINDOW
        b = max(b, t + 1 > WINDOW ? t + 1 - WINDOW : 0);
        e = min(e, t + WINDOW);
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
#elif WINDOW
    uint rowBeg = tq + 1 > WINDOW ? tq + 1 - WINDOW : 0, jFirst = 0;   // symmetric window around this row
    rowEnd = min(rowEnd, tq + WINDOW);
#else
    uint rowBeg = 0, jFirst = 0;
#endif
#if CHUNK && !FULL_ATTN
    rowBeg = max(rowBeg, rowEnd > 0 ? (rowEnd - 1) / CHUNK * CHUNK : 0);       // chunked attention: this row's chunk
    jFirst = max(jFirst, (nPast + t0) / CHUNK * CHUNK / FA_BC * FA_BC);           // the block's first query's chunk
#endif
#endif
    GroupMemoryBarrierWithGroupSync();

    for (uint j0 = jFirst; j0 < L; j0 += FA_BC)
    {
        fa_load_kv(p.kc, kvRow, j0, L, hk, D, D, FA_DP, tid);
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

        fa_load_kv(p.vc, kvRow, j0, L, hk, DV, VRS, FA_DVP, tid);

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
#if HAS_ALIBI
            sc += alibi_bias(h, j, faQueryPos);
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

#ifdef KERNEL_FA_MQA
// ---------------------------------------------------------------------------------------------
// Flash attention on tensor cores for many query heads per key head whose values are the first DV entries of the key rows
// (VRS == D): the absorbed multi-head latent attention cache (D = latent + rope = 576 / 288, DV = latent = 512 / 256,
// HKV = 1). A workgroup handles FM_BR heads of one query row (a tile of its key head's group, FM_TILES tiles per group)
// and one key range, so each cache row is loaded once per tile and serves as key and value. Prefill and decode use the
// same kernel: a row's keys are split at run time like `decode` (decode_split_keys: long batches run one range per row and
// write the output directly, few rows spread their keys over up to decode_max_splits(T) workgroups and leave partials for
// `decode_combine`).
// Per FM_BC-key tile: cache rows (fetched into registers during the previous tile) into shared memory (f16), S = Q·Kᵀ split along D over the subgroups (the query fragments
// stay in registers), online softmax in scalar code, then O += P·V with the output tiles in cooperative-matrix
// accumulators, rescaled per row by an element-wise product with a matrix of the row corrections.
using namespace linalg;
#define FM_BR 16
#define FM_BC 32
#define FM_T 128
#define FM_NSG (FM_T / 32)
#define FM_G (HQ / HKV)
#define FM_TILES ((FM_G + FM_BR - 1) / FM_BR)
#define FM_DP ((D + 15) / 16 * 16)
#define FM_KS (FM_DP + 8)
#define FM_KCH (FM_DP / 16)
#define FM_KPS ((FM_KCH + FM_NSG - 1) / FM_NSG)   // 16-wide D chunks of the scores per subgroup
#define FM_OT ((DV + 15) / 16)                      // 16-wide output tiles
#define FM_OPS ((FM_OT + FM_NSG - 1) / FM_NSG)      // output tiles per subgroup
#define FM_PS (FM_BC + 8)
#define FM_SS (FM_BR * FM_BC)                       // floats of one subgroup's partial scores (and staging slice)
typealias FmA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias FmB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias FmC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

groupshared uint4 fmKV[FM_BC * FM_KS / 8];   // key/value tile as packed f16 x 8; holds the query tile before the key loop
groupshared float fmS[FM_NSG * FM_SS];   // per-subgroup partial scores, then per-subgroup staging of output tiles
groupshared half fmP[FM_BR * FM_PS];
groupshared float fmCorr[FM_BR];
groupshared float fmCorrM[FM_BR * 16];   // per-tile correction factor of row r in every column of row r

#define FM_CPR (FM_DP / 8)                              // 8-value chunks per tile row
#define FM_LPT ((FM_BC * FM_CPR + FM_T - 1) / FM_T)      // chunks per thread and tile
#define FM_QPT ((FM_BR * FM_CPR + FM_T - 1) / FM_T)      // 8-value query chunks per thread

// One tile of cache rows in registers between fm_fetch and fm_store, so the next tile is loaded while the current one is
// processed. f16 / f32: 8-value chunks. q8_0 / q4_0: whole 32-value blocks, each read as the 32-bit words covering it from
// the aligned word at or before its start (blocks are only 2-byte aligned; `mis` bit k marks unit k as starting at 2 mod 4),
// which takes far fewer load instructions than 8-value chunks.
#if KV_QUANT
#define FM_NW ((KV_BLOCK_BYTES + 2) / 4)                // words per block
#define FM_BPR (D / 32)                                 // blocks per row
#define FM_UPT ((FM_BC * FM_BPR + FM_T - 1) / FM_T)     // blocks per thread and tile
struct FmTile
{
    uint w[FM_UPT * FM_NW];
    uint mis;
};
#else
struct FmTile
{
    Kv8 c[FM_LPT];
};
#endif

// Issues the loads of cache rows j0 .. j0 + FM_BC (all D columns, zero-padded to FM_DP; rows >= jEnd are zero).
void fm_fetch(out FmTile buf, uint8_t* cache, uint64_t kvRow, uint j0, uint jEnd, uint hk, uint tid)
{
#if KV_QUANT
    buf.mis = 0;
    [ForceUnroll]
    for (uint k = 0; k < FM_UPT; k++)
    {
        uint i = tid + k * FM_T;
        uint j = j0 + i / FM_BPR;
        [ForceUnroll] for (uint n = 0; n < FM_NW; n++) buf.w[k * FM_NW + n] = 0;
        if (i < FM_BC * FM_BPR && j < jEnd)
        {
            uint64_t elem = (kv_row(kvRow, j) + hk) * D + (i % FM_BPR) * 32;
            uint64_t s = (elem >> 5) * KV_BLOCK_BYTES;
            uint* w = (uint*)byteAt(cache, s & ~uint64_t(3));
            [ForceUnroll] for (uint n = 0; n + 1 < FM_NW; n++) buf.w[k * FM_NW + n] = w[n];
            if ((s & 2) != 0)
            {
                buf.w[k * FM_NW + FM_NW - 1] = w[FM_NW - 1];
                buf.mis |= 1u << k;
            }
            else
            {
                buf.w[k * FM_NW + FM_NW - 1] = uint(*(uint16_t*)(w + (FM_NW - 1)));   // the block's last 2 bytes
            }
        }
    }
#else
    [ForceUnroll]
    for (uint k = 0; k < FM_LPT; k++)
    {
        uint i = tid + k * FM_T;
        uint r = i / FM_CPR, c = (i % FM_CPR) * 8;
        uint j = j0 + r;
        bool valid = i < FM_BC * FM_CPR && j < jEnd && c < D;
        buf.c[k] = valid ? load_kv8(cache, (kv_row(kvRow, j) + hk) * D + c) : zero_kv8();
    }
#endif
}

// Stores fetched rows into fmKV as f16 (one 16-byte store per 8 values).
void fm_store(FmTile buf, uint tid)
{
#if KV_QUANT
    [ForceUnroll]
    for (uint k = 0; k < FM_UPT; k++)
    {
        uint i = tid + k * FM_T;
        if (i < FM_BC * FM_BPR)
        {
            bool mis = ((buf.mis >> k) & 1) != 0;
            uint v[FM_NW];   // the block's bytes, 4 per word
            [ForceUnroll]
            for (uint n = 0; n < FM_NW; n++)
            {
                uint hi = n + 1 < FM_NW ? buf.w[k * FM_NW + n + 1] << 16 : 0;
                v[n] = mis ? (buf.w[k * FM_NW + n] >> 16) | hi : buf.w[k * FM_NW + n];
            }

            float d = f16tof32(v[0] & 0xFFFF);
            uint dst = i / FM_BPR * (FM_KS / 8) + i % FM_BPR * 4;
            [ForceUnroll]
            for (uint c = 0; c < 4; c++)
            {
                float x[8];
                [ForceUnroll]
                for (uint e = 0; e < 8; e++)
                {
                    uint el = c * 8 + e;
#if KV_FMT == FMT_Q8_0
                    uint b = 2 + el;
                    x[e] = d * float(int(((v[b >> 2] >> ((b & 3) * 8)) & 0xFF) << 24) >> 24);
#else
                    uint b = 2 + (el & 15);
                    x[e] = d * (float((v[b >> 2] >> ((b & 3) * 8 + (el >= 16 ? 4 : 0))) & 0xF) - 8.0);
#endif
                }
                fmKV[dst + c] = pack_half8(x);
            }
        }
    }
#else
    [ForceUnroll]
    for (uint k = 0; k < FM_LPT; k++)
    {
        uint i = tid + k * FM_T;
        if (i < FM_BC * FM_CPR)
        {
            fmKV[i / FM_CPR * (FM_KS / 8) + i % FM_CPR] = kv8_to_half(buf.c[k]);
        }
    }
#endif
}

[shader("compute")]
[numthreads(FM_T, 1, 1)]
void fa_mqa(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
#if PAGED
    g_pages = p.pages;
#endif
    uint hk = gid.x / FM_TILES;
    uint g0 = gid.x % FM_TILES * FM_BR;   // first head of the tile within the group
    uint t = gid.y + p.row0;
    uint tid = ltid.x, sg = tid / 32, lane = tid % 32;

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

    // Query tile (scaled, f16) through fmKV into register fragments: subgroup sg owns the D chunks sg * FM_KPS ...
    // All loads are issued before the first store (one memory round trip).
    FmTile next;
    fm_fetch(next, p.kc, kvRow, jBeg, jEnd, hk, tid);
    float qv[FM_QPT][8];
    [ForceUnroll]
    for (uint k = 0; k < FM_QPT; k++)
    {
        uint i = tid + k * FM_T;
        uint r = i / FM_CPR, d0 = (i % FM_CPR) * 8, g = g0 + r;
        float* qr = p.q + uint64_t(t) * QS0 + uint64_t(hk * FM_G + g) * QS1;
        [ForceUnroll] for (uint e = 0; e < 8; e++) qv[k][e] = i < FM_BR * FM_CPR && g < FM_G && d0 + e < D ? qr[d0 + e] * SCALE : 0.0;
    }
    [ForceUnroll]
    for (uint k = 0; k < FM_QPT; k++)
    {
        uint i = tid + k * FM_T;
        if (i < FM_BR * FM_CPR)
        {
            fmKV[i / FM_CPR * (FM_KS / 8) + i % FM_CPR] = pack_half8(qv[k]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    FmA qa[FM_KPS];
    [ForceUnroll]
    for (uint c = 0; c < FM_KPS; c++)
    {
        uint k = min(sg * FM_KPS + c, uint(FM_KCH) - 1);   // chunks past D are loaded but not used
        qa[c] = FmA.Load<CoopMatMatrixLayout.RowMajor>(fmKV, k * 2, FM_KS / 8);
    }
    FmC o[FM_OPS];
    [ForceUnroll] for (uint i = 0; i < FM_OPS; i++) o[i] = FmC(0.0);
    GroupMemoryBarrierWithGroupSync();

    // Softmax layout: 8 adjacent lanes per row, 4 key columns each.
    uint row = tid / 8, sub = tid % 8;
    bool rowValid = g0 + row < FM_G;
    float m = -1e30, l = 0;

    for (uint j0 = jBeg; j0 < jEnd; j0 += FM_BC)
    {
        fm_store(next, tid);
        GroupMemoryBarrierWithGroupSync();
        if (j0 + FM_BC < jEnd)
        {
            fm_fetch(next, p.kc, kvRow, j0 + FM_BC, jEnd, hk, tid);   // in flight during this tile's products
        }

        // Partial scores of this subgroup's D chunks for both 16-key halves.
        {
            FmC s0 = FmC(0.0), s1 = FmC(0.0);
            [ForceUnroll]
            for (uint c = 0; c < FM_KPS; c++)
            {
                uint k = sg * FM_KPS + c;
                if (k < FM_KCH)
                {
                    FmB b0 = FmB.Load<CoopMatMatrixLayout.ColumnMajor>(fmKV, k * 2, FM_KS / 8);
                    FmB b1 = FmB.Load<CoopMatMatrixLayout.ColumnMajor>(fmKV, 2 * FM_KS + k * 2, FM_KS / 8);
                    s0 = coopMatMulAdd<float, false>(qa[c], b0, s0);
                    s1 = coopMatMulAdd<float, false>(qa[c], b1, s1);
                }
            }
            s0.Store<CoopMatMatrixLayout.RowMajor>(fmS, sg * FM_SS, FM_BC);
            s1.Store<CoopMatMatrixLayout.RowMajor>(fmS, sg * FM_SS + 16, FM_BC);
        }
        GroupMemoryBarrierWithGroupSync();

        float sv[4];
        float mx = -1e30;
        [ForceUnroll]
        for (uint e = 0; e < 4; e++)
        {
            uint c = sub * 4 + e;
            float sc = 0;
            [ForceUnroll] for (uint w = 0; w < FM_NSG; w++) sc += fmS[w * FM_SS + row * FM_BC + c];
#if HAS_SOFTCAP
            sc = SOFTCAP * tanh(sc / SOFTCAP);
#endif
            sv[e] = rowValid && j0 + c < jEnd ? sc : -1e30;
            mx = max(mx, sv[e]);
        }
        mx = max(mx, WaveReadLaneAt(mx, lane ^ 1));
        mx = max(mx, WaveReadLaneAt(mx, lane ^ 2));
        mx = max(mx, WaveReadLaneAt(mx, lane ^ 4));
        float mNew = max(m, mx);
        float corr = exp(m - mNew);
        m = mNew;
        float ps = 0;
        [ForceUnroll]
        for (uint e = 0; e < 4; e++)
        {
            float pe = sv[e] > -1e29 ? exp(sv[e] - m) : 0.0;
            ps += pe;
            fmP[row * FM_PS + sub * 4 + e] = half(pe);
        }
        ps += WaveReadLaneAt(ps, lane ^ 1);
        ps += WaveReadLaneAt(ps, lane ^ 2);
        ps += WaveReadLaneAt(ps, lane ^ 4);
        l = l * corr + ps;
        fmCorrM[row * 16 + sub * 2] = corr;
        fmCorrM[row * 16 + sub * 2 + 1] = corr;
        GroupMemoryBarrierWithGroupSync();   // P and corrections visible; partial scores consumed

        // Rescale the accumulators row by row: element-wise product with the matrix whose row r is corr[r] (same
        // accumulator type, so both share the implementation's element layout).
        FmC cm = FmC.Load<CoopMatMatrixLayout.RowMajor>(fmCorrM, 0, 16);
        [ForceUnroll] for (uint i = 0; i < FM_OPS; i++) o[i] = o[i] * cm;

        FmA p0 = FmA.Load<CoopMatMatrixLayout.RowMajor>(fmP, 0, FM_PS);
        FmA p1 = FmA.Load<CoopMatMatrixLayout.RowMajor>(fmP, 16, FM_PS);
        [ForceUnroll]
        for (uint i = 0; i < FM_OPS; i++)
        {
            uint ot = sg * FM_OPS + i;
            if (ot < FM_OT)
            {
                FmB v0 = FmB.Load<CoopMatMatrixLayout.RowMajor>(fmKV, ot * 2, FM_KS / 8);
                FmB v1 = FmB.Load<CoopMatMatrixLayout.RowMajor>(fmKV, 2 * FM_KS + ot * 2, FM_KS / 8);
                o[i] = coopMatMulAdd<float, false>(p0, v0, o[i]);
                o[i] = coopMatMulAdd<float, false>(p1, v1, o[i]);
            }
        }
        GroupMemoryBarrierWithGroupSync();   // tile, P and staging free for the next keys
    }

    // Row factors: 1/l for a direct output (sinks join the denominator), 1 for partials (decode_combine normalizes).
    bool direct = ns == 1;
    float f = 1.0;
    if (direct)
    {
#if HAS_SINKS
        if (rowValid)
        {
            float sk = p.sinks[hk * FM_G + g0 + row], mf = max(m, sk), cf = exp(m - mf);
            l = l * cf + exp(sk - mf);
            f = cf;
        }
#endif
        f = l > 0 ? f / l : 0.0;
    }
    else if (rowValid && sub == 0)
    {
        float* dst = p.part + ((uint64_t(split) * p.T + t) * HQ + hk * FM_G + g0 + row) * DEC_PS;
        dst[DV] = m;
        dst[DV + 1] = l;
    }
    fmCorrM[row * 16 + sub * 2] = f;
    fmCorrM[row * 16 + sub * 2 + 1] = f;
    GroupMemoryBarrierWithGroupSync();
    FmC fm = FmC.Load<CoopMatMatrixLayout.RowMajor>(fmCorrM, 0, 16);
    // Output rows are consecutive heads: [t][h][DV] (direct) or [split][t][h][DEC_PS] (partials).
    uint64_t rowBytes = direct ? DV * 4 : DEC_PS * 4;
    uint64_t first = direct ? (uint64_t(t) * HQ + hk * FM_G + g0) * DV * 4 : ((uint64_t(split) * p.T + t) * HQ + hk * FM_G + g0) * DEC_PS * 4;
    uint8_t* base = byteAt((uint8_t*)(direct ? p.o : p.part), first);
    [ForceUnroll]
    for (uint i = 0; i < FM_OPS; i++)
    {
        uint ot = sg * FM_OPS + i;
#if FM_G % FM_BR == 0 && DV % 16 == 0
        if (ot < FM_OT)
        {
            (o[i] * fm).Store<CoopMatMatrixLayout.RowMajor>((float*)byteAt(base, ot * 64), 0, uint(rowBytes / 4));
        }
#else
        // Partial head tiles or value widths: stage and write the valid elements.
        (o[i] * fm).Store<CoopMatMatrixLayout.RowMajor>(fmS, sg * FM_SS, 16);
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint e = 0; e < 8; e++)
        {
            uint idx = lane * 8 + e, r = idx / 16, d = ot * 16 + idx % 16;
            if (ot < FM_OT && g0 + r < FM_G && d < DV)
            {
                *(float*)byteAt(base, r * rowBytes + d * 4) = fmS[sg * FM_SS + idx];
            }
        }
        GroupMemoryBarrierWithGroupSync();
#endif
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

[shader("compute")]
[numthreads(DEC_WG, 1, 1)]
void decode(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
#if PAGED
    g_pages = p.pages;
#endif
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
            uint64_t row = kv_row(kvRow, j) + hk;
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
                vb[i][c] = j < jEnd && d0 < DV ? load_kv8(p.vc, row * VRS + d0) : zero_kv8();
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
#if HAS_ALIBI
                s[i][g] += alibi_bias(h0 + g, c0 + i * DEC_NRG + rg, L - 1);
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
