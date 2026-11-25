// Causal attention of T queries against a KV cache (online softmax, one workgroup per (head, query)).
// q: [T, HQ, D] with strides (QS0, QS1); caches: [ctx, HKV, D]/[ctx, HKV, DV] in format KV_FMT (FMT_F32, FMT_F16, or the
// GGUF block formats FMT_Q8_0 / FMT_Q4_0 with 32-value blocks; quantized rows are a multiple of 32 values);
// out: dense [T, HQ, DV]. Query t sees cache rows [0, n_past + t].
// Macros: HQ, HKV, D, DV, VRS (value cache row stride in elements, >= DV: the values are the first DV entries of each row,
//         e.g. the latent prefix of a multi-head latent attention cache whose K and V are one buffer), QS0, QS1, SCALE, KV_FMT
//         (KV_F16: KV_FMT == FMT_F16), WG (threads, = chunk of keys),
//         WINDOW (> 0: a query sees only keys less than WINDOW positions away: its last WINDOW keys when causal, the symmetric
//         window of bidirectional encoders with FULL_ATTN), HAS_ALIBI + ALIBI_SLOPES (initializer list of HQ floats: scores get
//         -slope[h] * |key position - query position|), HAS_POS_BIAS + POS_BIAS_RANGE + POS_BIAS_TABLE (initializer list of
//         HQ * (2 * POS_BIAS_RANGE + 1) floats: scores of head h get table[h][clamp(key position - query position, ±RANGE) + RANGE],
//         a learned relative position bias such as T5's),
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
// MIX_RUN > 0 (multi-sequence steps that mix short and long runs): `decode` / `decode_combine` skip the rows of runs with at
// least MIX_RUN rows and `fa` skips the runs with fewer, so the two kernels split one step's queries between them.
#ifndef MIX_RUN
#define MIX_RUN 0
#endif
#ifndef VRS
#define VRS DV
#endif
// CROSS_ATTN (FULL_ATTN only, without SPANS, WINDOW or CAUSAL): the keys are a separate sequence of S rows (parameter S), every
// query sees all of them (encoder-decoder cross-attention).
#ifndef CROSS_ATTN
#define CROSS_ATTN 0
#endif
// SPANS (FULL_ATTN only): spans [T, 2] (first key row, count) per query row — packed independent sequences; with CAUSAL a
// query row t additionally sees only key rows <= t.
#ifndef SPANS
#define SPANS 0
#endif
#ifndef CAUSAL
#define CAUSAL 0
#endif
// SEQ_ROWS > 0 (FULL_ATTN only, without SPANS, CROSS_ATTN, WINDOW, CAUSAL or score biases): the batched form, query rows
// [b·SEQ_ROWS, (b+1)·SEQ_ROWS) see only key rows [b·SEQ_KEYS, (b+1)·SEQ_KEYS) (equal-length sequences of a batch).
#ifndef SEQ_ROWS
#define SEQ_ROWS 0
#endif
#ifndef SEQ_KEYS
#define SEQ_KEYS 0
#endif
// HAS_MASK (FULL_ATTN only): p.mask is an additive f32 score mask, dense [rows, key rows] with MASK_PER_ROW, else one bias
// per key row shared by every query.
#ifndef HAS_MASK
#define HAS_MASK 0
#endif
#ifndef MASK_PER_ROW
#define MASK_PER_ROW 0
#endif
// FA_SPLIT (`fa` only): f32 Q/K/V/P are each split into two f16 parts (hi + lo) and every product is formed as
// hi·hi + hi·lo + lo·hi on the tensor cores, which keeps about 22 significant bits (near-f32 accuracy) for values within
// the f16 range. P is scaled by 2^12 before the split, so the lo parts of small probabilities stay normal.
#ifndef FA_SPLIT
#define FA_SPLIT 0
#endif
// FA_KSPLIT > 1 (`fa`, FULL_ATTN without SPANS, WINDOW, CAUSAL or sinks): the keys of every query block are split into FA_KSPLIT
// ranges, one workgroup each (grid z = sequence · FA_KSPLIT + range), which write unnormalized partials like SPLITS for `combine`.
// Gives short or few-head encoders enough workgroups to fill the device.
#ifndef FA_KSPLIT
#define FA_KSPLIT 1
#endif
// SELECTED > 0 (`main` only): sparse attention, query row t attends only the SELECTED key positions p.sel[t][..] (entries below 0 and
// positions the row cannot see are skipped), e.g. a lightning indexer's top-k.
#ifndef SELECTED
#define SELECTED 0
#endif
// COMP_ATTN (`main` only): compressed attention. p.cc is a compressed cache in KV_FMT with the key cache's row layout (KV_COMP positions per
// entry, rows addressed by seq_row_c); after the cache keys, the query at position qp attends in the same softmax the entries of its complete
// windows [0, (qp + 1) / KV_COMP), or with SELECTED the selected ones among them (the selection then lists entries, not key positions).
#ifndef COMP_ATTN
#define COMP_ATTN 0
#endif
#define RAW_SELECTED (SELECTED && !COMP_ATTN)
// REL_BIAS = E > 0 (`main` only): query-conditioned relative position bias. p.relb is [T, HQ, E]; the score of query row t, head h and the key
// at distance d = query position - key position gets p.relb[t][h][d] added (after the soft cap) when 0 <= d < E.
#ifndef REL_BIAS
#define REL_BIAS 0
#endif
// SEL_HEADS (with SELECTED): selections per query row, [T, SEL_HEADS, SELECTED]; query head h reads the selection of h / (HQ / SEL_HEADS).
#ifndef SEL_HEADS
#define SEL_HEADS 1
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
#ifndef HAS_POS_BIAS
#define HAS_POS_BIAS 0
#endif
#if HAS_POS_BIAS
static const float kPosBias[HQ * (2 * POS_BIAS_RANGE + 1)] = POS_BIAS_TABLE;
#endif
#define HAS_SCORE_BIAS (HAS_ALIBI || HAS_POS_BIAS)
#if HAS_SCORE_BIAS
// Additive score bias of head h for key position j and query position qp (ALiBi and/or the learned relative position bias).
float score_bias(uint h, uint j, uint qp)
{
    float b = 0;
#if HAS_ALIBI
    b += alibi_bias(h, j, qp);
#endif
#if HAS_POS_BIAS
    int d = clamp(int(j) - int(qp), -POS_BIAS_RANGE, POS_BIAS_RANGE);
    b += kPosBias[h * (2 * POS_BIAS_RANGE + 1) + uint(d + POS_BIAS_RANGE)];
#endif
    return b;
}
#endif
// KV_FP8: f8e4m3_b32 caches, blocks of { f32 d; u8 e4m3[32] } (36 bytes, 4-byte aligned), value = d * e4m3(q).
#define KV_FP8 (KV_FMT == FMT_F8E4M3_B32)
#define KV_QUANT (KV_FMT == FMT_Q8_0 || KV_FMT == FMT_Q4_0 || KV_FP8)
#define KV_BLOCK_BYTES (KV_FP8 ? 36 : KV_FMT == FMT_Q8_0 ? 34 : 18)
#define KV_SCALE_BYTES (KV_FP8 ? 4 : 2)
// One byte per value (q8_0, f8e4m3_b32) rather than q4_0's nibbles.
#define KV_BYTE_QUANTS (KV_FMT == FMT_Q8_0 || KV_FP8)

// ---- KV cache element access (elements are addressed linearly in the cache; quantized caches in 32-value blocks) -------
// Block of cache element `elem`, its scale (f16; f32 for f8e4m3_b32), and the 16-bit word holding the byte at offset `b` of its
// quants (blocks are 2-byte aligned: 34 / 18 bytes; 4-byte aligned: 36).
uint8_t* kv_block(uint8_t* base, uint64_t elem) { return byteAt(base, (elem >> 5) * KV_BLOCK_BYTES); }
uint kv_word(uint8_t* blk, uint b) { return uint(*(uint16_t*)(blk + KV_SCALE_BYTES + b)); }
float kv_scale(uint8_t* blk)
{
#if KV_FP8
    return *(float*)blk;
#else
    return h2f(*(uint16_t*)blk);
#endif
}

// An E4M3 code (low byte of `code`) as a float, exact: its sign and 7 exponent / mantissa bits placed as half bits are the
// value * 2^-8 (the exponent biases differ by 8; subnormals included).
float e4m3_value(uint code)
{
    return f16tof32(((code & 0x7Fu) << 7) | ((code & 0x80u) << 8)) * 256.0;
}

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
    uint b = KV_BYTE_QUANTS ? within : within & 15;
    r.d = kv_scale(blk);
#if KV_FP8
    r.q = uint2(*(uint*)(blk + 4 + b), *(uint*)(blk + 8 + b));
#else
    r.q = uint2(kv_word(blk, b) | (kv_word(blk, b + 2) << 16), kv_word(blk, b + 4) | (kv_word(blk, b + 6) << 16));
#endif
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
#if KV_FP8
    return e4m3_value(byte);
#elif KV_FMT == FMT_Q8_0
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
    uint b = KV_BYTE_QUANTS ? within : within & 15;
    uint byte = (kv_word(blk, b & ~1u) >> ((b & 1) * 8)) & 0xFF;
#if KV_FP8
    return kv_scale(blk) * e4m3_value(byte);
#elif KV_FMT == FMT_Q8_0
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

// First element of K / V head hk in cache row `row` (a kv_row, a multiple of HKV) for the kernels that take strided K / V
// (main, fe): the row and head strides KV_KS0 / KV_KS1 and KV_VS0 / KV_VS1 (multiples of 8 values) default to dense rows;
// wider row strides read slices of a fused Q / K / V projection in place.
#ifndef KV_KS0
#define KV_KS0 (uint64_t(HKV) * D)
#define KV_KS1 D
#define KV_VS0 (uint64_t(HKV) * VRS)
#define KV_VS1 VRS
#endif
#define K_ELEM(row, hk) ((row) / HKV * (KV_KS0) + uint64_t(hk) * (KV_KS1))
#define V_ELEM(row, hk) ((row) / HKV * (KV_VS0) + uint64_t(hk) * (KV_VS1))

// ---------------------------------------------------------------------------------------------
// Split-K flash decoding (`decode` + `decode_combine`, few queries against a KV cache). A workgroup handles HPW query
// heads that share one KV head (GQA group), so K/V rows are read once per group, and one contiguous range of keys of one
// query row. The split count follows the key count at run time: a row is split into ranges of at least DEC_MIN_KEYS
// keys, and into at most decode_max_splits(T) ranges (the grid z of the launch), so short contexts run a single range
// and write the output directly, while long ones spread over many workgroups. Ranges beyond a row's split count exit.
// Multi-sequence: every row has its own context length. Partials: part[split][t][h][DEC_PS] (acc, m, l).
// Macros: HPW (divisor of HQ/HKV), KLANES (lanes sharing a key/value row, power of two <= SG), DEC_WG (threads), DEC_R
// (rows per lane group and batch), DEC_BC (keys per batch = DEC_R * DEC_WG / KLANES), DEC_MAX_SPLITS, DEC_MIN_KEYS
// (multiple of DEC_BC), DEC_TARGET_WG (workgroups the split count aims for), DEC_INVARIANT_ROWS (optional: launches of up to
// this many rows split like single-row launches, so every row's result equals that of a launch of its own).
#ifndef DEC_FUSED_COMBINE
#define DEC_FUSED_COMBINE 0   // decode: merge the split partials in the last workgroup of a row (see the end of decode)
#endif
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
#ifdef DEC_INVARIANT_ROWS
    if (T <= DEC_INVARIANT_ROWS)
    {
        T = 1;   // row-invariant launches: every row splits its keys as in a single-row launch
    }
#endif
    return clamp(uint(DEC_TARGET_WG) / max(T * DEC_GROUPS, 1u), 1u, uint(DEC_MAX_SPLITS));
}

// Keys per split range of a row with `span` keys (a multiple of DEC_BC). With ranges of several batches (DEC_MIN_KEYS > DEC_BC):
// as many ranges as hold DEC_MIN_KEYS keys each, at most decode_max_splits(T), the keys spread evenly over them; rounding the
// range up to DEC_MIN_KEYS instead left a short last range (300 keys in 256 + 44: a second workgroup and the combine for little
// work). Launches with few workgroups (under an eighth of DEC_TARGET_WG) take one range more instead (the count rounded up), since
// there the latency of a range, not the work, sets the time (RTX PRO 4500, one Qwen3-8B-FP8 row at ~280 keys: 2 ranges 11.40 ms
// per step, 3 ranges 11.23). Ranges of one batch: the even split rounded up to whole batches.
uint decode_split_keys(uint span, uint T)
{
    uint ms = decode_max_splits(T);
#if DEC_MIN_KEYS > DEC_BC
    uint ns = clamp(span / uint(DEC_MIN_KEYS), 1u, ms);
#ifdef DEC_INVARIANT_ROWS
    uint rows = T <= DEC_INVARIANT_ROWS ? 1u : T;
#else
    uint rows = T;
#endif
    if (rows * DEC_GROUPS * ns < uint(DEC_TARGET_WG) / 8)
    {
        ns = clamp((span + DEC_MIN_KEYS - 1) / uint(DEC_MIN_KEYS), 1u, ms);   // few workgroups: one range more, shorter ones
    }
    uint per = (span + ns - 1) / ns;
    return (per + DEC_BC - 1) / DEC_BC * DEC_BC;
#else
    uint per = (span + ms - 1) / ms;
    per = (per + DEC_BC - 1) / DEC_BC * DEC_BC;
    return max(per, uint(DEC_MIN_KEYS));
#endif
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

#if defined(KERNEL_DECODE_COMBINE) || (defined(KERNEL_DECODE) && DEC_FUSED_COMBINE)
#define DC_WG 128
#define DC_D4 (DV / 4)                                  // float4 columns of a partial row
#define DC_SL (DC_WG / DC_D4 > 0 ? DC_WG / DC_D4 : 1)   // split lanes: thread (sl, c) reads splits sl, sl + DC_SL, ...
// Splits per thread and round (loads in flight). The fused combine of decode loads fewer per round, which keeps its registers
// (and so the occupancy of decode's key loop) low; the later rounds visit the splits in the same order, so the sums are equal.
#ifdef KERNEL_DECODE
#define DC_K 4
#else
#define DC_K 16
#endif
groupshared float dcW[DEC_MAX_SPLITS];
groupshared float dcRed[DC_WG / SG];
groupshared float4 dcAcc[DC_SL * DC_D4];

float4 load4(float* ptr)
{
    return *(float4*)ptr;
}

// Merges the ns split partials of head h of query row t into o (DC_WG threads, tid the thread's index). The split
// statistics and the partial rows are loaded in the same round trip; weights go through shared memory.
void decode_combine_row(float* part, float* o, float* sinks, uint h, uint t, uint ns, uint T, uint tid)
{
    uint64_t rowStride = uint64_t(T) * HQ * DEC_PS;   // floats between the partials of consecutive splits
    float* base = part + (uint64_t(t) * HQ + h) * DEC_PS;
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
    M = max(M, sinks[h]);
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
    den += exp(sinks[h] - M);
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
    float* orow = o + (uint64_t(t) * HQ + h) * DV;
    for (uint i = tid; i < DC_D4; i += DC_WG)
    {
        float4 sum = float4(0);
        [ForceUnroll] for (uint k = 0; k < DC_SL; k++) sum += dcAcc[k * DC_D4 + i];
        *(float4*)(orow + i * 4) = sum * inv;
    }
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

// One workgroup per (head, row): merges the row's split partials; rows that ran a single range were written by `decode`.
[shader("compute")]
[numthreads(DC_WG, 1, 1)]
void decode_combine(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform DecodeCombineParams p)
{
    uint h = gid.x, t = gid.y, tid = ltid.x;
#if MULTI_SEQ && MIX_RUN > 0
    if (p.seq[t * 4 + 2] >= MIX_RUN)
    {
        return;
    }
#endif
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

    decode_combine_row(p.part, p.o, p.sinks, h, t, ns, p.T, tid);
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
#if SELECTED
    int* sel;    // [T, SEL_HEADS, SELECTED] key positions (COMP_ATTN: entries) per query row (and selection head)
#endif
#if COMP_ATTN
    uint8_t* cc; // compressed cache [entries, HKV, VRS]
#endif
#if HAS_MASK
    float* mask; // additive score mask (see HAS_MASK)
#endif
#if REL_BIAS
    float* relb; // [T, HQ, REL_BIAS] query-conditioned relative position bias by query - key distance
#endif
    uint* arrivals;   // DEC_FUSED_COMBINE: [T, DEC_GROUPS] split arrivals of `decode`, zero between launches (else unused)
    uint T;
    uint row0;   // first query row of this launch (long prefills are sliced into several launches)
#if CROSS_ATTN
    uint S;      // key rows (FULL_ATTN cross-attention: K/V are [S, HKV, D], independent of the T queries)
#endif
};

#if HAS_MASK
// Offset of query row t's mask row in p.mask (0 when one row serves every query).
uint64_t mask_row(uint T, uint S, uint t)
{
#if MASK_PER_ROW
#if SEQ_ROWS
    return uint64_t(t) * (uint64_t(T / SEQ_ROWS) * SEQ_KEYS);
#elif CROSS_ATTN
    return uint64_t(t) * S;
#else
    return uint64_t(t) * T;
#endif
#else
    return 0;
#endif
}
#endif



#ifndef WG
#define WG 128
#endif
#define DPT ((DV + WG - 1) / WG)   // output dims per thread

groupshared float sQ[D];
groupshared float sP[WG];
#if SELECTED
groupshared uint sJ[WG];   // key position of every entry of the current chunk
#endif
#if COMP_ATTN
groupshared uint64_t sJ2[WG];   // compressed cache row (x HKV + head) of every entry of the current chunk
#endif
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
#if D % 8 == 0
    [ForceUnroll]
    for (uint i = 0; i < D / 8; i++)
    {
        float k[8];
        unpack_kv8(load_kv8(kc, rowElem + i * 8), k);
        [ForceUnroll] for (uint e = 0; e < 8; e++) s += sQ[8 * i + e] * k[e];
    }
#else
    // Head sizes that are not a multiple of 8: key rows are not 16-byte aligned, so one element per load.
    for (uint i = 0; i < D; i++) s += sQ[i] * load_kv(kc, rowElem + i);
#endif
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
    uint keyBase = 0;     // batched form: first key row of the query's sequence
#if FULL_ATTN && SEQ_ROWS
    keyBase = t / SEQ_ROWS * SEQ_KEYS;
    kvRow = uint64_t(keyBase) * HKV;
    uint L = SEQ_KEYS;
#elif FULL_ATTN && SPANS
    uint spBeg = uint(p.spans[t * 2]);
    uint L = spBeg + uint(p.spans[t * 2 + 1]);
#if CAUSAL
    L = min(L, t + 1);
#endif
    L = max(L, spBeg);
#elif FULL_ATTN
#if CROSS_ATTN
    uint L = p.S;   // cross-attention: every query sees all S keys
#else
    uint L = p.T;   // bidirectional: every query sees all T keys (K/V are the dense [T, HKV, D] tensors of this launch)
#endif
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
#if RAW_SELECTED
    jBeg = 0;          // entries of the selection; L stays the visible key count
    jEnd = SELECTED;
#endif

    float* qr = p.q + uint64_t(t) * QS0 + uint64_t(h) * QS1;
    for (uint i = tid; i < D; i += WG) sQ[i] = qr[i] * SCALE;
    GroupMemoryBarrierWithGroupSync();
#if HAS_MASK
#if CROSS_ATTN
    float* maskRow = p.mask + mask_row(p.T, p.S, t) + keyBase;
#else
    float* maskRow = p.mask + mask_row(p.T, 0, t) + keyBase;
#endif
#endif

    float m = -1e30, l = 0;
    float acc[DPT];
    [ForceUnroll] for (int i = 0; i < DPT; i++) acc[i] = 0;

    for (uint c0 = jBeg; c0 < jEnd; c0 += WG)
    {
#if RAW_SELECTED
        int selected = c0 + tid < jEnd ? p.sel[(uint64_t(t) * SEL_HEADS + h / (HQ / SEL_HEADS)) * SELECTED + c0 + tid] : -1;
        bool valid = selected >= 0 && uint(selected) < L;
        uint j = valid ? uint(selected) : 0u;
        sJ[tid] = j;
#else
        uint j = c0 + tid;
        bool valid = j < jEnd;
#endif
        float s = -1e30;
        if (valid) s = dot_k(p.kc, K_ELEM(kv_row(kvRow, j), hk));
#if HAS_SOFTCAP
        if (valid) s = SOFTCAP * tanh(s / SOFTCAP);
#endif
#if HAS_SCORE_BIAS
        if (valid) s += score_bias(h, j, qPos);
#endif
#if REL_BIAS
        if (valid && j <= qPos && qPos - j < REL_BIAS) s += p.relb[(uint64_t(t) * HQ + h) * REL_BIAS + (qPos - j)];
#endif
#if HAS_MASK
        if (valid) s += maskRow[j];
#endif
        float mc = reduce_max(s, tid);
        float mNew = max(m, mc);
        float pj = valid ? exp(s - mNew) : 0;
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
#if RAW_SELECTED
                    a += sP[jj] * load_kv(p.vc, V_ELEM(kv_row(kvRow, sJ[jj]), hk) + d);
#else
                    a += sP[jj] * load_kv(p.vc, V_ELEM(kv_row(kvRow, c0 + jj), hk) + d);
#endif
            acc[di] = a;
        }
        GroupMemoryBarrierWithGroupSync();
    }

#if COMP_ATTN
    // The compressed entries of the complete windows (or the selected ones among them), continuing the online softmax.
#if MULTI_SEQ
    uint cSlot = uint(p.seq[t * 4]);
#else
    uint cSlot = 0;
#endif
    uint complete = (qPos + 1) / KV_COMP;
#if SELECTED
    uint cEnd = SELECTED;
#else
    uint cEnd = complete;
#endif
    for (uint c0 = 0; c0 < cEnd; c0 += WG)
    {
#if SELECTED
        int picked = c0 + tid < cEnd ? p.sel[(uint64_t(t) * SEL_HEADS + h / (HQ / SEL_HEADS)) * SELECTED + c0 + tid] : -1;
        bool valid = picked >= 0 && uint(picked) < complete;
        uint e = valid ? uint(picked) : 0u;
#else
        uint e = c0 + tid;
        bool valid = e < cEnd;
#endif
        uint64_t cRow = valid ? (seq_row_c(SEQ_PAGES(p), cSlot, e) * HKV + hk) : 0;
        sJ2[tid] = cRow;
        float s = -1e30;
        if (valid) s = dot_k(p.cc, cRow * D);
        float mc = reduce_max(s, tid);
        float mNew = max(m, mc);
        float pj = valid ? exp(s - mNew) : 0;
        sP[tid] = pj;
        float corr = exp(m - mNew);
        l = l * corr + reduce_sum(pj, tid);   // reduce_sum syncs, so sP and sJ2 are visible afterwards
        m = mNew;
        uint cnt = min(uint(WG), cEnd - c0);
        [ForceUnroll]
        for (int di = 0; di < DPT; di++)
        {
            uint d = tid + di * WG;
            float a = acc[di] * corr;
            if (d < DV)
                for (uint jj = 0; jj < cnt; jj++)
                    if (sP[jj] != 0) a += sP[jj] * load_kv(p.cc, sJ2[jj] * VRS + d);
            acc[di] = a;
        }
        GroupMemoryBarrierWithGroupSync();
    }
#endif

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

#if FA_SPLIT
// x as hi + lo: hi is x rounded to 11 significant bits (exact in f16 within its normal range), lo the remainder. The rounding
// works on the bits: compilers may fold a float(half(x)) round trip to x, which would leave lo = 0.
void fa_split(float x, out half hi, out half lo)
{
    float h = asfloat((asuint(x) + 0x1000u) & 0xFFFFE000u);
    hi = half(h);
    lo = half(x - h);
}
#endif

#ifdef KERNEL_FA
// ---------------------------------------------------------------------------------------------
// Flash attention for many queries (prefill) on tensor cores. A workgroup handles FA_BR consecutive queries of one
// head with 4 subgroups. Per FA_BC-key block: S = Q·Kᵀ with cooperative matrices (stored to shared memory), online
// softmax in scalar code — each thread owns the statistics of one query row and a quarter of its output row in
// registers, so rescaling rows is plain arithmetic — then P·V with cooperative matrices, staged through shared memory
// in FA_DC-wide chunks and accumulated into the registers. Causal: query t sees keys [0, n_past + t]; FULL_ATTN:
// every key of the launch (bidirectional); WINDOW: only the last WINDOW keys (FULL_ATTN: the symmetric window); HAS_SOFTCAP: capped
// scores; HAS_ALIBI / HAS_POS_BIAS: distance biases.
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
#if SL_TARGET_CUDA
#include "cuda_mma.slang"
typealias FaA = CuA16;
typealias FaB = CuB16;
typealias FaC = CuAccF;
#else
typealias FaA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias FaB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias FaC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
#endif
// FA_FRAG8 (DeviceFeatures.MatrixF16Small, Metal simdgroup_matrix): each 16x16 tile of S and of P·V is computed as 2 x 2 fragments of
// 8x8 over K steps of 8, with the same shared-memory layout and stores.
#ifndef FA_FRAG8
#define FA_FRAG8 0
#endif
#if FA_FRAG8
typealias Fa8A = CoopMat<half, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixA>;
typealias Fa8B = CoopMat<half, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixB>;
typealias Fa8C = CoopMat<float, MemoryScope.Subgroup, 8, 8, CoopMatMatrixUse.MatrixAccumulator>;
#endif

groupshared half faQ[FA_BR * FA_QS];
groupshared half faKV[FA_BC * FA_KS];
groupshared float faS[FA_BR * FA_SS];
groupshared half faP[FA_BR * FA_PS];
groupshared float faO[FA_BR * FA_DC];
#if FA_SPLIT
groupshared half faQl[FA_BR * FA_QS];   // lo parts (FA_SPLIT)
groupshared half faKVl[FA_BC * FA_KS];
groupshared half faPl[FA_BR * FA_PS];
#define FA_P_SCALE 4096.0
// Power-of-two scales (split_scale_outside) of each query row, each key row of the K tile and the V tile as a whole whose largest
// magnitude is outside [2^-6, 2^15), applied before the f16 split so neither part over- or underflows (as fe's). One array of f32 bits (separate groupshared scalars here gave wrong
// results): FA_BR query row scales, FA_BC key row maxima, FA_BC inverse key row scales, the V tile's maximum and inverse scale.
groupshared uint faScales[FA_BR + 2 * FA_BC + 2];
#define FA_QSCALE (0)
#define FA_KMAX (FA_BR)
#define FA_KINV (FA_BR + FA_BC)
#define FA_VMAX (FA_BR + 2 * FA_BC)
#define FA_VINV (FA_BR + 2 * FA_BC + 1)
#endif

// Rows j0.. of K or V (dim columns of rows rowStride wide, zero-padded to dimP) into faKV as f16 (FA_SPLIT: hi parts in faKV,
// lo parts in faKVl); rows >= L are zero.
void fa_load_kv(uint8_t* cache, uint64_t kvRow, uint j0, uint L, uint hk, uint dim, uint rowStride, uint dimP, uint tid, bool values)
{
    uint per = dimP / 8;
#if FA_SPLIT
    // FA_SPLIT: the maxima first (per key row, or over the V tile; cleared by the other operand's load), then the
    // scaled parts.
    float vmax = 0;
    for (uint i = tid; i < FA_BC * per; i += FA_T)
    {
        uint r = i / per, c = (i % per) * 8;
        uint j = j0 + r;
        if (j < L && c < dim)
        {
            float v[8];
            unpack_kv8(load_kv8(cache, (kv_row(kvRow, j) + hk) * rowStride + c), v);
            float a = 0;
            [ForceUnroll] for (uint e = 0; e < 8; e++) a = max(a, abs(v[e]));
            if (values) vmax = max(vmax, a); else InterlockedMax(faScales[FA_KMAX + r], asuint(a));
        }
    }
    if (values)
    {
        vmax = WaveActiveMax(vmax);
        if (WaveIsFirstLane()) InterlockedMax(faScales[FA_VMAX], asuint(vmax));
    }
    GroupMemoryBarrierWithGroupSync();
    float vScale = split_scale_outside(asfloat(faScales[FA_VMAX]));
    if (values && tid == 0) faScales[FA_VINV] = asuint(1.0 / vScale);
    // Each operand's maxima were last read before the barrier ahead of this load.
    if (values) { for (uint r = tid; r < FA_BC; r += FA_T) faScales[FA_KMAX + r] = 0; }
    else if (tid == 0) faScales[FA_VMAX] = 0;
#endif
    for (uint i = tid; i < FA_BC * per; i += FA_T)
    {
        uint r = i / per, c = (i % per) * 8;
        uint j = j0 + r;
        uint o = r * FA_KS + c;
        if (j < L && c < dim)
        {
            float v[8];
            unpack_kv8(load_kv8(cache, (kv_row(kvRow, j) + hk) * rowStride + c), v);
#if FA_SPLIT
            float scale = values ? vScale : split_scale_outside(asfloat(faScales[FA_KMAX + r]));
            if (!values && c == 0) faScales[FA_KINV + r] = asuint(1.0 / scale);
#endif
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
#if FA_SPLIT
                fa_split(v[e] * scale, faKV[o + e], faKVl[o + e]);
#else
                faKV[o + e] = half(v[e]);
#endif
            }
        }
        else
        {
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
                faKV[o + e] = 0.0h;
#if FA_SPLIT
                faKVl[o + e] = 0.0h;
#endif
            }
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
    uint keyBase = 0;           // batched form: first key row of the sequence
#if FULL_ATTN && SEQ_ROWS
    // Batched form: workgroup (h, blk, b) handles query tile blk of sequence b (grid z = sequences).
    uint nPast = 0;
    qT = SEQ_ROWS; qRow0 = gid.z / FA_KSPLIT * SEQ_ROWS; keyBase = gid.z / FA_KSPLIT * SEQ_KEYS;
    kvRow = uint64_t(keyBase) * HKV;
#elif FULL_ATTN
    uint nPast = 0;
#elif MULTI_SEQ
    // Blocks never straddle sequences: workgroup (h, blk, u) handles query tile blk of run u (grid z = runs,
    // y = tiles of the longest possible run); tiles past a run's end exit.
    uint u = gid.z, blk = gid.y;
    int4 run = int4(p.runs[u * 4], p.runs[u * 4 + 1], p.runs[u * 4 + 2], p.runs[u * 4 + 3]);
    if (blk * FA_BR >= uint(run.z)) return;
#if MIX_RUN > 0
    if (uint(run.z) < MIX_RUN) return;
#endif
    t0 = blk * FA_BR; qT = uint(run.z); qRow0 = uint(run.y);
    uint nPast = uint(run.w);
    kvRow = kv_base(uint(run.x));
#else
    uint nPast = p.rt[RT_N_PAST];
#endif

#if FA_SPLIT
    for (uint r = tid; r < FA_BR; r += FA_T)
    {
        uint t = t0 + r;
        float a = 0;
        for (uint d = 0; d < D; d++) a = max(a, t < qT ? abs(p.q[uint64_t(qRow0 + t) * QS0 + uint64_t(h) * QS1 + d]) : 0.0);
        faScales[FA_QSCALE + r] = asuint(split_scale_outside(a));
    }
    for (uint r = tid; r < FA_BC; r += FA_T) faScales[FA_KMAX + r] = 0;
    if (tid == 0) faScales[FA_VMAX] = 0;
    GroupMemoryBarrierWithGroupSync();
#endif
    for (uint i = tid; i < FA_BR * FA_DP; i += FA_T)
    {
        uint r = i / FA_DP, d = i % FA_DP, t = t0 + r;
#if FA_SPLIT
        // Times the row's power-of-two scale, without SCALE (applied to the f32 scores), so small entries keep their lo parts.
        float x = t < qT && d < D ? p.q[uint64_t(qRow0 + t) * QS0 + uint64_t(h) * QS1 + d] * asfloat(faScales[FA_QSCALE + r]) : 0.0;
        fa_split(x, faQ[r * FA_QS + d], faQl[r * FA_QS + d]);
#else
        faQ[r * FA_QS + d] = half(t < qT && d < D ? p.q[uint64_t(qRow0 + t) * QS0 + uint64_t(h) * QS1 + d] * SCALE : 0.0);
#endif
    }

    uint row = tid / 4, sub = tid % 4;
    uint tq = t0 + row;
    bool rowValid = tq < qT;
#if HAS_SCORE_BIAS
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
#if FULL_ATTN && SEQ_ROWS
    uint L = SEQ_KEYS, rowEnd = SEQ_KEYS;
#elif FULL_ATTN
#if CROSS_ATTN
    uint L = p.S, rowEnd = p.S;
#else
    uint L = p.T, rowEnd = p.T;
#endif
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
#if HAS_MASK
#if CROSS_ATTN
    float* maskRow = p.mask + mask_row(p.T, p.S, qRow0 + min(tq, qT - 1)) + keyBase;
#else
    float* maskRow = p.mask + mask_row(p.T, 0, qRow0 + min(tq, qT - 1)) + keyBase;
#endif
#endif
#if FA_KSPLIT > 1
    uint ksplit = gid.z % FA_KSPLIT;
    uint kPer = ((L - jFirst + FA_KSPLIT - 1) / FA_KSPLIT + FA_BC - 1) / FA_BC * FA_BC;
    uint jStart = min(L, jFirst + ksplit * kPer), jStop = min(L, jStart + kPer);
#else
    uint jStart = jFirst, jStop = L;
#endif
    GroupMemoryBarrierWithGroupSync();

    for (uint j0 = jStart; j0 < jStop; j0 += FA_BC)
    {
        fa_load_kv(p.kc, kvRow, j0, L, hk, D, D, FA_DP, tid, false);
        GroupMemoryBarrierWithGroupSync();

        // S: subgroup sg computes rows (sg / 2) * 16, keys (sg % 2) * 16.
#if FA_FRAG8
        {
            Fa8C s8[2][2];
            [ForceUnroll] for (uint i = 0; i < 2; i++) [ForceUnroll] for (uint jj = 0; jj < 2; jj++) s8[i][jj] = Fa8C(0.0);
            uint qa = (sg / 2) * 16 * FA_QS, kb = (sg % 2) * 16 * FA_KS;
            [ForceUnroll]
            for (uint k = 0; k < FA_DP; k += 8)
            {
                Fa8A a[2];
                Fa8B b[2];
                [ForceUnroll] for (uint i = 0; i < 2; i++) a[i] = Fa8A.Load<CoopMatMatrixLayout.RowMajor>(faQ, qa + i * 8 * FA_QS + k, FA_QS);
                [ForceUnroll] for (uint jj = 0; jj < 2; jj++) b[jj] = Fa8B.Load<CoopMatMatrixLayout.ColumnMajor>(faKV, kb + jj * 8 * FA_KS + k, FA_KS);
#if FA_SPLIT
                Fa8A al[2];
                Fa8B bl[2];
                [ForceUnroll] for (uint i = 0; i < 2; i++) al[i] = Fa8A.Load<CoopMatMatrixLayout.RowMajor>(faQl, qa + i * 8 * FA_QS + k, FA_QS);
                [ForceUnroll] for (uint jj = 0; jj < 2; jj++) bl[jj] = Fa8B.Load<CoopMatMatrixLayout.ColumnMajor>(faKVl, kb + jj * 8 * FA_KS + k, FA_KS);
                [ForceUnroll] for (uint i = 0; i < 2; i++) [ForceUnroll] for (uint jj = 0; jj < 2; jj++)
                {
                    s8[i][jj] = coopMatMulAdd<float, false>(al[i], b[jj], s8[i][jj]);
                    s8[i][jj] = coopMatMulAdd<float, false>(a[i], bl[jj], s8[i][jj]);
                }
#endif
                [ForceUnroll] for (uint i = 0; i < 2; i++) [ForceUnroll] for (uint jj = 0; jj < 2; jj++) s8[i][jj] = coopMatMulAdd<float, false>(a[i], b[jj], s8[i][jj]);
            }
            [ForceUnroll] for (uint i = 0; i < 2; i++) [ForceUnroll] for (uint jj = 0; jj < 2; jj++)
                s8[i][jj].Store<CoopMatMatrixLayout.RowMajor>(faS, ((sg / 2) * 16 + i * 8) * FA_SS + (sg % 2) * 16 + jj * 8, FA_SS);
        }
#else
        {
            FaC sacc = FaC(0.0);
            uint qa = (sg / 2) * 16 * FA_QS, kb = (sg % 2) * 16 * FA_KS;
            [ForceUnroll]
            for (uint k = 0; k < FA_DP; k += 16)
            {
                FaA a = FaA.Load<CoopMatMatrixLayout.RowMajor>(faQ, qa + k, FA_QS);
                FaB b = FaB.Load<CoopMatMatrixLayout.ColumnMajor>(faKV, kb + k, FA_KS);
#if FA_SPLIT
                FaA al = FaA.Load<CoopMatMatrixLayout.RowMajor>(faQl, qa + k, FA_QS);
                FaB bl = FaB.Load<CoopMatMatrixLayout.ColumnMajor>(faKVl, kb + k, FA_KS);
                sacc = coopMatMulAdd<float, false>(al, b, sacc);
                sacc = coopMatMulAdd<float, false>(a, bl, sacc);
#endif
                sacc = coopMatMulAdd<float, false>(a, b, sacc);
            }
            sacc.Store<CoopMatMatrixLayout.RowMajor>(faS, (sg / 2) * 16 * FA_SS + (sg % 2) * 16, FA_SS);
        }
#endif
        GroupMemoryBarrierWithGroupSync();   // S visible; K no longer needed

        fa_load_kv(p.vc, kvRow, j0, L, hk, DV, VRS, FA_DVP, tid, true);

        // Online softmax of row `row`, columns sub * 8 .. + 8 (the row's 4 threads are adjacent lanes).
        float sv[8];
        float mx = -1e30;
        [ForceUnroll]
        for (uint e = 0; e < 8; e++)
        {
            uint c = sub * 8 + e, j = j0 + c;
            float sc = faS[row * FA_SS + c];
#if FA_SPLIT
            sc = sc * (1.0 / asfloat(faScales[FA_QSCALE + row]) * asfloat(faScales[FA_KINV + c])) * SCALE;
#endif
#if HAS_SOFTCAP
            sc = SOFTCAP * tanh(sc / SOFTCAP);
#endif
#if HAS_SCORE_BIAS
            sc += score_bias(h, j, faQueryPos);
#endif
            bool visible = rowValid && j < rowEnd && j >= rowBeg && j < jStop;
#if HAS_MASK
            if (visible) sc += maskRow[j];
#endif
            sv[e] = visible ? sc : -1e30;
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
#if FA_SPLIT
            fa_split(pe * FA_P_SCALE, faP[row * FA_PS + sub * 8 + e], faPl[row * FA_PS + sub * 8 + e]);
#else
            faP[row * FA_PS + sub * 8 + e] = half(pe);
#endif
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
#if FA_FRAG8
                Fa8C acc8[2][2];
                [ForceUnroll] for (uint i = 0; i < 2; i++) [ForceUnroll] for (uint jj = 0; jj < 2; jj++) acc8[i][jj] = Fa8C(0.0);
                [ForceUnroll]
                for (uint k = 0; k < FA_BC; k += 8)
                {
                    Fa8A a[2];
                    Fa8B b[2];
                    [ForceUnroll] for (uint i = 0; i < 2; i++) a[i] = Fa8A.Load<CoopMatMatrixLayout.RowMajor>(faP, (rb * 16 + i * 8) * FA_PS + k, FA_PS);
                    [ForceUnroll] for (uint jj = 0; jj < 2; jj++) b[jj] = Fa8B.Load<CoopMatMatrixLayout.RowMajor>(faKV, k * FA_KS + c * FA_DC + cb * 16 + jj * 8, FA_KS);
#if FA_SPLIT
                    Fa8A al[2];
                    Fa8B bl[2];
                    [ForceUnroll] for (uint i = 0; i < 2; i++) al[i] = Fa8A.Load<CoopMatMatrixLayout.RowMajor>(faPl, (rb * 16 + i * 8) * FA_PS + k, FA_PS);
                    [ForceUnroll] for (uint jj = 0; jj < 2; jj++) bl[jj] = Fa8B.Load<CoopMatMatrixLayout.RowMajor>(faKVl, k * FA_KS + c * FA_DC + cb * 16 + jj * 8, FA_KS);
                    [ForceUnroll] for (uint i = 0; i < 2; i++) [ForceUnroll] for (uint jj = 0; jj < 2; jj++)
                    {
                        acc8[i][jj] = coopMatMulAdd<float, false>(al[i], b[jj], acc8[i][jj]);
                        acc8[i][jj] = coopMatMulAdd<float, false>(a[i], bl[jj], acc8[i][jj]);
                    }
#endif
                    [ForceUnroll] for (uint i = 0; i < 2; i++) [ForceUnroll] for (uint jj = 0; jj < 2; jj++) acc8[i][jj] = coopMatMulAdd<float, false>(a[i], b[jj], acc8[i][jj]);
                }
                [ForceUnroll] for (uint i = 0; i < 2; i++) [ForceUnroll] for (uint jj = 0; jj < 2; jj++)
                    acc8[i][jj].Store<CoopMatMatrixLayout.RowMajor>(faO, (rb * 16 + i * 8) * FA_DC + cb * 16 + jj * 8, FA_DC);
#else
                FaC acc = FaC(0.0);
                [ForceUnroll]
                for (uint k = 0; k < FA_BC; k += 16)
                {
                    FaA a = FaA.Load<CoopMatMatrixLayout.RowMajor>(faP, rb * 16 * FA_PS + k, FA_PS);
                    FaB b = FaB.Load<CoopMatMatrixLayout.RowMajor>(faKV, k * FA_KS + c * FA_DC + cb * 16, FA_KS);
#if FA_SPLIT
                    FaA al = FaA.Load<CoopMatMatrixLayout.RowMajor>(faPl, rb * 16 * FA_PS + k, FA_PS);
                    FaB bl = FaB.Load<CoopMatMatrixLayout.RowMajor>(faKVl, k * FA_KS + c * FA_DC + cb * 16, FA_KS);
                    acc = coopMatMulAdd<float, false>(al, b, acc);
                    acc = coopMatMulAdd<float, false>(a, bl, acc);
#endif
                    acc = coopMatMulAdd<float, false>(a, b, acc);
                }
                acc.Store<CoopMatMatrixLayout.RowMajor>(faO, rb * 16 * FA_DC + cb * 16, FA_DC);
#endif
            }
            GroupMemoryBarrierWithGroupSync();
#if FA_SPLIT
            [ForceUnroll] for (uint e = 0; e < 16; e++) o[c * 16 + e] += faO[row * FA_DC + sub * 16 + e] * (asfloat(faScales[FA_VINV]) * (1.0 / FA_P_SCALE));
#else
            [ForceUnroll] for (uint e = 0; e < 16; e++) o[c * 16 + e] += faO[row * FA_DC + sub * 16 + e];
#endif
            GroupMemoryBarrierWithGroupSync();
        }
    }

#if FA_KSPLIT > 1
    if (rowValid)
    {
        // Unnormalized partial of this key range (see SPLITS): part[range][row][h] = (o[DV], m, l).
        float* ps = p.part + ((uint64_t(ksplit) * p.T + qRow0 + tq) * HQ + h) * (DV + 2);
        [ForceUnroll]
        for (uint c = 0; c < FA_NC; c++)
            [ForceUnroll] for (uint e = 0; e < 16; e++) if (c * FA_DC + sub * 16 + e < DV) ps[c * FA_DC + sub * 16 + e] = o[c * 16 + e];
        if (sub == 0) { ps[DV] = m; ps[DV + 1] = l; }
    }
    return;
#endif
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

#ifdef KERNEL_FE
// ---------------------------------------------------------------------------------------------
// Flash attention for bidirectional encoders: FULL_ATTN without SPANS, WINDOW, score biases, softcap or sinks (optional
// HAS_MASK, SEQ_ROWS, CROSS_ATTN, FA_KSPLIT, FA_SPLIT). A workgroup handles FE_BR = 64 queries of one head with 4 subgroups;
// subgroup sg owns rows sg * 16 .. + 16 for the whole key loop, so its query fragments and output accumulators stay in
// registers and its scores and probabilities only pass through its own shared slices (subgroup barriers). The workgroup shares
// the K/V tiles of FE_BC keys, fetched into registers one tile ahead: two workgroup barriers per tile. FA_SPLIT: Q, K, V and P
// in two f16 parts (three products each, near-f32 accuracy); otherwise single f16 products. SCALE multiplies the f32 scores; P
// is scaled by 2^12 before its f16 conversion. Head sizes are zero-padded to 16 (D % 8 == DV % 8 == 0, padded sizes <= 128; with
// FE_REG up to 512, FE_DS above 256);
// 32-wide subgroups.
// FE_CAUSAL (not FULL_ATTN; prefill against a KV cache): query t sees keys [0, n_past + t] (WINDOW: its last WINDOW of them, the
// block's key loop starting at its first query's window); a block reads only the keys its
// last query sees. One sequence: the blocks run longest first. MULTI_SEQ: grid z = runs (optionally paged, MIX_RUN).
// FE_REG (not FA_SPLIT; DeviceFeatures.MatrixMmaLayout): the fragments hold their elements in the layout of the 16x8x16 MMA
// instructions (lane l: rows l / 4 and l / 4 + 8, columns 2 (l % 4) + {0, 1, 8, 9}; accumulators and A operands alike), so
// the softmax runs on the score accumulators in registers (row maxima and sums over the four lanes of a row) and the
// probabilities become the A operands of P·V directly: no score, probability or row-factor staging in shared memory and no
// subgroup barriers in the key loop. FE_BC (keys per tile) 64 with FE_REG, else 32.
using namespace linalg;
#ifndef FE_CAUSAL
#define FE_CAUSAL 0
#endif
#ifndef FE_REG
#define FE_REG 0
#endif
#if FE_REG && FA_SPLIT
#error "FE_REG takes single f16 products"
#endif
// FE_BR: queries per workgroup, 16 per subgroup (64 unless FE_REG).
#ifndef FE_BR
#define FE_BR 64
#endif
#if !FE_REG && FE_BR != 64
#error "the staged softmax takes 64-query blocks"
#endif
#ifndef FE_BC
#define FE_BC (FE_REG ? 64 : 32)
#endif
// FE_DS (FE_REG only): subgroups sharing a 16-row group, each holding the query fragments of its share of the key dimensions and
// the output accumulators of its share of the value dimensions (2 for head sizes above 256, so the registers stay those of 256).
// Per tile the pair's partial scores are exchanged through feX and summed in the same order in both, so both run the same softmax.
#ifndef FE_DS
#define FE_DS 1
#endif
#if FE_DS != 1 && (!FE_REG || FE_DS != 2)
#error "FE_DS = 2 takes the register softmax"
#endif
#define FE_T (FE_BR * 2 * FE_DS)
#define FE_DP ((D + 15) / 16 * 16)
#define FE_DVP ((DV + 15) / 16 * 16)
#define FE_QS (FE_DP + 8)
#define FE_KS (FE_DP + 8)
#define FE_VS (FE_DVP + 8)
#define FE_PS (FE_BC + 8)
#define FE_SS (FE_BC + 4)                         // score row stride (floats): conflict-free fragment stores
#define FE_KT (FE_BC * FE_KS)                      // halves of one K part
#define FE_VT (FE_BC * FE_VS)                      // halves of one V part
// FE_QPASS (FE_REG only): Q is staged in passes of FE_BR / FE_QPASS rows, each pass's subgroups loading their fragments before the
// next one overwrites it, so that a block of queries whose staged rows would not fit the shared memory (64 queries of head size 512:
// 66 KiB) shares each key tile.
#ifndef FE_QPASS
#define FE_QPASS 1
#endif
#if FE_QPASS > 1 && (!FE_REG || FA_SPLIT || (FE_BR / FE_QPASS) % 16 != 0)
#error "FE_QPASS takes the register softmax and passes of whole row groups"
#endif
#define FE_QR (FE_BR / FE_QPASS)                    // query rows staged per pass
#define FE_QT (FE_QR * FE_QS)                      // halves of one staged Q part
#if FA_SPLIT
#define FE_PARTS 2                                 // f16 parts per value (hi, lo)
#else
#define FE_PARTS 1
#endif
#define FE_VO (FE_PARTS * FE_KT)                   // first half of the V tile (after the K parts)
#define FE_KVH (FE_PARTS * (FE_KT + FE_VT))
#define FE_BUF (FE_KVH > FE_PARTS * FE_QT ? FE_KVH : FE_PARTS * FE_QT)
#define FE_KC (FE_DP / 8)                           // 8-value chunks per K row
#define FE_VC (FE_DVP / 8)
#define FE_KPT ((FE_BC * FE_KC + FE_T - 1) / FE_T)  // K chunks per thread and tile
#define FE_VPT ((FE_BC * FE_VC + FE_T - 1) / FE_T)
#define FE_ND (FE_DP / 16)
#define FE_NO (FE_DVP / 16)
#define FE_NDW (FE_ND / FE_DS)                     // K steps of a subgroup's share of Q·Kᵀ
#define FE_NOW (FE_NO / FE_DS)                     // output fragments of a subgroup
#if FE_DS > 1 && (FE_ND % FE_DS != 0 || FE_NO % FE_DS != 0)
#error "FE_DS needs head sizes split into whole fragments"
#endif
#define FE_P_SCALE 4096.0
#if SL_TARGET_CUDA
#include "cuda_mma.slang"
typealias FeA = CuA16;
typealias FeB = CuB16;
typealias FeC = CuAccF;
#else
typealias FeA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias FeB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias FeC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
#endif

groupshared uint4 feBuf[FE_BUF / 8];          // staged Q (hi, lo), then the K / V tiles (hi, lo each): packed halves, 8 per uint4
#if !FE_REG
groupshared float feS[4 * 16 * FE_SS];        // per-subgroup scores, then output staging
groupshared half feP[4 * 16 * FE_PS];         // per-subgroup probabilities (hi parts)
#if FA_SPLIT
groupshared half fePl[4 * 16 * FE_PS];        // (lo parts)
#endif
groupshared float feC[4 * 256];               // per-subgroup row factors as 16 x 16 matrices
#endif
#if FA_SPLIT
// Power-of-two scales (split_scale_outside) that bring each query row, each key row of the staged tile and the staged V tile as
// a whole whose largest magnitude is outside [2^-6, 2^15) into [2^14, 2^15) before the f16 split, so neither part over- or
// underflows whatever the magnitudes (in range: scale 1, the unscaled rounding); the scores and
// the tile's P·V products are unscaled exactly. Maxima as f32 bits (non-negative floats order like their bits).
// One array of f32 bits (separate groupshared scalars miscompiled in fa): FE_BR query row scales, FE_BC key row maxima of the
// tile being staged, FE_BC inverse key row scales of the staged tile, the V tile's maximum and its inverse scale.
groupshared uint feScales[FE_BR + 2 * FE_BC + 2];
#define FE_QSCALE 0
#define FE_KMAX FE_BR
#define FE_KINV (FE_BR + FE_BC)
#define FE_VMAX (FE_BR + 2 * FE_BC)
#define FE_VINV (FE_BR + 2 * FE_BC + 1)
#endif

// 8 values into feBuf at half offset `index` (a multiple of 8) as one packed store (FA_SPLIT: hi parts there, lo parts at
// loIndex).
void fe_put8(float v[8], uint index, uint loIndex)
{
#if FA_SPLIT
    float h[8], l[8];
    [ForceUnroll]
    for (uint e = 0; e < 8; e++)
    {
        h[e] = asfloat((asuint(v[e]) + 0x1000u) & 0xFFFFE000u);
        l[e] = v[e] - h[e];
    }
    feBuf[index / 8] = pack_half8(h);
    feBuf[loIndex / 8] = pack_half8(l);
#else
    feBuf[index / 8] = pack_half8(v);
#endif
}

#if FA_SPLIT
// fe_put8 of v times the power of two `scale`.
void fe_put8_scaled(float v[8], float scale, uint index, uint loIndex)
{
    float w[8];
    [ForceUnroll] for (uint e = 0; e < 8; e++) w[e] = v[e] * scale;
    fe_put8(w, index, loIndex);
}

float max_abs8(float v[8])
{
    float a = 0;
    [ForceUnroll] for (uint e = 0; e < 8; e++) a = max(a, abs(v[e]));
    return a;
}
#endif

// Loads of K and V rows j0 .. j0 + FE_BC into registers (rows >= jEnd and padding columns are zero).
void fe_fetch(out Kv8 kb[FE_KPT], out Kv8 vb[FE_VPT], uint8_t* kc, uint8_t* vc, uint64_t kvRow, uint j0, uint jEnd, uint hk, uint tid)
{
    [ForceUnroll]
    for (uint k = 0; k < FE_KPT; k++)
    {
        uint i = tid + k * FE_T, r = i / FE_KC, c = (i % FE_KC) * 8;
        bool valid = i < FE_BC * FE_KC && j0 + r < jEnd && c < D;
        kb[k] = valid ? load_kv8(kc, K_ELEM(kv_row(kvRow, j0 + r), hk) + c) : zero_kv8();
    }
    [ForceUnroll]
    for (uint k = 0; k < FE_VPT; k++)
    {
        uint i = tid + k * FE_T, r = i / FE_VC, c = (i % FE_VC) * 8;
        bool valid = i < FE_BC * FE_VC && j0 + r < jEnd && c < DV;
        vb[k] = valid ? load_kv8(vc, V_ELEM(kv_row(kvRow, j0 + r), hk) + c) : zero_kv8();
    }
}

// Fetched rows into feBuf as f16 parts: K hi at 0, K lo at FE_KT, V hi at FE_VO, V lo at FE_VO + FE_VT. FA_SPLIT: the maxima of
// the key rows and the V tile first (feScales, zero on entry), then the scaled parts; the maxima are cleared after the caller's
// next barrier.
void fe_store(Kv8 kb[FE_KPT], Kv8 vb[FE_VPT], uint tid)
{
#if FA_SPLIT
    float vmax = 0;
    [ForceUnroll]
    for (uint k = 0; k < FE_KPT; k++)
    {
        uint i = tid + k * FE_T;
        if (i < FE_BC * FE_KC)
        {
            float v[8];
            unpack_kv8(kb[k], v);
            InterlockedMax(feScales[FE_KMAX + i / FE_KC], asuint(max_abs8(v)));
        }
    }
    [ForceUnroll]
    for (uint k = 0; k < FE_VPT; k++)
    {
        uint i = tid + k * FE_T;
        if (i < FE_BC * FE_VC)
        {
            float v[8];
            unpack_kv8(vb[k], v);
            vmax = max(vmax, max_abs8(v));
        }
    }
    vmax = WaveActiveMax(vmax);
    if (WaveIsFirstLane())
    {
        InterlockedMax(feScales[FE_VMAX], asuint(vmax));
    }
    GroupMemoryBarrierWithGroupSync();
    float vScale = split_scale_outside(asfloat(feScales[FE_VMAX]));
    [ForceUnroll]
    for (uint k = 0; k < FE_KPT; k++)
    {
        uint i = tid + k * FE_T;
        if (i < FE_BC * FE_KC)
        {
            float v[8];
            unpack_kv8(kb[k], v);
            float kScale = split_scale_outside(asfloat(feScales[FE_KMAX + i / FE_KC]));
            uint o = i / FE_KC * FE_KS + (i % FE_KC) * 8;
            fe_put8_scaled(v, kScale, o, FE_KT + o);
            if (i % FE_KC == 0)
            {
                feScales[FE_KINV + i / FE_KC] = asuint(1.0 / kScale);
            }
        }
    }
    [ForceUnroll]
    for (uint k = 0; k < FE_VPT; k++)
    {
        uint i = tid + k * FE_T;
        if (i < FE_BC * FE_VC)
        {
            float v[8];
            unpack_kv8(vb[k], v);
            uint o = FE_VO + i / FE_VC * FE_VS + (i % FE_VC) * 8;
            fe_put8_scaled(v, vScale, o, FE_VT + o);
        }
    }
    if (tid == 0)
    {
        feScales[FE_VINV] = asuint(1.0 / vScale);
    }
#else
    [ForceUnroll]
    for (uint k = 0; k < FE_KPT; k++)
    {
        uint i = tid + k * FE_T;
        if (i < FE_BC * FE_KC)
        {
            uint o = i / FE_KC * FE_KS + (i % FE_KC) * 8;
#if KV_FMT == FMT_F16 && !FA_SPLIT
            feBuf[o / 8] = kb[k].u;   // already packed halves
#else
            float v[8];
            unpack_kv8(kb[k], v);
            fe_put8(v, o, FE_KT + o);
#endif
        }
    }
    [ForceUnroll]
    for (uint k = 0; k < FE_VPT; k++)
    {
        uint i = tid + k * FE_T;
        if (i < FE_BC * FE_VC)
        {
            uint o = FE_VO + i / FE_VC * FE_VS + (i % FE_VC) * 8;
#if KV_FMT == FMT_F16 && !FA_SPLIT
            feBuf[o / 8] = vb[k].u;
#else
            float v[8];
            unpack_kv8(vb[k], v);
            fe_put8(v, o, FE_VT + o);
#endif
        }
    }
#endif
}

#if FE_REG
#if FE_DS > 1
groupshared float feX[(FE_T / 32) * (FE_BC / 16) * 8 * 32];   // partial scores of each subgroup (FE_DS = 2)
#endif
// One key tile of FE_REG (see above): scores S = Q·Kᵀ of the subgroup's 16 rows against the staged FE_BC keys, the online
// softmax on the accumulators (element e of a fragment: row g + 8 (e >> 2), key column 2c + (e & 1) + 8 ((e >> 1) & 1)), and
// o = o · corr + P·V with P converted element by element into A operands of the same layout. `part`: the subgroup's share of the
// head dimensions (FE_DS); every subgroup of the workgroup calls it once per tile (FE_DS = 2 synchronizes the workgroup).
void fe_reg_tile(FeA qa[FE_NDW], inout FeC o[FE_NOW], inout float2 m2, inout float2 l2, uint j0, uint jStop, uint sg, uint part, uint lane
#if FE_CAUSAL
    , uint nPast
#endif
#if HAS_MASK
    , float* maskRow0, float* maskRow1
#endif
    , uint tq0, uint qT)
{
    uint c2 = 2 * (lane % 4);
    FeC sc[FE_BC / 16];
    [ForceUnroll]
    for (uint kh = 0; kh < FE_BC / 16; kh++)
    {
        sc[kh] = FeC(0.0);
        [ForceUnroll]
        for (uint k = 0; k < FE_NDW; k++)
        {
            FeB b = FeB.Load<CoopMatMatrixLayout.ColumnMajor>(feBuf, (kh * 16 * FE_KS + (part * FE_NDW + k) * 16) / 8, FE_KS / 8);
            sc[kh] = coopMatMulAdd<float, false>(qa[k], b, sc[kh]);
        }
    }
#if FE_DS > 1
    // The partner's partial scores added (mine + its in part 0, its + mine in part 1: the same sum in both subgroups).
    [ForceUnroll]
    for (uint kh = 0; kh < FE_BC / 16; kh++)
    {
        [ForceUnroll] for (uint e = 0; e < 8; e++) feX[((sg * (FE_BC / 16) + kh) * 8 + e) * 32 + lane] = sc[kh][e];
    }
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint kh = 0; kh < FE_BC / 16; kh++)
    {
        [ForceUnroll]
        for (uint e = 0; e < 8; e++)
        {
            float x = feX[(((sg ^ 1) * (FE_BC / 16) + kh) * 8 + e) * 32 + lane];
            sc[kh][e] = part == 0 ? sc[kh][e] + x : x + sc[kh][e];
        }
    }
    GroupMemoryBarrierWithGroupSync();
#endif
    float sv[FE_BC / 16][8];
    float2 mx = float2(-1e30, -1e30);
    [ForceUnroll]
    for (uint kh = 0; kh < FE_BC / 16; kh++)
    {
        [ForceUnroll]
        for (uint e = 0; e < 8; e++)
        {
            uint hr = e >> 2, tq = tq0 + 8 * hr, j = j0 + kh * 16 + c2 + (e & 1) + 8 * ((e >> 1) & 1);
            float v = sc[kh][e] * SCALE;
#if FE_CAUSAL
            bool visible = tq < qT && j <= nPast + tq && (WINDOW == 0 || j + WINDOW > nPast + tq);
#else
            bool visible = tq < qT && j < jStop;
#endif
#if HAS_MASK
            if (visible) v += (hr == 0 ? maskRow0 : maskRow1)[j];
#endif
            sv[kh][e] = visible ? v : -1e30;
            if (hr == 0) mx.x = max(mx.x, sv[kh][e]); else mx.y = max(mx.y, sv[kh][e]);
        }
    }
    mx = max(mx, float2(WaveReadLaneAt(mx.x, lane ^ 1), WaveReadLaneAt(mx.y, lane ^ 1)));
    mx = max(mx, float2(WaveReadLaneAt(mx.x, lane ^ 2), WaveReadLaneAt(mx.y, lane ^ 2)));
    float2 mNew = max(m2, mx);
    float2 corr = float2(exp(m2.x - mNew.x), exp(m2.y - mNew.y));
    m2 = mNew;
    float2 ps = float2(0, 0);
    FeA pa[FE_BC / 16];
    [ForceUnroll]
    for (uint kh = 0; kh < FE_BC / 16; kh++)
    {
        [ForceUnroll]
        for (uint e = 0; e < 8; e++)
        {
            float mr = e < 4 ? m2.x : m2.y;
            float pe = sv[kh][e] > -1e29 ? exp(sv[kh][e] - mr) : 0.0;
            if (e < 4) ps.x += pe; else ps.y += pe;
            pa[kh][e] = half(pe * FE_P_SCALE);
        }
    }
    ps += float2(WaveReadLaneAt(ps.x, lane ^ 1), WaveReadLaneAt(ps.y, lane ^ 1));
    ps += float2(WaveReadLaneAt(ps.x, lane ^ 2), WaveReadLaneAt(ps.y, lane ^ 2));
    l2 = l2 * corr + ps;
    // The tile's P·V products accumulate from zero on the tensor cores and are added to the rescaled f32 output with an IEEE
    // add, so the tensor cores' truncating accumulation only rounds sums over one key tile.
    [ForceUnroll]
    for (uint i = 0; i < FE_NOW; i++)
    {
        FeC t = FeC(0.0);
        [ForceUnroll]
        for (uint kh = 0; kh < FE_BC / 16; kh++)
        {
            FeB v = FeB.Load<CoopMatMatrixLayout.RowMajor>(feBuf, (FE_VO + kh * 16 * FE_VS + (part * FE_NOW + i) * 16) / 8, FE_VS / 8);
            t = coopMatMulAdd<float, false>(pa[kh], v, t);
        }
        [ForceUnroll] for (uint e = 0; e < 8; e++) o[i][e] = o[i][e] * (e < 4 ? corr.x : corr.y) + t[e];
    }
}
#endif

[shader("compute")]
[numthreads(FE_T, 1, 1)]
void fe(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    uint h = gid.x, t0 = gid.y * FE_BR;
    uint tid = ltid.x, sg = tid / 32, lane = tid % 32;
    uint rg = sg / FE_DS, part = sg % FE_DS;   // the subgroup's 16 rows and its share of the head dimensions
    uint hk = h / (HQ / HKV);
#if FE_CAUSAL && MULTI_SEQ
    // Runs of a multi-sequence step: workgroup (h, blk, u) handles query tile blk of run u (grid z = runs); tiles past a run's
    // end exit. A run's tiles are those of the single-sequence form, so its rows' sums are the same.
#if PAGED
    g_pages = p.pages;
#endif
    int4 run = int4(p.runs[gid.z * 4], p.runs[gid.z * 4 + 1], p.runs[gid.z * 4 + 2], p.runs[gid.z * 4 + 3]);
    if (gid.y * FE_BR >= uint(run.z)) return;
#if MIX_RUN > 0
    if (uint(run.z) < MIX_RUN) return;
#endif
    uint nPast = uint(run.w);
    uint qT = uint(run.z), qRow0 = uint(run.y), keyBase = 0, L = nPast + min(qT, t0 + FE_BR);
    uint64_t kvRow = kv_base(uint(run.x));
#elif FE_CAUSAL
    uint nPast = p.rt[RT_N_PAST];
    t0 = ((p.T + FE_BR - 1) / FE_BR - 1 - gid.y) * FE_BR;   // the longest blocks first
    uint qT = p.T, qRow0 = 0, keyBase = 0, L = nPast + min(qT, t0 + FE_BR);
#elif SEQ_ROWS
    uint seq = gid.z / FA_KSPLIT;
    uint qT = SEQ_ROWS, qRow0 = seq * SEQ_ROWS, keyBase = seq * SEQ_KEYS, L = SEQ_KEYS;
#elif CROSS_ATTN
    uint qT = p.T, qRow0 = 0, keyBase = 0, L = p.S;
#else
    uint qT = p.T, qRow0 = 0, keyBase = 0, L = p.T;
#endif
#if !(FE_CAUSAL && MULTI_SEQ)
    uint64_t kvRow = uint64_t(keyBase) * HKV;
#endif
    uint ksplit = gid.z % FA_KSPLIT;
    uint kPer = ((L + FA_KSPLIT - 1) / FA_KSPLIT + FE_BC - 1) / FE_BC * FE_BC;
    uint jStart = min(L, ksplit * kPer), jStop = min(L, jStart + kPer);
#if FE_CAUSAL && WINDOW
    // Sliding window: the block's keys start at its first query's window (whole tiles, so the tiles match the unwindowed form's).
    jStart = max(jStart, nPast + t0 + 1 > WINDOW ? (nPast + t0 + 1 - WINDOW) / FE_BC * FE_BC : 0);
#endif

#if FE_REG
    // The fragment layout FE_REG relies on, verified once (as in matmul_skinny's sk_direct_layout): a wrong detection gives NaN
    // outputs instead of silently wrong ones.
    bool layoutOk;
    {
        for (uint i = tid; i < 128; i += FE_T)
        {
            feBuf[i / 4][i % 4] = f32tof16(float(2 * i)) | (f32tof16(float(2 * i + 1)) << 16);
        }
        GroupMemoryBarrierWithGroupSync();
        FeA probe = FeA.Load<CoopMatMatrixLayout.RowMajor>(feBuf, 0, 2);
        uint g = lane / 4, c = lane % 4;
        layoutOk = true;
        [ForceUnroll]
        for (uint e = 0; e < 8; e++)
        {
            layoutOk = layoutOk && uint(float(probe[e])) == (g + 8 * (e >> 2)) * 16 + 2 * c + (e & 1) + 8 * ((e >> 1) & 1);
        }
        layoutOk = WaveActiveAllTrue(layoutOk);
        GroupMemoryBarrierWithGroupSync();
    }
#endif
#if FA_SPLIT
    // Query row scales (and cleared tile maxima for the first fe_store).
    for (uint r = tid; r < FE_BR; r += FE_T)
    {
        uint t = t0 + r;
        uint64_t qi = uint64_t(qRow0 + min(t, qT - 1)) * QS0 + uint64_t(h) * QS1;
        float a = 0;
        for (uint d = 0; d < D; d++) a = max(a, t < qT ? abs(p.q[qi + d]) : 0.0);
        feScales[FE_QSCALE + r] = asuint(split_scale_outside(a));
    }
    for (uint i = tid; i < FE_BC; i += FE_T) feScales[FE_KMAX + i] = 0;
    if (tid == 0) feScales[FE_VMAX] = 0;
    GroupMemoryBarrierWithGroupSync();
#endif
    // Q (unscaled; FA_SPLIT: times the row scale) through shared memory into register fragments, FE_QR rows per pass.
    Kv8 kb[FE_KPT];
    Kv8 vb[FE_VPT];
    FeA qa[FE_NDW];
#if FA_SPLIT
    FeA qal[FE_ND];
#endif
    for (uint pass = 0; pass < FE_QPASS; pass++)
    {
        if (pass > 0)
        {
            GroupMemoryBarrierWithGroupSync();   // the previous pass's rows are read
        }

        for (uint i = tid; i < FE_QR * FE_KC; i += FE_T)
        {
            uint r = i / FE_KC, d0 = (i % FE_KC) * 8, t = t0 + pass * FE_QR + r;
            uint64_t qi = uint64_t(qRow0 + min(t, qT - 1)) * QS0 + uint64_t(h) * QS1 + d0;
            float x[8];
            [ForceUnroll] for (uint e = 0; e < 8; e++) x[e] = t < qT && d0 + e < D ? p.q[qi + e] : 0.0;
#if FA_SPLIT
            fe_put8_scaled(x, asfloat(feScales[FE_QSCALE + pass * FE_QR + r]), r * FE_QS + d0, FE_QT + r * FE_QS + d0);
#else
            fe_put8(x, r * FE_QS + d0, FE_QT + r * FE_QS + d0);
#endif
        }
        if (pass == 0)
        {
            fe_fetch(kb, vb, p.kc, p.vc, kvRow, jStart, jStop, hk, tid);
        }
        GroupMemoryBarrierWithGroupSync();
        if (rg * 16 / FE_QR == pass)
        {
            uint qr = rg * 16 - pass * FE_QR;
            [ForceUnroll]
            for (uint k = 0; k < FE_NDW; k++)
            {
                qa[k] = FeA.Load<CoopMatMatrixLayout.RowMajor>(feBuf, (qr * FE_QS + (part * FE_NDW + k) * 16) / 8, FE_QS / 8);
#if FA_SPLIT
                qal[k] = FeA.Load<CoopMatMatrixLayout.RowMajor>(feBuf, (FE_QT + sg * 16 * FE_QS + k * 16) / 8, FE_QS / 8);
#endif
            }
        }
    }
    FeC o[FE_NOW];
    [ForceUnroll] for (uint i = 0; i < FE_NOW; i++) o[i] = FeC(0.0);

#if FE_REG
    // Lane rows g and g + 8 of the subgroup's 16 (g = lane / 4): m2 / l2 hold their running maxima and sums.
    uint tq0 = t0 + rg * 16 + lane / 4;
    float2 m2 = float2(-1e30, -1e30), l2 = float2(0, 0);
#if HAS_MASK
#if CROSS_ATTN
    float* maskRow0 = p.mask + mask_row(p.T, p.S, qRow0 + min(tq0, qT - 1)) + keyBase;
    float* maskRow1 = p.mask + mask_row(p.T, p.S, qRow0 + min(tq0 + 8, qT - 1)) + keyBase;
#else
    float* maskRow0 = p.mask + mask_row(p.T, 0, qRow0 + min(tq0, qT - 1)) + keyBase;
    float* maskRow1 = p.mask + mask_row(p.T, 0, qRow0 + min(tq0 + 8, qT - 1)) + keyBase;
#endif
#endif
#else
    // Softmax: two adjacent lanes per row, 16 key columns each.
    uint row = lane / 2, col0 = (lane % 2) * 16;
    uint tq = t0 + sg * 16 + row;
    bool rowValid = tq < qT;
#if HAS_MASK
#if CROSS_ATTN
    float* maskRow = p.mask + mask_row(p.T, p.S, qRow0 + min(tq, qT - 1)) + keyBase;
#else
    float* maskRow = p.mask + mask_row(p.T, 0, qRow0 + min(tq, qT - 1)) + keyBase;
#endif
#endif
    float m = -1e30, l = 0;
    uint sBase = sg * 16 * FE_SS, pBase = sg * 16 * FE_PS, cBase = sg * 256;
#if FA_SPLIT
    float qInv = 1.0 / asfloat(feScales[FE_QSCALE + sg * 16 + row]);
#endif
#endif

    for (uint j0 = jStart; j0 < jStop; j0 += FE_BC)
    {
#ifndef FE_EXP
#define FE_EXP 0
#endif
#if FE_EXP != 1
        GroupMemoryBarrierWithGroupSync();   // the previous tile (or the staged Q) is no longer read
        fe_store(kb, vb, tid);
        GroupMemoryBarrierWithGroupSync();
#if FA_SPLIT
        // The maxima were read by fe_store before this barrier; the next tile's are gathered after the next one.
        for (uint i = tid; i < FE_BC; i += FE_T) feScales[FE_KMAX + i] = 0;
        if (tid == 0) feScales[FE_VMAX] = 0;
#endif
        if (j0 + FE_BC < jStop)
        {
            fe_fetch(kb, vb, p.kc, p.vc, kvRow, j0 + FE_BC, jStop, hk, tid);   // in flight during this tile
        }
#endif

#if FE_REG
        fe_reg_tile(qa, o, m2, l2, j0, jStop, sg, part, lane
#if FE_CAUSAL
            , nPast
#endif
#if HAS_MASK
            , maskRow0, maskRow1
#endif
            , tq0, qT);
    }
#else
        // S (16 rows x FE_BC keys) of this subgroup.
        [ForceUnroll]
        for (uint kh = 0; kh < FE_BC / 16; kh++)
        {
            FeC s = FeC(0.0);
            [ForceUnroll]
            for (uint k = 0; k < FE_ND; k++)
            {
                FeB b = FeB.Load<CoopMatMatrixLayout.ColumnMajor>(feBuf, (kh * 16 * FE_KS + k * 16) / 8, FE_KS / 8);
#if FA_SPLIT
                FeB bl = FeB.Load<CoopMatMatrixLayout.ColumnMajor>(feBuf, (FE_KT + kh * 16 * FE_KS + k * 16) / 8, FE_KS / 8);
                s = coopMatMulAdd<float, false>(qal[k], b, s);
                s = coopMatMulAdd<float, false>(qa[k], bl, s);
#endif
                s = coopMatMulAdd<float, false>(qa[k], b, s);
            }
            s.Store<CoopMatMatrixLayout.RowMajor>(feS, sBase + kh * 16, FE_SS);
        }
        GroupMemoryBarrierWithWaveSync();

        // Element e of a lane is key column col0 + ((e + row) & 15): the rotation spreads the 32 lanes' reads and writes over
        // distinct banks.
        float sv[16];
        float mx = -1e30;
        [ForceUnroll]
        for (uint e = 0; e < 16; e++)
        {
            uint c = col0 + ((e + row) & 15), j = j0 + c;
#if FA_SPLIT
            float sc = feS[sBase + row * FE_SS + c] * (qInv * asfloat(feScales[FE_KINV + c])) * SCALE;
#else
            float sc = feS[sBase + row * FE_SS + c] * SCALE;
#endif
#if FE_CAUSAL
            bool visible = rowValid && j <= nPast + tq && (WINDOW == 0 || j + WINDOW > nPast + tq);
#else
            bool visible = rowValid && j < jStop;
#endif
#if HAS_MASK
            if (visible) sc += maskRow[j];
#endif
            sv[e] = visible ? sc : -1e30;
            mx = max(mx, sv[e]);
        }
        mx = max(mx, WaveReadLaneAt(mx, lane ^ 1));
        float mNew = max(m, mx);
        float corr = exp(m - mNew);
        m = mNew;
        float ps = 0;
        [ForceUnroll]
        for (uint e = 0; e < 16; e++)
        {
            uint c = col0 + ((e + row) & 15);
            float pe = sv[e] > -1e29 ? exp(sv[e] - m) : 0.0;
            ps += pe;
#if FA_SPLIT
            half hi, lo;
            fa_split(pe * FE_P_SCALE, hi, lo);
            feP[pBase + row * FE_PS + c] = hi;
            fePl[pBase + row * FE_PS + c] = lo;
#else
            feP[pBase + row * FE_PS + c] = half(pe * FE_P_SCALE);
#endif
        }
        ps += WaveReadLaneAt(ps, lane ^ 1);
        l = l * corr + ps;
        [ForceUnroll] for (uint e = 0; e < 8; e++) feC[cBase + row * 16 + (lane % 2) * 8 + ((e + row / 2) & 7)] = corr;
        GroupMemoryBarrierWithWaveSync();

        // The tile's P·V products accumulate from zero on the tensor cores and are added to the rescaled f32 output with
        // an IEEE add, so the tensor cores' truncating accumulation only rounds sums over one key tile.
        FeC cm = FeC.Load<CoopMatMatrixLayout.RowMajor>(feC, cBase, 16);
        FeA pa[FE_BC / 16];
#if FA_SPLIT
        FeA pal[FE_BC / 16];
#endif
        [ForceUnroll]
        for (uint kh = 0; kh < FE_BC / 16; kh++)
        {
            pa[kh] = FeA.Load<CoopMatMatrixLayout.RowMajor>(feP, pBase + kh * 16, FE_PS);
#if FA_SPLIT
            pal[kh] = FeA.Load<CoopMatMatrixLayout.RowMajor>(fePl, pBase + kh * 16, FE_PS);
#endif
        }
        [ForceUnroll]
        for (uint i = 0; i < FE_NO; i++)
        {
            FeC t = FeC(0.0);
            [ForceUnroll]
            for (uint kh = 0; kh < FE_BC / 16; kh++)
            {
                FeB v = FeB.Load<CoopMatMatrixLayout.RowMajor>(feBuf, (FE_VO + kh * 16 * FE_VS + i * 16) / 8, FE_VS / 8);
#if FA_SPLIT
                FeB vl = FeB.Load<CoopMatMatrixLayout.RowMajor>(feBuf, (FE_VO + FE_VT + kh * 16 * FE_VS + i * 16) / 8, FE_VS / 8);
                t = coopMatMulAdd<float, false>(pal[kh], v, t);
                t = coopMatMulAdd<float, false>(pa[kh], vl, t);
#endif
                t = coopMatMulAdd<float, false>(pa[kh], v, t);
            }
#if FA_SPLIT
            o[i] = o[i] * cm + t * FeC(asfloat(feScales[FE_VINV]));
#else
            o[i] = o[i] * cm + t;
#endif
        }
        GroupMemoryBarrierWithWaveSync();   // corr and P read before the next tile rewrites them
    }
#endif

#if FE_REG
    // Row factors as below; each lane stores its own elements (rows g, g + 8; columns 2c + {0, 1, 8, 9} of each fragment).
    {
        float2 f2 = float2(FA_KSPLIT > 1 ? 1.0 : (l2.x > 0 ? 1.0 / l2.x : 0.0), FA_KSPLIT > 1 ? 1.0 : (l2.y > 0 ? 1.0 / l2.y : 0.0)) * (1.0 / FE_P_SCALE);
        if (!layoutOk)
        {
            f2 = float2(asfloat(0x7FC00000u));
        }
        uint c2 = 2 * (lane % 4);
#if FA_KSPLIT > 1
        if (lane % 4 == 0)
        {
            [ForceUnroll]
            for (uint hr = 0; hr < 2; hr++)
            {
                uint t = tq0 + 8 * hr;
                if (t < qT)
                {
                    float* pr = p.part + ((uint64_t(ksplit) * p.T + qRow0 + t) * HQ + h) * (DV + 2);
                    pr[DV] = hr == 0 ? m2.x : m2.y;
                    pr[DV + 1] = hr == 0 ? l2.x : l2.y;
                }
            }
        }
#endif
        [ForceUnroll]
        for (uint i = 0; i < FE_NOW; i++)
        {
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
                uint t = tq0 + 8 * (e >> 2), d = (part * FE_NOW + i) * 16 + c2 + (e & 1) + 8 * ((e >> 1) & 1);
                if (t < qT && d < DV)
                {
                    float v = o[i][e] * (e < 4 ? f2.x : f2.y);
#if FA_KSPLIT > 1
                    p.part[((uint64_t(ksplit) * p.T + qRow0 + t) * HQ + h) * (DV + 2) + d] = v;
#else
                    p.o[(uint64_t(qRow0 + t) * HQ + h) * DV + d] = v;
#endif
                }
            }
        }
    }
#else
    // Row factors: 1 / (2^12 · l) for the output; 1 / 2^12 for unnormalized partials (part[range][row][h] = (o[DV], m, l)).
    float f = (FA_KSPLIT > 1 ? 1.0 : (l > 0 ? 1.0 / l : 0.0)) * (1.0 / FE_P_SCALE);
    [ForceUnroll] for (uint e = 0; e < 8; e++) feC[cBase + row * 16 + (lane % 2) * 8 + ((e + row / 2) & 7)] = f;
#if FA_KSPLIT > 1
    if (rowValid && lane % 2 == 0)
    {
        float* pr = p.part + ((uint64_t(ksplit) * p.T + qRow0 + tq) * HQ + h) * (DV + 2);
        pr[DV] = m;
        pr[DV + 1] = l;
    }
#endif
    GroupMemoryBarrierWithWaveSync();
    FeC fm = FeC.Load<CoopMatMatrixLayout.RowMajor>(feC, cBase, 16);
    [ForceUnroll]
    for (uint i = 0; i < FE_NO; i++)
    {
        (o[i] * fm).Store<CoopMatMatrixLayout.RowMajor>(feS, sBase, 16);
        GroupMemoryBarrierWithWaveSync();
        [ForceUnroll]
        for (uint e = 0; e < 8; e++)
        {
            // Two rows of 16 consecutive values per store (coalesced).
            uint idx = e * 32 + lane, r = idx / 16, d = i * 16 + idx % 16, t = t0 + sg * 16 + r;
            if (t < qT && d < DV)
            {
#if FA_KSPLIT > 1
                p.part[((uint64_t(ksplit) * p.T + qRow0 + t) * HQ + h) * (DV + 2) + d] = feS[sBase + idx];
#else
                p.o[(uint64_t(qRow0 + t) * HQ + h) * DV + d] = feS[sBase + idx];
#endif
            }
        }
        GroupMemoryBarrierWithWaveSync();
    }
#endif
}
#endif

#ifdef KERNEL_FES
// ---------------------------------------------------------------------------------------------
// Flash attention for bidirectional encoders without matrix units (the f32 counterpart of `fe`, same forms: FULL_ATTN with
// optional HAS_MASK, SEQ_ROWS, CROSS_ATTN, FA_KSPLIT). A workgroup of FS_BR threads handles FS_BR queries of one head, one per
// thread: the scaled query and the output accumulators stay in registers, and the K/V tiles of FS_BC keys pass through shared
// memory, where every thread reads the same row (broadcast). f32 throughout (strict precision). D, DV <= 64, multiples of 8.
#define FS_BR 64
#define FS_BC 32
#define FS_KC (D / 8)
#define FS_VC (DV / 8)
groupshared float4 fsK[FS_BC * D / 4];
groupshared float4 fsV[FS_BC * DV / 4];

[shader("compute")]
[numthreads(FS_BR, 1, 1)]
void fes(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    uint h = gid.x, tid = ltid.x;
    uint hk = h / (HQ / HKV);
#if SEQ_ROWS
    uint seq = gid.z / FA_KSPLIT;
    uint qT = SEQ_ROWS, qRow0 = seq * SEQ_ROWS, keyBase = seq * SEQ_KEYS, L = SEQ_KEYS;
#elif CROSS_ATTN
    uint qT = p.T, qRow0 = 0, keyBase = 0, L = p.S;
#else
    uint qT = p.T, qRow0 = 0, keyBase = 0, L = p.T;
#endif
#if !(FE_CAUSAL && MULTI_SEQ)
    uint64_t kvRow = uint64_t(keyBase) * HKV;
#endif
    uint ksplit = gid.z % FA_KSPLIT;
    uint kPer = ((L + FA_KSPLIT - 1) / FA_KSPLIT + FS_BC - 1) / FS_BC * FS_BC;
    uint jStart = min(L, ksplit * kPer), jStop = min(L, jStart + kPer);
    uint tq = gid.y * FS_BR + tid;
    bool rowValid = tq < qT;
    uint tr = qRow0 + min(tq, qT - 1);

    float4 q[D / 4];
    float* qr = p.q + uint64_t(tr) * QS0 + uint64_t(h) * QS1;
    [ForceUnroll] for (uint i = 0; i < D / 4; i++) q[i] = float4(qr[i * 4], qr[i * 4 + 1], qr[i * 4 + 2], qr[i * 4 + 3]) * SCALE;
#if HAS_MASK
#if CROSS_ATTN
    float* maskRow = p.mask + mask_row(p.T, p.S, tr) + keyBase;
#else
    float* maskRow = p.mask + mask_row(p.T, 0, tr) + keyBase;
#endif
#endif
    float4 acc[DV / 4];
    [ForceUnroll] for (uint i = 0; i < DV / 4; i++) acc[i] = float4(0.0);
    float m = -1e30, l = 0;

    for (uint j0 = jStart; j0 < jStop; j0 += FS_BC)
    {
        GroupMemoryBarrierWithGroupSync();   // the previous tile is no longer read
        for (uint i = tid; i < FS_BC * FS_KC; i += FS_BR)
        {
            uint r = i / FS_KC, c = (i % FS_KC) * 8;
            float v[8];
            unpack_kv8(j0 + r < jStop ? load_kv8(p.kc, K_ELEM(kv_row(kvRow, j0 + r), hk) + c) : zero_kv8(), v);
            fsK[(r * D + c) / 4] = float4(v[0], v[1], v[2], v[3]);
            fsK[(r * D + c) / 4 + 1] = float4(v[4], v[5], v[6], v[7]);
        }
        for (uint i = tid; i < FS_BC * FS_VC; i += FS_BR)
        {
            uint r = i / FS_VC, c = (i % FS_VC) * 8;
            float v[8];
            unpack_kv8(j0 + r < jStop ? load_kv8(p.vc, V_ELEM(kv_row(kvRow, j0 + r), hk) + c) : zero_kv8(), v);
            fsV[(r * DV + c) / 4] = float4(v[0], v[1], v[2], v[3]);
            fsV[(r * DV + c) / 4 + 1] = float4(v[4], v[5], v[6], v[7]);
        }
        GroupMemoryBarrierWithGroupSync();

        float sc[FS_BC];
        float mx = -1e30;
        [ForceUnroll]
        for (uint c = 0; c < FS_BC; c++)
        {
            float4 d4 = float4(0.0);
            [ForceUnroll] for (uint i = 0; i < D / 4; i++) d4 = fma(q[i], fsK[c * (D / 4) + i], d4);
            float v = d4.x + d4.y + d4.z + d4.w;
#if HAS_MASK
            if (j0 + c < jStop) v += maskRow[j0 + c];
#endif
            sc[c] = j0 + c < jStop ? v : -1e30;
            mx = max(mx, sc[c]);
        }
        float mNew = max(m, mx);
        float corr = exp(m - mNew);
        m = mNew;
        l *= corr;
        [ForceUnroll] for (uint i = 0; i < DV / 4; i++) acc[i] *= corr;
        [ForceUnroll]
        for (uint c = 0; c < FS_BC; c++)
        {
            float pc = sc[c] > -1e29 ? exp(sc[c] - m) : 0.0;
            l += pc;
            [ForceUnroll] for (uint i = 0; i < DV / 4; i++) acc[i] = fma(float4(pc), fsV[c * (DV / 4) + i], acc[i]);
        }
    }

    if (!rowValid)
    {
        return;
    }
#if FA_KSPLIT > 1
    float* orow = p.part + ((uint64_t(ksplit) * p.T + qRow0 + tq) * HQ + h) * (DV + 2);
    orow[DV] = m;
    orow[DV + 1] = l;
    float f = 1.0;
#else
    float* orow = p.o + (uint64_t(qRow0 + tq) * HQ + h) * DV;
    float f = l > 0 ? 1.0 / l : 0.0;
#endif
    [ForceUnroll]
    for (uint i = 0; i < DV / 4; i++)
    {
        float4 v = acc[i] * f;
        orow[i * 4] = v.x;
        orow[i * 4 + 1] = v.y;
        orow[i * 4 + 2] = v.z;
        orow[i * 4 + 3] = v.w;
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
// FM_SEPV: separate key and value caches of one head size (D == DV, f16 / f32 / q8_0 / q4_0 caches: Gemma 4's and Qwen3.5's head sizes 256 /
// 512 on targets without the hand-written CUDA kernel): the decode rows' key and value tiles are fetched and staged side by side
// (fmV), FM_BC keys per tile (16 or 32, so both tiles fit the shared memory).
using namespace linalg;
#define FM_BR 16
#ifndef FM_BC
#define FM_BC 32
#endif
#ifndef FM_SEPV
#define FM_SEPV 0
#endif
#define FM_NH (FM_BC / 16)                          // 16-key halves of a tile
#define FM_CPL (FM_BC / 8)                          // key columns per lane in the softmax (8 lanes per row)
#if (FM_BC != 16 && FM_BC != 32) || (FM_SEPV && DV != D)
#error "fa_mqa: unsupported tile"
#endif
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
#if SL_TARGET_CUDA
#include "cuda_mma.slang"
typealias FmA = CuA16;
typealias FmB = CuB16;
typealias FmC = CuAccF;
#else
typealias FmA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias FmB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias FmC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
#endif

groupshared uint4 fmKV[FM_BC * FM_KS / 8];   // key/value tile as packed f16 x 8; holds the query tile before the key loop
#if FM_SEPV
groupshared uint4 fmV[FM_BC * FM_KS / 8];    // value tile (FM_SEPV)
#endif
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
void fm_fetch(out FmTile buf, uint8_t* cache, uint64_t kvRow, uint j0, uint jEnd, uint hk, uint tid, uint rowElems = D)
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
            uint64_t elem = (kv_row(kvRow, j) + hk) * rowElems + (i % FM_BPR) * 32;
            uint64_t s = (elem >> 5) * KV_BLOCK_BYTES;
            uint* w = (uint*)byteAt(cache, s & ~uint64_t(3));
            [ForceUnroll] for (uint n = 0; n + 1 < FM_NW; n++) buf.w[k * FM_NW + n] = w[n];
            if (KV_FP8)
            {
                buf.w[k * FM_NW + FM_NW - 1] = w[FM_NW - 1];   // 4-byte aligned blocks of 9 whole words
            }
            else if ((s & 2) != 0)
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
        buf.c[k] = valid ? load_kv8(cache, (kv_row(kvRow, j) + hk) * rowElems + c) : zero_kv8();
    }
#endif
}

// Stores fetched rows into fmKV (FM_SEPV values: fmV) as f16 (one 16-byte store per 8 values).
void fm_store(FmTile buf, uint tid, bool values = false)
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

            float d = KV_FP8 ? asfloat(v[0]) : f16tof32(v[0] & 0xFFFF);
            uint dst = i / FM_BPR * (FM_KS / 8) + i % FM_BPR * 4;
            [ForceUnroll]
            for (uint c = 0; c < 4; c++)
            {
                float x[8];
                [ForceUnroll]
                for (uint e = 0; e < 8; e++)
                {
                    uint el = c * 8 + e;
#if KV_FP8
                    uint b = 4 + el;
                    x[e] = d * e4m3_value(v[b >> 2] >> ((b & 3) * 8));
#elif KV_FMT == FMT_Q8_0
                    uint b = 2 + el;
                    x[e] = d * float(int(((v[b >> 2] >> ((b & 3) * 8)) & 0xFF) << 24) >> 24);
#else
                    uint b = 2 + (el & 15);
                    x[e] = d * (float((v[b >> 2] >> ((b & 3) * 8 + (el >= 16 ? 4 : 0))) & 0xF) - 8.0);
#endif
                }
#if FM_SEPV
                if (values)
                {
                    fmV[dst + c] = pack_half8(x);
                    continue;
                }
#endif
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
#if FM_SEPV
            if (values)
            {
                fmV[i / FM_CPR * (FM_KS / 8) + i % FM_CPR] = kv8_to_half(buf.c[k]);
                continue;
            }
#endif
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
#if MULTI_SEQ && MIX_RUN > 0
    if (p.seq[t * 4 + 2] >= MIX_RUN)
    {
        return;   // rows of long runs: the flash kernel of the mixed step
    }
#endif

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
#if FM_SEPV
    FmTile nextV;
    fm_fetch(nextV, p.vc, kvRow, jBeg, jEnd, hk, tid, VRS);
#endif
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

    // Softmax layout: 8 adjacent lanes per row, FM_CPL key columns each.
    uint row = tid / 8, sub = tid % 8;
    bool rowValid = g0 + row < FM_G;
    float m = -1e30, l = 0;

    for (uint j0 = jBeg; j0 < jEnd; j0 += FM_BC)
    {
        fm_store(next, tid);
#if FM_SEPV
        fm_store(nextV, tid, true);
#endif
        GroupMemoryBarrierWithGroupSync();
        if (j0 + FM_BC < jEnd)
        {
            fm_fetch(next, p.kc, kvRow, j0 + FM_BC, jEnd, hk, tid);   // in flight during this tile's products
#if FM_SEPV
            fm_fetch(nextV, p.vc, kvRow, j0 + FM_BC, jEnd, hk, tid, VRS);
#endif
        }

        // Partial scores of this subgroup's D chunks for the 16-key halves.
        {
            FmC sh[FM_NH];
            [ForceUnroll] for (uint h = 0; h < FM_NH; h++) sh[h] = FmC(0.0);
            [ForceUnroll]
            for (uint c = 0; c < FM_KPS; c++)
            {
                uint k = sg * FM_KPS + c;
                if (k < FM_KCH)
                {
                    [ForceUnroll]
                    for (uint h = 0; h < FM_NH; h++)
                    {
                        FmB b = FmB.Load<CoopMatMatrixLayout.ColumnMajor>(fmKV, h * 2 * FM_KS + k * 2, FM_KS / 8);
                        sh[h] = coopMatMulAdd<float, false>(qa[c], b, sh[h]);
                    }
                }
            }
            [ForceUnroll] for (uint h = 0; h < FM_NH; h++) sh[h].Store<CoopMatMatrixLayout.RowMajor>(fmS, sg * FM_SS + h * 16, FM_BC);
        }
        GroupMemoryBarrierWithGroupSync();

        float sv[FM_CPL];
        float mx = -1e30;
        [ForceUnroll]
        for (uint e = 0; e < FM_CPL; e++)
        {
            uint c = sub * FM_CPL + e;
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
        for (uint e = 0; e < FM_CPL; e++)
        {
            float pe = sv[e] > -1e29 ? exp(sv[e] - m) : 0.0;
            ps += pe;
            fmP[row * FM_PS + sub * FM_CPL + e] = half(pe);
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

        FmA ph[FM_NH];
        [ForceUnroll] for (uint h = 0; h < FM_NH; h++) ph[h] = FmA.Load<CoopMatMatrixLayout.RowMajor>(fmP, h * 16, FM_PS);
        [ForceUnroll]
        for (uint i = 0; i < FM_OPS; i++)
        {
            uint ot = sg * FM_OPS + i;
            if (ot < FM_OT)
            {
                [ForceUnroll]
                for (uint h = 0; h < FM_NH; h++)
                {
#if FM_SEPV
                    FmB v = FmB.Load<CoopMatMatrixLayout.RowMajor>(fmV, h * 2 * FM_KS + ot * 2, FM_KS / 8);
#else
                    FmB v = FmB.Load<CoopMatMatrixLayout.RowMajor>(fmKV, h * 2 * FM_KS + ot * 2, FM_KS / 8);
#endif
                    o[i] = coopMatMulAdd<float, false>(ph[h], v, o[i]);
                }
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

#if DEC_FUSED_COMBINE
#if DEC_WG != DC_WG
#error "the fused combine runs decode_combine_row with the decode workgroup"
#endif
groupshared uint dLast;
#endif
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
#if MULTI_SEQ && MIX_RUN > 0
    if (p.seq[t * 4 + 2] >= MIX_RUN)
    {
        return;
    }
#endif

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
#if HAS_SCORE_BIAS
                s[i][g] += score_bias(h0 + g, c0 + i * DEC_NRG + rg, L - 1);
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
#if DEC_FUSED_COMBINE
    // DEC_FUSED_COMBINE: the last of a row's ns key ranges to finish merges the partials of its HPW heads, as decode_combine
    // does (same code, same thread count), instead of a combine launch behind a barrier. Each workgroup publishes its partials
    // (device-scope release) before counting its arrival; the last one acquires them and resets the counter for the next launch.
    if (!direct)
    {
        DeviceMemoryBarrier();
        GroupMemoryBarrierWithGroupSync();
        if (tid == 0)
        {
            uint arrived;
            InterlockedAdd(p.arrivals[t * DEC_GROUPS + gid.x], 1u, arrived);
            dLast = arrived + 1 == ns ? 1u : 0u;
        }
        GroupMemoryBarrierWithGroupSync();
        if (dLast != 0)
        {
            DeviceMemoryBarrier();
            for (uint g = 0; g < HPW; g++)
            {
                decode_combine_row(p.part, p.o, p.sinks, h0 + g, t, ns, p.T, tid);
                GroupMemoryBarrierWithGroupSync();
            }
            if (tid == 0)
            {
                p.arrivals[t * DEC_GROUPS + gid.x] = 0;
            }
        }
    }
#endif
}
#endif

#endif
