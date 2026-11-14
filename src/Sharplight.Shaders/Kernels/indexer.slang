// Lightning-indexer key selection (IndexerTopK: DeepSeek sparse attention and its block / pooled group variants) in two launches over the query
// rows [row0, row0 + rows) of a launch:
//   index_scores: the score of every candidate group g of GROUP positions the row sees, into scores[((t - row0) * SH + s) * NG + g]:
//                 GROUP_POOLED: the score of the group's pooled key (per channel, softmax over the members of gate + position gate weights the
//                 keys); otherwise the best score of its visible positions. Position score: sum_h w[t][h] * relu(q[t][h] . key), or with
//                 PER_HEAD q[t][s] . key for selection head s. The row's latest LOCAL groups score +inf.
//                 Grid (ceil(NG / WG), rows); one thread per group (the row's queries and head weights are uniform loads).
//   index_select: one workgroup per (row, selection head); the BUDGET / GROUP best groups (radix select, then an ordered collection: groups above
//                 the threshold in ascending order, then the ties at it in ascending order; a row with at most that many candidate groups keeps
//                 all of them), listed with their positions (-1 where the row cannot see), then with TAIL the visible positions of the row's
//                 incomplete group, -1 padded to WIDTH entries.
// q: [T, H, D] dense f32; keys: the cache [ctx, 1, D] (pooled: [ctx, 1, 2D], key | gate) in KEY_F16 ? f16 : f32; w: [T, H] dense f32 (not PER_HEAD);
// gate: [GROUP, D] dense f32 (GROUP_POOLED); out: [T, SH, WIDTH] i32.
// Macros: H, D (multiple of 8), BUDGET, GROUP, GROUP_POOLED, LOCAL, TAIL, PER_HEAD, WIDTH, CTX (positions per row: the per-sequence context),
// KEY_F16, WG, RT_N_PAST; MULTI_SEQ / SLOT_CTX and PAGED / PAGE_BS / PAGE_MB for the multi-sequence and paged forms (row t sees positions
// [0, seq[t].position] of its slot).
#include "common.slang"
#include "radix_select.slang"

#if PER_HEAD
#define SH H
#else
#define SH 1
#endif
#define NG ((CTX + GROUP - 1) / GROUP)
#define KG (BUDGET / GROUP)
#if GROUP_POOLED
#define ROW_VALUES (2 * D)
#else
#define ROW_VALUES D
#endif

struct IndexerParams
{
    float* q;
    uint8_t* keys;
    float* w;
    float* gate;
    float* scores;
    int* out;
    uint* rt;
#if MULTI_SEQ
    int* seq;    // [T, 4] (slot, position, run length, run index) per query row
#endif
#if PAGED
    int* pages;  // [slots, PAGE_MB] block table of the paged cache
#endif
    uint T;
    uint row0;   // first query row of this launch
};

// Visible positions of query row t and its slot.
uint visible_keys(IndexerParams p, uint t, out uint slot)
{
#if MULTI_SEQ
    slot = uint(p.seq[t * 4]);
    return uint(p.seq[t * 4 + 1]) + 1;
#else
    slot = 0;
    return p.rt[RT_N_PAST] + t + 1;
#endif
}

// Candidate groups of a row that sees L positions: complete groups when pooled, otherwise every group with a visible position.
uint candidate_groups(uint L)
{
#if GROUP_POOLED
    return L / GROUP;
#else
    return (L + GROUP - 1) / GROUP;
#endif
}

uint64_t key_row(IndexerParams p, uint slot, uint j)
{
#if MULTI_SEQ
#if PAGED
    return seq_row(p.pages, slot, j);
#else
    return seq_row(p.seq, slot, j);   // contiguous slots: the pages argument is unused
#endif
#else
    return j;
#endif
}

void load_key8(IndexerParams p, uint64_t row, uint chunk, out float v[8])
{
#if KEY_F16
    load8_f16(p.keys, ROW_VALUES * 2, uint(row), chunk, v);
#else
    load8_f32(p.keys, ROW_VALUES * 4, uint(row), chunk, v);
#endif
}

#ifdef KERNEL_INDEX_SCORES
// Per-head dot products of the row's queries with the key of cache row `row`, added to acc.
void dot_heads(IndexerParams p, float* qr, uint64_t row, inout float acc[H])
{
    for (uint c = 0; c < D / 8; c++)
    {
        float k[8];
        load_key8(p, row, c, k);
        [ForceUnroll]
        for (uint h = 0; h < H; h++)
        {
            float s = 0;
            [ForceUnroll] for (uint e = 0; e < 8; e++) s += qr[h * D + c * 8 + e] * k[e];
            acc[h] += s;
        }
    }
}

// The scores of one key from its per-head dot products: per selection head, or the weighted sum of the positive ones.
void head_scores(IndexerParams p, uint t, float acc[H], out float score[SH])
{
#if PER_HEAD
    [ForceUnroll] for (uint h = 0; h < H; h++) score[h] = acc[h];
#else
    float s = 0;
    [ForceUnroll] for (uint h = 0; h < H; h++) s += p.w[uint64_t(t) * H + h] * max(acc[h], 0.0);
    score[0] = s;
#endif
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void index_scores(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform IndexerParams p)
{
    uint t = gid.y + p.row0;
    uint tid = ltid.x;
    uint slot;
    uint L = visible_keys(p, t, slot);
    uint g = gid.x * WG + tid;
    if (g >= candidate_groups(L))
    {
        return;
    }

    float* qr = p.q + uint64_t(t) * (H * D);
    float best[SH];
#if GROUP_POOLED
    float acc[H];
    [ForceUnroll] for (uint h = 0; h < H; h++) acc[h] = 0;
    for (uint c = 0; c < D / 8; c++)
    {
        // Online softmax over the members, per channel, of gate + position gate weighting the keys.
        float m[8], l[8], pooled[8];
        [ForceUnroll] for (uint e = 0; e < 8; e++) { m[e] = -1e30; l[e] = 0; pooled[e] = 0; }
        for (uint i = 0; i < GROUP; i++)
        {
            uint64_t row = key_row(p, slot, g * GROUP + i);
            float k[8], gt[8];
            load_key8(p, row, c, k);
            load_key8(p, row, D / 8 + c, gt);
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
                float logit = gt[e] + p.gate[i * D + c * 8 + e];
                float mNew = max(m[e], logit);
                float corr = exp(m[e] - mNew);
                float pe = exp(logit - mNew);
                l[e] = l[e] * corr + pe;
                pooled[e] = pooled[e] * corr + pe * k[e];
                m[e] = mNew;
            }
        }
        [ForceUnroll]
        for (uint h = 0; h < H; h++)
        {
            float s = 0;
            [ForceUnroll] for (uint e = 0; e < 8; e++) s += qr[h * D + c * 8 + e] * (pooled[e] / l[e]);
            acc[h] += s;
        }
    }
    head_scores(p, t, acc, best);
#else
    [ForceUnroll] for (uint s = 0; s < SH; s++) best[s] = asfloat(0xFF800000u);
    uint end = min(L, (g + 1) * GROUP);
    for (uint j = g * GROUP; j < end; j++)
    {
        float acc[H];
        [ForceUnroll] for (uint h = 0; h < H; h++) acc[h] = 0;
        dot_heads(p, qr, key_row(p, slot, j), acc);
        float score[SH];
        head_scores(p, t, acc, score);
        [ForceUnroll] for (uint s = 0; s < SH; s++) best[s] = max(best[s], score[s]);
    }
#endif
#if LOCAL > 0
    uint last = (L - 1) / GROUP;
    if (g + LOCAL > last)
    {
        [ForceUnroll] for (uint s = 0; s < SH; s++) best[s] = asfloat(0x7F800000u);
    }
#endif
    [ForceUnroll] for (uint s = 0; s < SH; s++) p.scores[(uint64_t(t - p.row0) * SH + s) * NG + g] = best[s];
}
#endif

#ifdef KERNEL_INDEX_SELECT
groupshared int sSel[KG];   // the picked groups, in output order
groupshared uint sCount;

[shader("compute")]
[numthreads(WG, 1, 1)]
void index_select(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform IndexerParams p)
{
    uint t = gid.x / SH + p.row0;
    uint head = gid.x % SH;
    uint tid = ltid.x;
    uint slot;
    uint L = visible_keys(p, t, slot);
    uint candidates = candidate_groups(L);
    if (candidates <= KG)
    {
        for (uint i = tid; i < candidates; i += WG)
        {
            sSel[i] = int(i);
        }
        if (tid == 0)
        {
            sCount = candidates;
        }
    }
    else
    {
        float* row = p.scores + (uint64_t(t - p.row0) * SH + head) * NG;
        uint threshold;
        uint needEq;
        radix_threshold(tid, row, 1, candidates, KG, threshold, needEq);
        uint greater = KG - needEq;
        uint chunk = (candidates + WG - 1) / WG;
        uint begin = min(tid * chunk, candidates);
        uint end = min(begin + chunk, candidates);
        uint gtSlot;
        uint eqRank;
        radix_ranks(tid, row, 1, begin, end, threshold, gtSlot, eqRank);
        for (uint g = begin; g < end; g++)
        {
            uint key = radix_key(row[g]);
            if (key > threshold)
            {
                sSel[gtSlot] = int(g);
                gtSlot++;
            }
            else if (key == threshold)
            {
                if (eqRank < needEq)
                {
                    sSel[greater + eqRank] = int(g);
                }
                eqRank++;
            }
        }
        if (tid == 0)
        {
            sCount = KG;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    int* out = p.out + (uint64_t(t) * SH + head) * WIDTH;
    uint count = sCount;
    for (uint i = tid; i < WIDTH; i += WG)
    {
        int position = -1;
        if (i < BUDGET)
        {
            uint entry = i / GROUP;
            if (entry < count)
            {
                uint j = uint(sSel[entry]) * GROUP + i % GROUP;
                position = j < L ? int(j) : -1;
            }
        }
        else
        {
            uint j = candidates * GROUP + (i - BUDGET);   // the tail: the incomplete group's visible positions
            position = j < L ? int(j) : -1;
        }
        out[i] = position;
    }
}
#endif
