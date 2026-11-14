// Lightning-indexer key selection (IndexerTopK, DeepSeek sparse attention) in two launches over the query rows [row0, row0 + rows) of a launch:
//   index_scores: score[t][j] = sum_h w[t][h] * relu(q[t][h] . key[j]) for every key position j the row sees, into scores[(t - row0) * CTX + j].
//                 Grid (ceil(CTX / WG), rows); one thread per key position (the row's queries and head weights are uniform loads).
//   index_select: one workgroup per row; the positions of the K best scores (radix select, then an ordered collection: positions above the
//                 K-th score in ascending order, then the ties at it in ascending order). A row that sees at most K positions lists all of them,
//                 followed by -1.
// q: [T, H, D] dense f32; keys: the cache [ctx, 1, D] in KEY_F16 ? f16 : f32; w: [T, H] dense f32; out: [T, K] i32.
// Macros: H, D (multiple of 8), K, CTX (positions per score row: the per-sequence context), KEY_F16, WG, RT_N_PAST; MULTI_SEQ / SLOT_CTX and
// PAGED / PAGE_BS / PAGE_MB for the multi-sequence and paged forms (row t sees positions [0, seq[t].position] of its slot).
#include "common.slang"
#include "radix_select.slang"

struct IndexerParams
{
    float* q;
    uint8_t* keys;
    float* w;
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

// Visible positions of query row t and the cache row of its slot's position 0 (contiguous caches; paged caches map every position).
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

#ifdef KERNEL_INDEX_SCORES
[shader("compute")]
[numthreads(WG, 1, 1)]
void index_scores(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform IndexerParams p)
{
    uint t = gid.y + p.row0;
    uint tid = ltid.x;
    uint slot;
    uint L = visible_keys(p, t, slot);
    uint j = gid.x * WG + tid;
    if (j >= L)
    {
        return;
    }

    float* qr = p.q + uint64_t(t) * (H * D);
    float acc[H];
    [ForceUnroll] for (uint h = 0; h < H; h++) acc[h] = 0;
    uint64_t row = key_row(p, slot, j);
    for (uint g = 0; g < D / 8; g++)
    {
        float k[8];
#if KEY_F16
        load8_f16(p.keys, D * 2, uint(row), g, k);
#else
        load8_f32(p.keys, D * 4, uint(row), g, k);
#endif
        [ForceUnroll]
        for (uint h = 0; h < H; h++)
        {
            float s = 0;
            [ForceUnroll] for (uint e = 0; e < 8; e++) s += qr[h * D + g * 8 + e] * k[e];
            acc[h] += s;
        }
    }

    float score = 0;
    [ForceUnroll] for (uint h = 0; h < H; h++) score += p.w[uint64_t(t) * H + h] * max(acc[h], 0.0);
    p.scores[uint64_t(t - p.row0) * CTX + j] = score;
}
#endif

#ifdef KERNEL_INDEX_SELECT
[shader("compute")]
[numthreads(WG, 1, 1)]
void index_select(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform IndexerParams p)
{
    uint t = gid.x + p.row0;
    uint tid = ltid.x;
    uint slot;
    uint L = visible_keys(p, t, slot);
    int* out = p.out + uint64_t(t) * K;
    if (L <= K)
    {
        for (uint i = tid; i < K; i += WG)
        {
            out[i] = i < L ? int(i) : -1;
        }
        return;
    }

    float* row = p.scores + uint64_t(t - p.row0) * CTX;
    uint threshold;
    uint needEq;
    radix_threshold(tid, row, 1, L, K, threshold, needEq);
    uint greater = K - needEq;
    uint chunk = (L + WG - 1) / WG;
    uint begin = min(tid * chunk, L);
    uint end = min(begin + chunk, L);
    uint gtSlot;
    uint eqRank;
    radix_ranks(tid, row, 1, begin, end, threshold, gtSlot, eqRank);
    for (uint j = begin; j < end; j++)
    {
        uint key = radix_key(row[j]);
        if (key > threshold)
        {
            out[gtSlot] = int(j);
            gtSlot++;
        }
        else if (key == threshold)
        {
            if (eqRank < needEq)
            {
                out[greater + eqRank] = int(j);
            }
            eqRank++;
        }
    }
}
#endif
