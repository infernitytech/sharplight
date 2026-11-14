// Top-k along one axis of a dense f32 tensor viewed as [outer, LEN, INNER]; one row = one (outer, inner) pair.
// Output order (ONNX sorted=1): best value first, ties broken by the lower index; NaN ranks as the largest value,
// -0 equals +0. Values are re-read from x, so they are bit-exact copies of the input.
// Macros: LEN K INNER LARGEST (1 = largest, 0 = smallest).
//   topk_rows:     one workgroup of WG threads per row (radix select of the k-th key, ordered collection, bitonic sort).
//                  Macros: WG (power of two, 256), P2 (power of two >= K). Uses only shared memory and barriers.
//   topk_subgroup: one subgroup (SG lanes) per row for small K (<= 16): per-lane sorted lists merged by subgroup reductions.
// Grid: x * 65535 + y covers the rows; rows beyond p.rows return immediately.
#include "common.slang"
// The radix selection compares topk_key (declared below; Slang resolves global declarations in any order).
#define RADIX_KEY(v) topk_key(v)
#include "radix_select.slang"

struct TopKParams { float* x; float* vals; int* idx; uint rows; };

// Order-preserving key: larger key = better element (for LARGEST = 0 the order is reversed).
uint topk_key(float v)
{
#if LARGEST
    return radix_key(v);
#else
    return ~radix_key(v);
#endif
}

uint topk_row(uint3 gid) { return gid.y * 65535u + gid.x; }
uint64_t topk_in_base(uint row) { return uint64_t(row / INNER) * (uint64_t(LEN) * INNER) + (row % INNER); }
uint64_t topk_out_base(uint row) { return uint64_t(row / INNER) * (uint64_t(K) * INNER) + (row % INNER); }

#ifdef KERNEL_TOPK_ROWS
groupshared uint sKey[P2];
groupshared uint sIdx[P2];

[shader("compute")]
[numthreads(WG, 1, 1)]
void topk_rows(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform TopKParams p)
{
    uint row = topk_row(gid);
    if (row >= p.rows)
    {
        return;
    }

    uint tid = ltid.x;
    float* xr = p.x + topk_in_base(row);

    // 1. Radix select: the key T of the k-th best element and the number of elements equal to T that belong to the top k.
    uint threshold;
    uint needEq;
    radix_threshold(tid, xr, INNER, LEN, K, threshold, needEq);
    uint greater = K - needEq;

    // 2. Ordered collection: thread t owns the contiguous chunk [t*C, (t+1)*C); elements equal to the threshold are
    //    taken in index order (lowest indices win ties), elements above it all go into slots [0, greater).
    const uint C = (LEN + WG - 1) / WG;
    uint begin = min(tid * C, uint(LEN));
    uint end = min(begin + C, uint(LEN));
    uint gtSlot;
    uint eqRank;
    radix_ranks(tid, xr, INNER, begin, end, threshold, gtSlot, eqRank);
    for (uint j = begin; j < end; j++)
    {
        uint k = topk_key(xr[uint64_t(j) * INNER]);
        if (k > threshold)
        {
            sKey[gtSlot] = k;
            sIdx[gtSlot] = j;
            gtSlot++;
        }
        else if (k == threshold)
        {
            if (eqRank < needEq)
            {
                sKey[greater + eqRank] = k;
                sIdx[greater + eqRank] = j;
            }
            eqRank++;
        }
    }
    for (uint s = K + tid; s < P2; s += WG)
    {
        sKey[s] = 0u;
        sIdx[s] = 0xFFFFFFFFu;
    }
    GroupMemoryBarrierWithGroupSync();

    // 3. Bitonic sort of the P2 slots, best first (key descending, index ascending; padding sorts last).
    for (uint size = 2; size <= P2; size <<= 1)
    {
        for (uint stride = size >> 1; stride > 0; stride >>= 1)
        {
            for (uint i = tid; i < P2; i += WG)
            {
                uint partner = i ^ stride;
                if (partner > i)
                {
                    uint ka = sKey[i];
                    uint kb = sKey[partner];
                    uint ia = sIdx[i];
                    uint ib = sIdx[partner];
                    bool partnerBetter = kb > ka || (kb == ka && ib < ia);
                    bool bestFirst = (i & size) == 0;
                    if (partnerBetter == bestFirst)
                    {
                        sKey[i] = kb;
                        sKey[partner] = ka;
                        sIdx[i] = ib;
                        sIdx[partner] = ia;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    uint64_t ob = topk_out_base(row);
    for (uint r = tid; r < K; r += WG)
    {
        uint j = sIdx[r];
        p.vals[ob + uint64_t(r) * INNER] = xr[uint64_t(j) * INNER];
        p.idx[ob + uint64_t(r) * INNER] = int(j);
    }
}
#endif

#ifdef KERNEL_TOPK_SUBGROUP
[shader("compute")]
[numthreads(SG, 1, 1)]
void topk_subgroup(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform TopKParams p)
{
    uint row = topk_row(gid);
    if (row >= p.rows)
    {
        return;
    }

    uint lane = ltid.x;
    float* xr = p.x + topk_in_base(row);

    // Each lane keeps its best K elements (over indices lane, lane + SG, ...) sorted best first; indices arrive in
    // increasing order, so an equal key goes behind the ones already kept.
    uint keys[K];
    uint ids[K];
    for (uint s = 0; s < K; s++)
    {
        keys[s] = 0u;
        ids[s] = 0xFFFFFFFFu;
    }
    uint count = 0u;
    for (uint j = lane; j < LEN; j += SG)
    {
        uint k = topk_key(xr[uint64_t(j) * INNER]);
        if (count == K && k <= keys[K - 1])
        {
            continue;
        }
        uint pos = count < K ? count : K - 1;
        count = min(count + 1u, uint(K));
        while (pos > 0 && keys[pos - 1] < k)
        {
            keys[pos] = keys[pos - 1];
            ids[pos] = ids[pos - 1];
            pos--;
        }
        keys[pos] = k;
        ids[pos] = j;
    }

    // K rounds: the best list head across the subgroup wins and is popped from its lane.
    uint64_t ob = topk_out_base(row);
    for (uint r = 0; r < K; r++)
    {
        uint best = WaveActiveMax(keys[0]);
        uint winner = WaveActiveMin(keys[0] == best ? ids[0] : 0xFFFFFFFFu);
        if (ids[0] == winner)
        {
            p.vals[ob + uint64_t(r) * INNER] = xr[uint64_t(winner) * INNER];
            p.idx[ob + uint64_t(r) * INNER] = int(winner);
            for (uint s = 0; s + 1 < K; s++)
            {
                keys[s] = keys[s + 1];
                ids[s] = ids[s + 1];
            }
            keys[K - 1] = 0u;
            ids[K - 1] = 0xFFFFFFFFu;
        }
    }
}
#endif
