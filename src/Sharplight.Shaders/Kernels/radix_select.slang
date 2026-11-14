#pragma once
// Workgroup radix selection of the k best elements of one row (shared by topk.slang and indexer.slang).
// Elements are compared through RADIX_KEY(float) (a uint key, larger = better; defaults to radix_key, the largest-first order) and read as
// row[j * stride] for j < len. One workgroup of WG threads per row (a power of two of at least 256: the bin scans run one thread per bin).
//   radix_threshold: the key of the k-th best element and how many elements equal to it belong to the top k (8 bits per pass).
//   radix_ranks:     thread tid owns the contiguous chunk [begin, end) of the row; returns its first output slot among the elements above
//                    the threshold and its rank among the elements equal to it, so a collection pass keeps the index order (lowest wins ties).
#include "common.slang"

// Order-preserving key of the largest-first order: NaN ranks as the largest value, -0 equals +0.
uint radix_key(float v)
{
    if (isnan(v))
    {
        return 0xFFFFFFFFu;
    }

    uint u = asuint(v == 0.0 ? 0.0 : v);
    return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}

#ifndef RADIX_KEY
#define RADIX_KEY(v) radix_key(v)
#endif

groupshared uint sHist[256];
groupshared uint sGt[WG];
groupshared uint sEq[WG];
groupshared uint sPrefix;
groupshared uint sRemaining;

// Inclusive Hillis-Steele scans of sGt and sEq over the WG threads.
void radix_scan_pair(uint tid)
{
    for (uint off = 1; off < WG; off <<= 1)
    {
        uint g = tid >= off ? sGt[tid - off] : 0u;
        uint e = tid >= off ? sEq[tid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        sGt[tid] += g;
        sEq[tid] += e;
        GroupMemoryBarrierWithGroupSync();
    }
}

void radix_threshold(uint tid, float* row, uint64_t stride, uint len, uint k, out uint threshold, out uint needEq)
{
    if (tid == 0)
    {
        sPrefix = 0u;
        sRemaining = k;
    }
    uint mask = 0u;
    for (int shift = 24; shift >= 0; shift -= 8)
    {
        for (uint b = tid; b < 256u; b += WG)
        {
            sHist[b] = 0u;
        }
        GroupMemoryBarrierWithGroupSync();
        uint prefix = sPrefix;
        for (uint j = tid; j < len; j += WG)
        {
            uint key = RADIX_KEY(row[uint64_t(j) * stride]);
            if ((key & mask) == prefix)
            {
                InterlockedAdd(sHist[(key >> uint(shift)) & 255u], 1u);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        // Suffix sums: sGt[b] = number of candidates in bins > b (reusing the scan buffer, bins mapped high to low).
        for (uint b = tid; b < 256u; b += WG)
        {
            sGt[b] = sHist[255u - b];
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint off = 1; off < 256u; off <<= 1)
        {
            uint add = 0u;
            if (tid < 256u && tid >= off)
            {
                add = sGt[tid - off];
            }
            GroupMemoryBarrierWithGroupSync();
            if (tid < 256u)
            {
                sGt[tid] += add;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        uint remaining = sRemaining;
        GroupMemoryBarrierWithGroupSync();
        if (tid < 256u)
        {
            uint bin = 255u - tid;
            uint inclusive = sGt[tid];
            uint above = inclusive - sHist[bin];
            if (above < remaining && inclusive >= remaining)
            {
                sPrefix = prefix | (bin << uint(shift));
                sRemaining = remaining - above;
            }
        }
        mask |= 255u << uint(shift);
        GroupMemoryBarrierWithGroupSync();
    }
    threshold = sPrefix;
    needEq = sRemaining;
}

void radix_ranks(uint tid, float* row, uint64_t stride, uint begin, uint end, uint threshold, out uint gtSlot, out uint eqRank)
{
    uint gt = 0u;
    uint eq = 0u;
    for (uint j = begin; j < end; j++)
    {
        uint key = RADIX_KEY(row[uint64_t(j) * stride]);
        gt += key > threshold ? 1u : 0u;
        eq += key == threshold ? 1u : 0u;
    }
    sGt[tid] = gt;
    sEq[tid] = eq;
    GroupMemoryBarrierWithGroupSync();
    radix_scan_pair(tid);
    gtSlot = sGt[tid] - gt;
    eqRank = sEq[tid] - eq;
}
