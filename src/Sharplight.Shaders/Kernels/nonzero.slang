// NonZero: coordinates of the non-zero elements of a dense tensor, in row-major order, as i32 rows [count, RANK], and their
// number (the data-dependent count of the output). One workgroup of WG threads: each thread counts the non-zero elements of a
// contiguous chunk, an exclusive scan over the threads gives the chunk's first row, then each thread writes its rows.
// Macros: RAW (element bits type: uint8_t, uint16_t, uint or uint64_t), MASK (bits that make an element non-zero: all but the
// sign of a float, so -0 is zero and NaN is not), RANK, DIMS (the RANK - 1 inner extents, comma separated; "1" for rank 1), WG.
#include "common.slang"

struct NonZeroParams { RAW* x; int* y; int* count; uint total; };

static const uint kInner[] = { DIMS };

groupshared uint sScan[WG];

bool nz_test(RAW v) { return (v & RAW(MASK)) != RAW(0); }

[shader("compute")]
[numthreads(WG, 1, 1)]
void nonzero(uint3 ltid : SV_GroupThreadID, uniform NonZeroParams p)
{
    uint tid = ltid.x;
    uint chunk = (p.total + WG - 1u) / WG;
    uint begin = min(tid * chunk, p.total);
    uint end = min(begin + chunk, p.total);
    uint found = 0u;
    for (uint i = begin; i < end; i++)
    {
        found += nz_test(p.x[i]) ? 1u : 0u;
    }

    sScan[tid] = found;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1u; offset < WG; offset <<= 1)
    {
        uint add = tid >= offset ? sScan[tid - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        sScan[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }

    uint row = sScan[tid] - found;
    if (tid == WG - 1u)
    {
        p.count[0] = int(sScan[tid]);
    }

    for (uint i = begin; i < end; i++)
    {
        if (nz_test(p.x[i]))
        {
            int* y = p.y + uint64_t(row) * RANK;
            uint rest = i;
            for (int k = RANK - 1; k > 0; k--)
            {
                y[k] = int(rest % kInner[k - 1]);
                rest /= kInner[k - 1];
            }
            y[0] = int(rest);
            row++;
        }
    }
}
