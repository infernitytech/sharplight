// Softmax-gated pooling of windows of RATIO positions (WindowPool, see StandardOps.WindowPool) in two launches:
//   window_pool:      one thread per (row t, output channel d): a row that closes a window (position p with (p + 1) % RATIO == 0) writes
//                     sum_q softmax_q(score[q][c] + bias[q % RATIO][c]) * value[q][c] over the window's positions q (OVERLAP: the window
//                     before it too, whose rows give channel d, the window's own rows channel OC + d), every other row zeros. Positions of an
//                     earlier pass come from the ring state. Grid (ceil(T * OC / 128)).
//   window_pool_ring: afterwards, one thread per (ring row, channel) and run: the last P rows of every run go to the ring (value | biased
//                     score at row position % P). Grid (ceil(P * C / 128), runs).
// values, scores: [T, C] dense f32; bias: [RATIO, C] f32; positions: i32, element t * PS is row t's position (single sequence); ring: [P, 2C]
// f32 (multi-sequence: [slots, P, 2C]); y: [T, OC] f32. Multi-sequence: seq [T, 4] (slot, position, run rows, run index), runs [U, 4] (slot,
// first row, row count, first position).
// Macros: C, OC (C / 2 with OVERLAP, else C), RATIO, OVERLAP, P (2 * RATIO with OVERLAP, else RATIO), PS, MULTI_SEQ.
#include "common.slang"

struct WindowPoolParams
{
    float* values;
    float* scores;
    float* bias;
    int* positions;
    float* ring;
    float* y;
#if MULTI_SEQ
    int* seq;
    int* runs;
#endif
    uint T;
    uint U;
};

// Run u: slot, first row, row count, first position.
void run_of(WindowPoolParams p, uint u, out uint slot, out uint first, out uint count, out uint firstPos)
{
#if MULTI_SEQ
    slot = uint(p.runs[u * 4]);
    first = uint(p.runs[u * 4 + 1]);
    count = uint(p.runs[u * 4 + 2]);
    firstPos = uint(p.runs[u * 4 + 3]);
#else
    slot = 0;
    first = 0;
    count = p.T;
    firstPos = uint(p.positions[0]);
#endif
}

uint position_of(WindowPoolParams p, uint t)
{
#if MULTI_SEQ
    return uint(p.seq[t * 4 + 1]);
#else
    return uint(p.positions[uint64_t(t) * PS]);
#endif
}

#ifdef KERNEL_WINDOW_POOL
// Value and biased score of channel c at position q: from this pass's rows, or from the ring for positions of an earlier pass.
void entry(WindowPoolParams p, uint q, uint c, uint slot, uint first, uint firstPos, out float value, out float score)
{
    if (q >= firstPos)
    {
        uint64_t row = uint64_t(first + (q - firstPos)) * C + c;
        value = p.values[row];
        score = p.scores[row] + p.bias[(q % RATIO) * C + c];
    }
    else
    {
        float* r = p.ring + (uint64_t(slot) * P + q % P) * (2 * C);
        value = r[c];
        score = r[C + c];
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void window_pool(uint3 dtid : SV_DispatchThreadID, uniform WindowPoolParams p)
{
    uint id = dtid.x;
    if (id >= p.T * OC) return;
    uint t = id / OC, d = id % OC;
    uint pos = position_of(p, t);
    float* y = p.y + uint64_t(t) * OC + d;
    if ((pos + 1) % RATIO != 0)
    {
        *y = 0;
        return;
    }

    uint slot, first, count, firstPos;
#if MULTI_SEQ
    run_of(p, uint(p.seq[t * 4 + 3]), slot, first, count, firstPos);
#else
    run_of(p, 0, slot, first, count, firstPos);
#endif
    uint windowStart = pos + 1 - RATIO;
#if OVERLAP
    uint from = windowStart >= RATIO ? windowStart - RATIO : 0;
#else
    uint from = windowStart;
#endif
    float m = -1e30, l = 0, acc = 0;
    for (uint q = from; q <= pos; q++)
    {
#if OVERLAP
        uint c = q >= windowStart ? OC + d : d;
#else
        uint c = d;
#endif
        float v, s;
        entry(p, q, c, slot, first, firstPos, v, s);
        float mNew = max(m, s);
        float corr = exp(m - mNew);
        float e = exp(s - mNew);
        l = l * corr + e;
        acc = acc * corr + e * v;
        m = mNew;
    }
    *y = acc / l;
}
#endif

#ifdef KERNEL_WINDOW_POOL_RING
[shader("compute")]
[numthreads(128, 1, 1)]
void window_pool_ring(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uniform WindowPoolParams p)
{
    uint id = dtid.x;
    uint u = gid.y;
    if (id >= P * C || u >= p.U) return;
    uint i = id / C, c = id % C;
    uint slot, first, count, firstPos;
    run_of(p, u, slot, first, count, firstPos);
    if (i >= count) return;
    uint row = first + count - min(count, uint(P)) + i;   // the run's last min(count, P) rows
    if (row >= first + count) return;
    uint q = firstPos + (row - first);
    float* r = p.ring + (uint64_t(slot) * P + q % P) * (2 * C);
    r[c] = p.values[uint64_t(row) * C + c];
    r[C + c] = p.scores[uint64_t(row) * C + c] + p.bias[(q % RATIO) * C + c];
}
#endif
