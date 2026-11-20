// In-GEMM split-K reduction and multi-output stores, shared by gemm_cm2 (matmul.slang) and gemm_fp8 (gemm_fp8.slang).
//
// SPLITK_FUSED (with SPLITK > 1): the workgroups of one output tile each write their K range's partial sums to slice z of p.y
// ([SPLITK][T][N], as for splitk_reduce), publish them (device-scope release) and count their arrival in p.arrivals[tile]. The
// last one to arrive acquires the other slices, sums them in slice order (z = 0 .. SPLITK - 1 from zero, the order of
// splitk_reduce, so the results are bit-identical to the separate reduction launch) and stores the result; it also resets the
// counter, so every launch leaves the counters zero. The workgroups never wait for each other, so no residency is assumed.
//
// MULTI (2 or 3, with SPLITK_FUSED): the weight rows are the concatenation of MULTI matrices of the same format and K that share
// x (e.g. the q / k / v projections), each its own buffer: rows [0, MULTI_O1) of p.w, [MULTI_O1, MULTI_O2) of p.mw1 and (MULTI
// 3) [MULTI_O2, N) of p.mw2. MULTI_O1 and MULTI_O2 are multiples of the column tile, so every tile reads one matrix. The
// reduction stores each output column to its matrix's own dense output (p.yout, p.yout1, p.yout2; rows of that matrix's width),
// with MULTI_BIAS the matrix's bias row (p.mb0 .. p.mb2) added, as the separate projections' reductions do.
//
// Expected of the including kernel: Params with y (the partial slices), yout, arrivals and T (and the MULTI fields), and
// SPLITK_STORE(p, s, t, n) defined below or by the kernel (the store of a reduced value s of output (t, n)).
#ifndef SPLITK_FUSED
#define SPLITK_FUSED 0
#endif
#ifndef MULTI
#define MULTI 1
#endif
#ifndef MULTI_BIAS
#define MULTI_BIAS 0
#endif
// Without SPLITK_FUSED, a MULTI GEMM writes the partial slices of the concatenation (its y) and splitk_reduce with MULTI stores them.

#if SPLITK_FUSED
groupshared uint gSplitLast;

// Publishes this workgroup's partial sums and counts its arrival at the tile's counter; true in the last workgroup of the tile
// (after acquiring the others' partials), which also resets the counter. Called by every thread of the workgroup.
bool splitk_arrive(uint* arrivals, uint tile, uint tid)
{
    DeviceMemoryBarrier();
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0)
    {
        uint arrived;
        // Indexed, not through a dereferenced pointer: Slang's SPIR-V of an atomic on `*ptr` does not reach the buffer.
        InterlockedAdd(arrivals[tile], 1u, arrived);
        bool last = arrived + 1 == SPLITK;
        if (last)
        {
            arrivals[tile] = 0;
        }
        gSplitLast = last ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    bool last = gSplitLast != 0;
    if (last)
    {
        DeviceMemoryBarrier();
    }
    return last;
}
#endif

#if MULTI > 1
#define MULTI_N0 MULTI_O1
#if MULTI > 2
#define MULTI_N1 (MULTI_O2 - MULTI_O1)
#define MULTI_N2 (N - MULTI_O2)
#else
#define MULTI_N1 (N - MULTI_O1)
#endif
#if MULTI_BIAS
#define MULTI_BIAS_OF(b, c) (b)[c]
#else
#define MULTI_BIAS_OF(b, c) 0.0
#endif
// Base pointer of the matrix holding weight row n0 (a tile's first row), offset so that rowStart(base, ROW_BYTES, n) addresses
// global row n of the concatenation.
#if MULTI > 2
#define MULTI_WBASE(p, n0) ((n0) < MULTI_O1 ? p.w : (n0) < MULTI_O2 ? (uint8_t*)(uint64_t(p.mw1) - uint64_t(MULTI_O1) * (ROW_BYTES)) \
                                                                  : (uint8_t*)(uint64_t(p.mw2) - uint64_t(MULTI_O2) * (ROW_BYTES)))
#else
#define MULTI_WBASE(p, n0) ((n0) < MULTI_O1 ? p.w : (uint8_t*)(uint64_t(p.mw1) - uint64_t(MULTI_O1) * (ROW_BYTES)))
#endif
#if MULTI > 2
#define MULTI_STORE(p, s, t, n)                                                                                         \
    { if ((n) < MULTI_O1) p.yout[uint64_t(t) * MULTI_N0 + (n)] = MULTI_BIAS_OF(p.mb0, n) + (s);                         \
      else if ((n) < MULTI_O2) p.yout1[uint64_t(t) * MULTI_N1 + (n) - MULTI_O1] = MULTI_BIAS_OF(p.mb1, (n) - MULTI_O1) + (s); \
      else p.yout2[uint64_t(t) * MULTI_N2 + (n) - MULTI_O2] = MULTI_BIAS_OF(p.mb2, (n) - MULTI_O2) + (s); }
#else
#define MULTI_STORE(p, s, t, n)                                                                                         \
    { if ((n) < MULTI_O1) p.yout[uint64_t(t) * MULTI_N0 + (n)] = MULTI_BIAS_OF(p.mb0, n) + (s);                         \
      else p.yout1[uint64_t(t) * MULTI_N1 + (n) - MULTI_O1] = MULTI_BIAS_OF(p.mb1, (n) - MULTI_O1) + (s); }
#endif
#endif

// The reduction of one BM x BN output tile by the last of its workgroups: THREADS threads, thread tid; p, the kernel's Params.
// Reads the slices four columns at a time when N allows (each column's sum in the same order). ROW_SCALE(t) multiplies each
// row's sum before the store (1 or the inverse of a row-ranged copy's scale).
#if SPLITK_FUSED
#define SPLITK_FINISH(p, tid, THREADS, BM, BN, tile, t0, n0)                                                            \
    if (splitk_arrive(p.arrivals, tile, tid))                                                                        \
    {                                                                                                                   \
        uint64_t sliceElems = uint64_t(p.T) * N;                                                                       \
        if (N % 4 == 0)                                                                                                 \
        {                                                                                                               \
            for (uint i = tid; i < (BM) * (BN) / 4; i += (THREADS))                                                     \
            {                                                                                                           \
                uint t = (t0) + i / ((BN) / 4), n = (n0) + (i % ((BN) / 4)) * 4;                                       \
                if (t < p.T && n < N)                                                                                   \
                {                                                                                                       \
                    uint64_t o = uint64_t(t) * N + n;                                                                   \
                    float4 s = 0;                                                                                       \
                    [ForceUnroll] for (uint z = 0; z < SPLITK; z++) s += *(float4*)(p.y + z * sliceElems + o);          \
                    float rs = SPLITK_ROW_SCALE(p, t);                                                                  \
                    SPLITK_STORE(p, s.x * rs, t, n) SPLITK_STORE(p, s.y * rs, t, n + 1)                                 \
                    SPLITK_STORE(p, s.z * rs, t, n + 2) SPLITK_STORE(p, s.w * rs, t, n + 3)                             \
                }                                                                                                       \
            }                                                                                                           \
        }                                                                                                               \
        else                                                                                                            \
        {                                                                                                               \
            for (uint i = tid; i < (BM) * (BN); i += (THREADS))                                                         \
            {                                                                                                           \
                uint t = (t0) + i / (BN), n = (n0) + i % (BN);                                                          \
                if (t < p.T && n < N)                                                                                   \
                {                                                                                                       \
                    uint64_t o = uint64_t(t) * N + n;                                                                   \
                    float s = 0;                                                                                        \
                    [ForceUnroll] for (uint z = 0; z < SPLITK; z++) s += p.y[z * sliceElems + o];                       \
                    SPLITK_STORE(p, s * SPLITK_ROW_SCALE(p, t), t, n)                                                   \
                }                                                                                                       \
            }                                                                                                           \
        }                                                                                                               \
    }

// The dual form: slices [SPLITK][T][2N] holding both products of output (t, n) at columns n and N + n; the last workgroup sums each
// in slice order and stores SPLITK_STORE2(p, g, u, t, n) (the kernel's pair epilogue).
#define SPLITK_FINISH_DUAL(p, tid, THREADS, BM, BN, tile, t0, n0)                                                       \
    if (splitk_arrive(p.arrivals, tile, tid))                                                                        \
    {                                                                                                                   \
        uint64_t sliceElems = uint64_t(p.T) * 2 * N;                                                                   \
        for (uint i = tid; i < (BM) * (BN); i += (THREADS))                                                             \
        {                                                                                                               \
            uint t = (t0) + i / (BN), n = (n0) + i % (BN);                                                              \
            if (t < p.T && n < N)                                                                                       \
            {                                                                                                           \
                uint64_t o = uint64_t(t) * 2 * N + n;                                                                   \
                float g = 0, u = 0;                                                                                     \
                [ForceUnroll] for (uint z = 0; z < SPLITK; z++) { g += p.y[z * sliceElems + o]; u += p.y[z * sliceElems + o + N]; } \
                SPLITK_STORE2(p, g, u, t, n)                                                                            \
            }                                                                                                           \
        }                                                                                                               \
    }
#endif
