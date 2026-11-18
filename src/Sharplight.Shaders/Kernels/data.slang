// Data movement kernels.
#include "common.slang"

#ifdef KERNEL_GATHER
// ---------------------------------------------------------------------------------------------
// Gather rows: table [V, D] of format WFMT (ROW_BYTES per row) or f32 with row stride TS
// (when WFMT == FMT_F32 and TS is defined the table may be an intermediate); idx i32 [N];
// out dense [N, D] f32. One thread per 8 values.
// Macros: D, WFMT, ROW_BYTES
struct GatherParams
{
#if HAS_GRID
    uint* grid;
#endif
    uint8_t* table;
    int* idx;
    float* y;
    uint N;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void gather(uint3 dtid : SV_DispatchThreadID, uniform GatherParams p)
{
    GRID_BIND(p);
    const uint G = D / 8;
    uint id = dtid.x;
    if (id >= p.N * G) return;
    uint n = id / G, g = id % G;
    uint row = uint(p.idx[n]);
    float v[8];
    load8(p.table, ROW_BYTES, row, g, v);
    float* yr = p.y + uint64_t(n) * D + g * 8;
    [ForceUnroll] for (int i = 0; i < 8; i++) yr[i] = v[i];
}

#endif

#if defined(KERNEL_CACHE_WRITE) || defined(KERNEL_CACHE_WRITE_Q)
// ---------------------------------------------------------------------------------------------
// Cache write: x [T, R1, INNER] with strides (S0, S1) → state rows [n_past + t] (dense rows of
// R1*INNER elements) converting to f16 when DST_F16; `cache_write_q` quantizes to the GGUF block format DST_FMT
// (FMT_Q8_0 / FMT_Q4_0, INNER a multiple of 32). KV_COMP > 1 (`cache_write` only): a compressed cache, see seq_row_c.
// Macros: R1, INNER, S0, S1, DST_F16 / DST_FMT, KV_COMP / COMP_SLOT_CTX
struct CacheParams
{
    float* x;
    uint8_t* dst;
    uint* rt;
#if MULTI_SEQ
    int* seq;   // [T, 4] (slot, position, ...): row t goes to cache row seq_row(slot, position)
#endif
#if PAGED
    int* pages; // [slots, PAGE_MB] block table
#endif
    uint T;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void cache_write(uint3 dtid : SV_DispatchThreadID, uniform CacheParams p)
{
    const uint ROW = R1 * INNER;
    uint id = dtid.x;
    if (id >= p.T * ROW) return;
    uint t = id / ROW, r = id % ROW;
    uint r1 = r / INNER, i = r % INNER;
    float v = p.x[uint64_t(t) * S0 + uint64_t(r1) * S1 + i];
#if KV_COMP > 1
    // Compressed cache: only the rows that close a window are written, to their window's entry.
#if MULTI_SEQ
    uint pos = uint(p.seq[t * 4 + 1]);
#else
    uint pos = p.rt[RT_N_PAST] + t;
#endif
    if ((pos + 1) % KV_COMP != 0) return;
#if MULTI_SEQ
    uint64_t di = seq_row_c(SEQ_PAGES(p), uint(p.seq[t * 4]), (pos + 1) / KV_COMP - 1) * ROW + r;
#else
    uint64_t di = uint64_t((pos + 1) / KV_COMP - 1) * ROW + r;
#endif
#elif MULTI_SEQ
    uint64_t di = seq_row(SEQ_PAGES(p), uint(p.seq[t * 4]), uint(p.seq[t * 4 + 1])) * ROW + r;
#else
    uint64_t di = uint64_t(p.rt[RT_N_PAST] + t) * ROW + r;
#endif
#if DST_F16
    *(uint16_t*)byteAt((uint8_t*)p.dst, di * 2) = f2h(v);
#else
    *(float*)byteAt((uint8_t*)p.dst, di * 4) = v;
#endif
}
#endif

#ifdef KERNEL_CACHE_WRITE_Q
// One thread per 32-value block: scale from the block's absolute maximum (q8_0: amax / 127; q4_0: the value of largest
// magnitude / -8, as llama.cpp's reference quantizers), then the f16 scale and the quants as 16-bit words (blocks of 34 / 18
// bytes are 2-byte aligned).
#define CQ_BPR (R1 * INNER / 32)                         // blocks per token row
#define CQ_BYTES (DST_FMT == FMT_Q8_0 ? 34 : 18)
[shader("compute")]
[numthreads(128, 1, 1)]
void cache_write_q(uint3 dtid : SV_DispatchThreadID, uniform CacheParams p)
{
    uint id = dtid.x;
    if (id >= p.T * CQ_BPR) return;
    uint t = id / CQ_BPR, blk = id % CQ_BPR;
    uint r1 = blk * 32 / INNER, i0 = blk * 32 % INNER;
    float* src = p.x + uint64_t(t) * S0 + uint64_t(r1) * S1 + i0;
    float v[32];
    float amax = 0, vmax = 0;
    [ForceUnroll]
    for (uint i = 0; i < 32; i++)
    {
        v[i] = src[i];
        if (abs(v[i]) > amax)
        {
            amax = abs(v[i]);
            vmax = v[i];
        }
    }
#if MULTI_SEQ
    uint64_t row = seq_row(SEQ_PAGES(p), uint(p.seq[t * 4]), uint(p.seq[t * 4 + 1]));
#else
    uint64_t row = p.rt[RT_N_PAST] + t;
#endif
    uint8_t* dst = byteAt(p.dst, (row * CQ_BPR + blk) * CQ_BYTES);
#if DST_FMT == FMT_Q8_0
    float d = amax / 127.0;
    float inv = d != 0 ? 1.0 / d : 0.0;
    *(uint16_t*)dst = f2h(d);
    [ForceUnroll]
    for (uint i = 0; i < 16; i++)
    {
        uint lo = uint(int(round(v[2 * i] * inv))) & 0xFF;
        uint hi = uint(int(round(v[2 * i + 1] * inv))) & 0xFF;
        *(uint16_t*)(dst + 2 + 2 * i) = uint16_t(lo | (hi << 8));
    }
#else
    float d = vmax / -8.0;
    float inv = d != 0 ? 1.0 / d : 0.0;
    *(uint16_t*)dst = f2h(d);
    [ForceUnroll]
    for (uint i = 0; i < 8; i++)
    {
        uint b0 = min(15u, uint(v[2 * i] * inv + 8.5)) | (min(15u, uint(v[2 * i + 16] * inv + 8.5)) << 4);
        uint b1 = min(15u, uint(v[2 * i + 1] * inv + 8.5)) | (min(15u, uint(v[2 * i + 17] * inv + 8.5)) << 4);
        *(uint16_t*)(dst + 2 + 2 * i) = uint16_t(b0 | (b1 << 8));
    }
#endif
}
#endif

#ifdef KERNEL_FILL
// ---------------------------------------------------------------------------------------------
// Zero-fill (state reset): one thread per uint.
struct FillParams
{
    uint* dst;
    uint count;
    uint value;
    uint stride;   // total threads in the dispatch (grid-stride loop)
};

[shader("compute")]
[numthreads(256, 1, 1)]
void fill(uint3 dtid : SV_DispatchThreadID, uniform FillParams p)
{
    for (uint i = dtid.x; i < p.count; i += p.stride) p.dst[i] = p.value;
}
#endif

#ifdef KERNEL_TO_F16
// Rows of f32 (row stride X_STRIDE elements) -> dense f16 rows of K elements, 8 per thread.
// Tensor-core GEMMs read their activations this way: half the bytes per tile load.
struct ToF16Params
{
    float* x;
    uint16_t* y;
    uint T;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void to_f16(uint3 dtid : SV_DispatchThreadID, uniform ToF16Params p)
{
    uint64_t g = dtid.x;
    uint64_t row = g / (K / 8), c = (g % (K / 8)) * 8;
    if (row >= p.T) return;
    float4* xp = (float4*)(p.x + row * X_STRIDE + c);
    float4 a = xp[0], b = xp[1];
    uint4 u;
    u.x = f32tof16(a.x) | (f32tof16(a.y) << 16);
    u.y = f32tof16(a.z) | (f32tof16(a.w) << 16);
    u.z = f32tof16(b.x) | (f32tof16(b.y) << 16);
    u.w = f32tof16(b.z) | (f32tof16(b.w) << 16);
    ((uint4*)(p.y + row * K + c))[0] = u;
}
#endif

#ifdef KERNEL_TO_F16_RANGED
// Rows of f32 (row stride X_STRIDE elements) -> a row-ranged f16 copy (common.slang, f16_range_scale): dense f16 rows of K
// elements, then one scale word per row at byte offset Y_RS. One workgroup per row (f16_range_row).
#include "f16_range_row.slang"

struct ToF16RangedParams
{
    float* x;
    uint8_t* y;
    uint T;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void to_f16_ranged(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform ToF16RangedParams p)
{
    uint row = gid.x, tid = ltid.x;
    if (row >= p.T) return;
    uint word = f16_range_row(p.x + uint64_t(row) * X_STRIDE, uint64_t(p.y) + uint64_t(row) * K * 2, tid);
    if (tid == 0) *(uint*)(uint64_t(p.y) + Y_RS + uint64_t(row) * 4) = word;
}
#endif

#ifdef KERNEL_F16_RANGE_FIXUP
// Completes a row-ranged f16 copy that its producer wrote unscaled (the dual gemm_cm2's store, DUAL_Y16): the producer raised
// flags[t] to the bits of the largest magnitude it saw from 2^15 on (atomic max; 0 otherwise). Rows with a flag are converted
// again from the f32 rows x (row stride K) with f16_range_scale, and every row gets its scale word (at byte offset Y_RS of y).
// The flags are left zero for the next launch. One workgroup per row.
struct F16RangeFixupParams
{
    float* x;
    uint8_t* y;
    uint* flags;
    uint T;
};

groupshared uint gRangeFlag;

[shader("compute")]
[numthreads(256, 1, 1)]
void f16_range_fixup(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform F16RangeFixupParams p)
{
    uint row = gid.x, tid = ltid.x;
    if (row >= p.T) return;
    if (tid == 0) gRangeFlag = p.flags[row];
    GroupMemoryBarrierWithGroupSync();
    uint flag = gRangeFlag;
    uint* word = (uint*)(uint64_t(p.y) + Y_RS + uint64_t(row) * 4);
    if (flag == 0)
    {
        if (tid == 0) *word = 0u;
        return;
    }
    float s = f16_range_scale(asfloat(flag));
    float* xr = p.x + uint64_t(row) * K;
    uint64_t yr = uint64_t(p.y) + uint64_t(row) * K * 2;
    for (uint c = tid * 8; c < K; c += 2048)
    {
        float4 a = ((float4*)(xr + c))[0] * s, b = ((float4*)(xr + c))[1] * s;
        *(uint4*)(yr + uint64_t(c) * 2) = uint4(f32tof16(a.x) | (f32tof16(a.y) << 16), f32tof16(a.z) | (f32tof16(a.w) << 16),
                                                f32tof16(b.x) | (f32tof16(b.y) << 16), f32tof16(b.z) | (f32tof16(b.w) << 16));
    }
    if (tid == 0)
    {
        *word = s == 1.0 ? 0u : asuint(1.0 / s);
        p.flags[row] = 0u;
    }
}
#endif

#ifdef KERNEL_F16_RANGE_UNSCALE
// Undoes the row scales of a row-ranged f16 copy (common.slang, f16_range_scale) in the output of a GEMM that read it: rows of
// y (N values, row stride Y_STRIDE) whose scale word (byte offset X_RS of x) is set are multiplied by its inverse. 64 rows per
// workgroup; the rows without a scale (nearly all) cost one word read.
struct F16RangeUnscaleParams
{
    uint8_t* x;
    float* y;
    uint T;
};

groupshared uint gUnscale[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void f16_range_unscale(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform F16RangeUnscaleParams p)
{
    uint t0 = gid.x * 64, tid = ltid.x;
    uint word = t0 + tid < p.T ? *(uint*)(uint64_t(p.x) + X_RS + uint64_t(t0 + tid) * 4) : 0u;
    gUnscale[tid] = word;
    GroupMemoryBarrierWithGroupSync();
    for (uint r = 0; r < 64; r++)
    {
        uint w = gUnscale[r];
        if (w == 0) continue;
        float inv = f16_range_inverse(w);
        float* yr = p.y + uint64_t(t0 + r) * Y_STRIDE;
        for (uint n = tid; n < N; n += 64) yr[n] *= inv;
    }
}
#endif

#ifdef KERNEL_SPLIT_ROWS
// Rows of f32 (row stride X_STRIDE elements) -> the split f16 form of the split-f16 GEMM (common.slang): per row
// 4K + 16 bytes holding K hi halves, K lo halves and 1 / split_scale of the row's largest magnitude (the row was multiplied
// by split_scale before splitting). One workgroup per row, 8 values per thread and step.
// SPLIT_PARTS = 1 (fast matrix precision): the hi halves only, 2K + 16 bytes per row (K hi halves, then 1 / split_scale).
#ifndef SPLIT_PARTS
#define SPLIT_PARTS 3
#endif
#if SPLIT_PARTS == 1
#define SPLIT_HALVES 1
#else
#define SPLIT_HALVES 2
#endif
struct SplitRowsParams
{
    float* x;
    uint8_t* y;
    uint T;
};

groupshared float gSplitMax[8];

[shader("compute")]
[numthreads(256, 1, 1)]
void split_rows(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform SplitRowsParams p)
{
    uint row = gid.x, tid = ltid.x;
    float* xr = p.x + uint64_t(row) * X_STRIDE;
    float m = 0;
    for (uint c = tid * 8; c < K; c += 2048)
    {
        float4 a = ((float4*)(xr + c))[0], b = ((float4*)(xr + c))[1];
        m = max(m, max(max(max(abs(a.x), abs(a.y)), max(abs(a.z), abs(a.w))), max(max(abs(b.x), abs(b.y)), max(abs(b.z), abs(b.w)))));
    }
    m = WaveActiveMax(m);
    if (tid % 32 == 0) gSplitMax[tid / 32] = m;
    GroupMemoryBarrierWithGroupSync();
    m = max(max(max(gSplitMax[0], gSplitMax[1]), max(gSplitMax[2], gSplitMax[3])), max(max(gSplitMax[4], gSplitMax[5]), max(gSplitMax[6], gSplitMax[7])));
    float s = split_scale(m);
    uint64_t yr = uint64_t(p.y) + uint64_t(row) * (2 * SPLIT_HALVES * uint64_t(K) + 16);
    for (uint c = tid * 8; c < K; c += 2048)
    {
        float4 a = ((float4*)(xr + c))[0], b = ((float4*)(xr + c))[1];
        float v[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
        uint4 lo;
        uint4 hi = split8_scaled(v, s, lo);
        *(uint4*)(yr + uint64_t(c) * 2) = hi;
#if SPLIT_HALVES == 2
        *(uint4*)(yr + (uint64_t(K) + c) * 2) = lo;
#endif
    }
    if (tid == 0) *(float*)(yr + 2 * SPLIT_HALVES * uint64_t(K)) = 1.0 / s;
}
#endif
