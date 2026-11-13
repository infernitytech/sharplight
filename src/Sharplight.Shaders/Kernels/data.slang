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

#ifdef KERNEL_CACHE_WRITE
// ---------------------------------------------------------------------------------------------
// Cache write: x [T, R1, INNER] with strides (S0, S1) → state rows [n_past + t] (dense rows of
// R1*INNER elements) converting to f16 when DST_F16.
// Macros: R1, INNER, S0, S1, DST_F16
struct CacheParams
{
    float* x;
    uint8_t* dst;
    uint* rt;
#if MULTI_SEQ
    int* seq;   // [T, 4] (slot, position, ...): row t goes to cache row slot * SLOT_CTX + position
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
#if MULTI_SEQ
    uint64_t di = (uint64_t(p.seq[t * 4]) * SLOT_CTX + uint(p.seq[t * 4 + 1])) * ROW + r;
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
