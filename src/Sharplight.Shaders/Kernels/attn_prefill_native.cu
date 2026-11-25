// Causal prefill attention against an f16 KV cache written in CUDA C++ for the CUDA provider (DeviceFeatures.NativeAttention, compute
// capability 8.0 and later), in the FlashAttention-2 form: a block handles FN_BR = 16 * FN_MR * FN_WARPS consecutive queries of one head;
// each warp owns 16 * FN_MR query rows for the whole key loop, so its query fragments (Q · SCALE in f16), score accumulators,
// softmax statistics and output accumulators stay in registers. The block shares tiles of FN_BC keys and values (64; 32 above head size
// 128, where a warp's output accumulators alone take 128 registers and a 64-key stage 64 KiB), copied with cp.async into a two-stage
// ring of dynamic shared memory (rows of 16-byte chunks, XOR swizzle so the copies and ldmatrix are free of bank conflicts) while the
// previous tile is multiplied: one block barrier per tile. FN_STAGES = 3 keeps two tiles in flight (one block per SM at head size 128,
// where a tile's copy takes about as long as its multiplication). Per tile a warp computes S = Q · Kᵀ (mma.sync.m16n8k16, f32), applies the
// mask only on tiles that cross a row's limits (the diagonal, a window's start), runs the online softmax on the accumulators (row maxima
// and sums over the four lanes of a row, exp2 with log2(e) folded in), and multiplies the probabilities, converted to f16 in the
// accumulator layout that is the A-operand layout of the next MMA, with the value tile (ldmatrix.trans). Tiles a warp's rows cannot see
// are skipped by a warp-uniform branch. Blocks run the longest query tiles first.
//
// FN_HG: the warps of a block hold the same 16 * FN_MR query rows of FN_WARPS query heads that share a KV head (grid x = HQ / FN_WARPS),
// so a block's key and value tiles serve all of them and its causal diagonal is 16 * FN_MR rows high instead of FN_WARPS times that:
// short prompts (mixed steps of a few hundred rows per run) skip most of the masked half of their tiles.
//
// Head sizes above 256 (Gemma 4's 512) split the head dimensions over a pair of warps (FN_DS = 2): both warps of a pair own the same
// 16 * FN_MR query rows, each holds the query fragments of its half of the key dimensions and the output accumulators of its half of
// the value dimensions, so their registers are those of head size 256. Per tile each warp multiplies its half of Q · Kᵀ, the two
// partial score tiles are exchanged through shared memory (one named barrier per pair and tile; the block barrier at the top of the
// next tile keeps a warp from overwriting its slot before the partner has read it) and
// summed (a + b in both warps, so both see the same scores and run the same softmax), and each warp multiplies P with its half of V.
//
// Semantics are those of `fa` in attention.slang for the forms it serves: query t (position n_past + t) sees cache keys [rowBeg, n_past +
// t] with rowBeg = 0, or n_past + t + 1 - WINDOW (sliding window), or the start of its CHUNK; HAS_SOFTCAP caps the scores; HAS_SINKS adds
// the per-head sink logit to the softmax denominator; GQA heads read KV head h / (HQ / HKV). MULTI_SEQ: runs (slot, first row,
// row count, first position), optionally PAGED; MIX_RUN skips runs shorter than MIX_RUN rows (mixed steps).
//
// FN_KVF8 (head sizes up to 256): f8e4m3_b32 caches (32-value blocks of an f32 scale and 32 E4M3 codes, 36 bytes). The stages hold the
// tiles' raw rows (FN_RB / FN_VB bytes, cp.async in 16-byte pieces where rows are 16-byte multiples, else 8); after a tile's copy
// completes the block converts it into one f16 tile (the layout of an f16 stage; d * e4m3 rounded once, as attention.slang's kv8_to_half),
// and a second barrier publishes it before the products. The next iteration's first barrier keeps the conversion from overwriting a
// tile a warp still reads.
//
// Macros: HQ, HKV, D, DV (multiples of 64 up to 256), VRS (value row stride), QS0, QS1, SCALE, WINDOW, CHUNK, HAS_SOFTCAP, SOFTCAP,
// HAS_SINKS, MULTI_SEQ, SLOT_CTX, PAGED, PAGE_BS, PAGE_MB, RT_N_PAST, MIX_RUN, FN_WARPS (4 or 8), FN_MR (16-row fragments per
// warp, default 1; the lowering takes 2 at head size 64, where deep contexts gain 7-12 %, and 1 at 128, where 2 spills), FN_STAGES
// (2 or 3, default 2), FN_BC (keys per tile, 16, 32 or 64; default 64 up to head size 128, 32 up to 256, else 16).
// Dynamic shared memory: FN_STAGES stages of FN_BC key and value rows, then (FN_DS = 2) the score exchange of FN_XCH floats.
// Grid: x = HQ, y = query tiles: without MULTI_SEQ those of the T rows; with MULTI_SEQ ceil(T / FN_BR) + runs, a bound on the tiles of all
// runs together, which the blocks enumerate run by run (the run count is grid y minus ceil(T / FN_BR)); z = 1.
#ifndef FN_WARPS
#define FN_WARPS 4
#endif
#ifndef WINDOW
#define WINDOW 0
#endif
#ifndef CHUNK
#define CHUNK 0
#endif
#ifndef HAS_SOFTCAP
#define HAS_SOFTCAP 0
#endif
#ifndef HAS_SINKS
#define HAS_SINKS 0
#endif
#ifndef MULTI_SEQ
#define MULTI_SEQ 0
#endif
#ifndef PAGED
#define PAGED 0
#endif
#ifndef MIX_RUN
#define MIX_RUN 0
#endif
#ifndef VRS
#define VRS DV
#endif
#ifndef FN_MR
#define FN_MR 1
#endif
#ifndef FN_STAGES
#define FN_STAGES 2
#endif
#ifndef FN_HG
#define FN_HG 0
#endif
#define FN_THREADS (FN_WARPS * 32)
#define FN_DS ((D > 256 || DV > 256) ? 2 : 1)   // warps sharing a row group (head dimensions split between them)
#define FN_WR (FN_MR * 16)          // query rows per warp
#if FN_HG
#define FN_BR FN_WR                 // FN_HG: the warps hold the same rows of FN_WARPS query heads of one KV head
#else
#define FN_BR (FN_WARPS / FN_DS * FN_WR)
#endif
#ifndef FN_BC
#define FN_BC ((D > 256 || DV > 256) ? 16 : (D > 128 || DV > 128) ? 32 : 64)
#endif
#define FN_NB (FN_BC / 8)           // 8-key score fragments of a tile
#define FN_KK (FN_BC / 16)          // 16-key K steps of P · V
#define FN_KCH (D / 8)              // 16-byte chunks per key row
#define FN_VCH (DV / 8)
#define FN_KT (FN_BC * D)           // halves of a K tile
#define FN_VT (FN_BC * DV)
#define FN_STAGE (FN_KT + FN_VT)    // halves of one stage (FN_KVF8: of the converted tile)
#ifndef FN_KVF8
#define FN_KVF8 0
#endif
#if FN_KVF8
#define FN_RB (D / 32 * 36)         // bytes of a raw key row
#define FN_VB (DV / 32 * 36)        // bytes of a raw value row (the first DV values of a cache row of VRS)
#define FN_VRB (VRS / 32 * 36)      // bytes between value rows in the cache
#define FN_CP ((FN_RB % 16 == 0 && FN_VB % 16 == 0 && FN_VRB % 16 == 0) ? 16 : 8)   // bytes per cp.async
#define FN_RSTAGE (FN_BC * (FN_RB + FN_VB))   // bytes of one raw stage
#define FN_TILE0 ((FN_STAGES * FN_RSTAGE + 127) / 128 * 128)   // byte offset of the converted tile
#else
#define FN_RSTAGE (FN_STAGE * 2)
#endif
#define FN_ND (D / FN_DS / 16)      // K steps of a warp's S = Q · Kᵀ
#define FN_NO (DV / FN_DS / 8)      // output fragments of a warp's rows
#define FN_XCH (FN_WARPS * FN_MR * FN_NB * 4 * 32)   // floats of the score exchange (FN_DS = 2)
#define FN_L2E 1.4426950408889634f
#define FN_SCALE ((float)(SCALE))   // the macros are double literals
#define FN_SOFTCAP ((float)(SOFTCAP))
#define FN_NEG (-1.0e30f)

#if (D % (64 * FN_DS)) != 0 || (DV % (64 * FN_DS)) != 0 || D > 512 || DV > 512 || (HQ % HKV) != 0 || (FN_BC != 16 && FN_BC != 32 && FN_BC != 64) \
    || (FN_WARPS % FN_DS) != 0 || (FN_HG && (FN_DS != 1 || ((HQ / HKV) % FN_WARPS) != 0)) || (FN_KVF8 && FN_DS != 1)
#error "attn_prefill_native: unsupported head sizes"
#endif

typedef unsigned int fn_u32;
typedef unsigned long long fn_u64;

struct FnParams
{
    const float* q;
    const unsigned short* kc;
    const unsigned short* vc;
    float* o;
    float* part;
    const float* sinks;
    const fn_u32* rt;
#if MULTI_SEQ
    const int* seq;
    const int* runs;
#endif
#if PAGED
    const int* pages;
#endif
    fn_u32* arrivals;
    fn_u32 T;
    fn_u32 row0;
};

__device__ __forceinline__ fn_u32 fn_smem(const void* p)
{
    return (fn_u32)__cvta_generic_to_shared(p);
}

// 16 bytes from global to shared; src_bytes 0 zero-fills (rows past the keys a block reads).
__device__ __forceinline__ void fn_copy(fn_u32 dst, const void* src, fn_u32 srcBytes)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(srcBytes));
}

__device__ __forceinline__ void fn_ldsm4(fn_u32 addr, fn_u32& r0, fn_u32& r1, fn_u32& r2, fn_u32& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void fn_ldsm4t(fn_u32 addr, fn_u32& r0, fn_u32& r1, fn_u32& r2, fn_u32& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void fn_mma(float* c, const fn_u32* a, fn_u32 b0, fn_u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two floats as f16x2 (lo in the low half), round to nearest.
__device__ __forceinline__ fn_u32 fn_pack(float lo, float hi)
{
    fn_u32 r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ float fn_exp2(float x)
{
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
    return r;
}

// Cache row (x HKV) of key j: contiguous slots or the block table of a paged cache.
__device__ __forceinline__ fn_u64 fn_kv_row(const FnParams& p, fn_u64 base, fn_u32 j)
{
#if PAGED
    int block = p.pages[base * PAGE_MB + min(j / PAGE_BS, (fn_u32)(PAGE_MB - 1))];
    return ((fn_u64)(fn_u32)max(block, 0) * PAGE_BS + j % PAGE_BS) * HKV;
#else
    return base + (fn_u64)j * HKV;
#endif
}

#if FN_KVF8
__device__ __forceinline__ void fn_copy8(fn_u32 dst, const void* src, fn_u32 srcBytes)
{
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" :: "r"(dst), "l"(src), "r"(srcBytes));
}

// Issues the copies of the raw key and value rows j0 .. j0 + FN_BC (rows >= jEnd zero-filled) into the raw stage at `sbase`.
__device__ __forceinline__ void fn_load(const FnParams& p, fn_u32 sbase, fn_u64 kvBase, fn_u32 hk, fn_u32 j0, fn_u32 jEnd, fn_u32 tid)
{
    const unsigned char* kc = (const unsigned char*)p.kc;
    const unsigned char* vc = (const unsigned char*)p.vc;
#pragma unroll
    for (fn_u32 i = tid; i < FN_BC * (FN_RB / FN_CP); i += FN_THREADS)
    {
        fn_u32 r = i / (FN_RB / FN_CP), c = i % (FN_RB / FN_CP), j = j0 + r;
        bool in = j < jEnd;
        fn_u64 row = fn_kv_row(p, kvBase, in ? j : j0);
        const unsigned char* src = kc + (row + hk) * FN_RB + c * FN_CP;
        if (FN_CP == 16)
        {
            fn_copy(sbase + r * FN_RB + c * 16, src, in ? 16 : 0);
        }
        else
        {
            fn_copy8(sbase + r * FN_RB + c * 8, src, in ? 8 : 0);
        }
    }
#pragma unroll
    for (fn_u32 i = tid; i < FN_BC * (FN_VB / FN_CP); i += FN_THREADS)
    {
        fn_u32 r = i / (FN_VB / FN_CP), c = i % (FN_VB / FN_CP), j = j0 + r;
        bool in = j < jEnd;
        fn_u64 row = fn_kv_row(p, kvBase, in ? j : j0);
        const unsigned char* src = vc + (row + hk) * FN_VRB + c * FN_CP;
        if (FN_CP == 16)
        {
            fn_copy(sbase + FN_BC * FN_RB + r * FN_VB + c * 16, src, in ? 16 : 0);
        }
        else
        {
            fn_copy8(sbase + FN_BC * FN_RB + r * FN_VB + c * 8, src, in ? 8 : 0);
        }
    }
}

// Two E4M3 codes (bits 0..15 of x) times d256 = 256 * the block scale, as f16x2 (the codes' bits placed as half bits are the values * 2^-8).
__device__ __forceinline__ fn_u32 fn_f8x2(fn_u32 x, float d256)
{
    fn_u32 h = ((x & 0x7Fu) << 7) | ((x & 0x80u) << 8) | ((x & 0x7F00u) << 15) | ((x & 0x8000u) << 16);
    float lo, hi;
    asm("cvt.f32.f16 %0, %1;\n" : "=f"(lo) : "h"((unsigned short)(h & 0xFFFFu)));
    asm("cvt.f32.f16 %0, %1;\n" : "=f"(hi) : "h"((unsigned short)(h >> 16)));
    return fn_pack(lo * d256, hi * d256);
}

// Converts the raw rows of a stage (rows of `width` values, `rowBytes` apart) into the f16 tile at `tile` (16-byte chunks with the
// stages' swizzle): one 8-value chunk per thread and pass.
__device__ __forceinline__ void fn_convert(const unsigned char* raw, unsigned short* tile, fn_u32 width, fn_u32 rowBytes, fn_u32 tid)
{
    const fn_u32 chunks = width / 8;
    for (fn_u32 i = tid; i < FN_BC * chunks; i += FN_THREADS)
    {
        fn_u32 r = i / chunks, c = i % chunks;
        const unsigned char* blk = raw + r * rowBytes + (c / 4) * 36;
        const float d256 = *(const float*)blk * 256.0f;
        const fn_u32 q0 = *(const fn_u32*)(blk + 4 + (c % 4) * 8), q1 = *(const fn_u32*)(blk + 8 + (c % 4) * 8);
        *(uint4*)(tile + r * width + ((c ^ (r & 7)) * 8)) = make_uint4(fn_f8x2(q0, d256), fn_f8x2(q0 >> 16, d256), fn_f8x2(q1, d256),
                                                                       fn_f8x2(q1 >> 16, d256));
    }
}
#else
// Issues the copies of the key and value rows j0 .. j0 + FN_BC (rows >= jEnd zero-filled) into stage `st`.
__device__ __forceinline__ void fn_load(const FnParams& p, fn_u32 sbase, fn_u64 kvBase, fn_u32 hk, fn_u32 j0, fn_u32 jEnd, fn_u32 tid)
{
#pragma unroll
    for (fn_u32 i = tid; i < FN_BC * FN_KCH; i += FN_THREADS)
    {
        fn_u32 r = i / FN_KCH, c = i % FN_KCH, j = j0 + r;
        bool in = j < jEnd;
        fn_u64 row = fn_kv_row(p, kvBase, in ? j : j0);
        const unsigned short* src = p.kc + (row + hk) * D + c * 8;
        fn_copy(sbase + (r * D + ((c ^ (r & 7)) * 8)) * 2, src, in ? 16 : 0);
    }
#pragma unroll
    for (fn_u32 i = tid; i < FN_BC * FN_VCH; i += FN_THREADS)
    {
        fn_u32 r = i / FN_VCH, c = i % FN_VCH, j = j0 + r;
        bool in = j < jEnd;
        fn_u64 row = fn_kv_row(p, kvBase, in ? j : j0);
        const unsigned short* src = p.vc + (row + hk) * VRS + c * 8;
        fn_copy(sbase + (FN_KT + r * DV + ((c ^ (r & 7)) * 8)) * 2, src, in ? 16 : 0);
    }
}
#endif


extern "C" __global__ void __launch_bounds__(FN_THREADS) attn_prefill_native(FnParams p)
{
    extern __shared__ __align__(128) unsigned char smem[];
    const fn_u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const fn_u32 rowGroup = warp / FN_DS, part = warp % FN_DS;   // the warp's rows and its half of the head dimensions
    const fn_u32 g = lane >> 2, c4 = lane & 3;
#if FN_HG
    const fn_u32 h = blockIdx.x * FN_WARPS + warp, hk = blockIdx.x * FN_WARPS / (HQ / HKV);
#else
    const fn_u32 h = blockIdx.x, hk = h / (HQ / HKV);
#endif
    fn_u32 qT = p.T, qRow0 = 0, nPast;
    fn_u64 kvBase = 0;
#if MULTI_SEQ
    fn_u32 blk;
    {
        // Grid y enumerates the query tiles of all runs the kernel serves (runs of at least MIX_RUN rows), each run's longest tiles
        // first; the lowering sizes it ceil(T / FN_BR) + runs, a bound on that count, so the run count is the rest of grid y.
        const fn_u32 nRuns = gridDim.y - (p.T + FN_BR - 1) / FN_BR;
        const fn_u32 L = blockIdx.y;
        fn_u32 acc = 0, rows = 0;
        int found = -1;
        for (fn_u32 r0 = 0; r0 < nRuns; r0 += 32)
        {
            const fn_u32 r = r0 + lane;
            fn_u32 rr = r < nRuns ? (fn_u32)p.runs[r * 4 + 2] : 0;
#if MIX_RUN > 0
            rr = rr < MIX_RUN ? 0 : rr;
#endif
            const fn_u32 n = (rr + FN_BR - 1) / FN_BR;
            fn_u32 incl = n;
#pragma unroll
            for (fn_u32 off = 1; off < 32; off <<= 1)
            {
                const fn_u32 v = __shfl_up_sync(0xffffffffu, incl, off);
                incl += lane >= off ? v : 0;
            }

            const fn_u32 total = __shfl_sync(0xffffffffu, incl, 31);
            if (L < acc + total)
            {
                const fn_u32 first = __ffs(__ballot_sync(0xffffffffu, L < acc + incl)) - 1;
                const fn_u32 before = __shfl_sync(0xffffffffu, incl - n, first);
                rows = __shfl_sync(0xffffffffu, rr, first);
                const fn_u32 tiles = (rows + FN_BR - 1) / FN_BR;
                blk = tiles - 1 - (L - acc - before);   // longest query tiles first
                found = (int)(r0 + first);
                break;
            }

            acc += total;
        }

        if (found < 0)
        {
            return;
        }

        const int* run = p.runs + found * 4;
        qT = rows;
        qRow0 = (fn_u32)run[1];
        nPast = (fn_u32)run[3];
#if PAGED
        kvBase = (fn_u32)run[0];
#else
        kvBase = (fn_u64)(fn_u32)run[0] * SLOT_CTX * HKV;
#endif
    }
#else
    const fn_u32 blk = gridDim.y - 1 - blockIdx.y;   // longest query tiles first
    nPast = p.rt[RT_N_PAST];
    if (blk * FN_BR >= qT)
    {
        return;
    }
#endif
    const fn_u32 t0 = blk * FN_BR;
    // Keys the block reads: from the first query's window / chunk start to the last query's causal end.
    const fn_u32 jEnd = nPast + min(qT, t0 + FN_BR);
    fn_u32 jFirst = 0;
#if WINDOW
    jFirst = nPast + t0 + 1 > WINDOW ? (nPast + t0 + 1 - WINDOW) / FN_BC * FN_BC : 0;
#endif
#if CHUNK
    jFirst = max(jFirst, (nPast + t0) / CHUNK * CHUNK / FN_BC * FN_BC);
#endif

    const fn_u32 sbase = fn_smem(smem);
#pragma unroll
    for (fn_u32 st = 0; st < FN_STAGES - 1; st++)
    {
        if (jFirst + st * FN_BC < jEnd)
        {
            fn_load(p, sbase + st * FN_RSTAGE, kvBase, hk, jFirst + st * FN_BC, jEnd, tid);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    // This warp's rows: tw0 + 16 mr + g (a) and + 8 (b); Q · SCALE as A fragments.
    const fn_u32 tw0 = t0 + (FN_HG ? 0 : rowGroup * FN_WR);
    fn_u32 qa[FN_MR][FN_ND][4];
#pragma unroll
    for (fn_u32 mr = 0; mr < FN_MR; mr++)
    {
        const fn_u32 ta = tw0 + mr * 16 + g, tb = ta + 8;
        const float* qra = p.q + (fn_u64)(qRow0 + min(ta, qT - 1)) * QS0 + (fn_u64)h * QS1;
        const float* qrb = p.q + (fn_u64)(qRow0 + min(tb, qT - 1)) * QS0 + (fn_u64)h * QS1;
#pragma unroll
        for (fn_u32 kc = 0; kc < FN_ND; kc++)
        {
            fn_u32 d = part * (D / FN_DS) + kc * 16 + 2 * c4;
            float2 a0 = *(const float2*)(qra + d), a1 = *(const float2*)(qrb + d);
            float2 a2 = *(const float2*)(qra + d + 8), a3 = *(const float2*)(qrb + d + 8);
            qa[mr][kc][0] = fn_pack(a0.x * FN_SCALE, a0.y * FN_SCALE);
            qa[mr][kc][1] = fn_pack(a1.x * FN_SCALE, a1.y * FN_SCALE);
            qa[mr][kc][2] = fn_pack(a2.x * FN_SCALE, a2.y * FN_SCALE);
            qa[mr][kc][3] = fn_pack(a3.x * FN_SCALE, a3.y * FN_SCALE);
        }
    }

    // Ends of the warp's first and last rows, and the window / chunk starts of its first and last rows.
    const fn_u32 warpEndMin = nPast + tw0 + 1, warpEndMax = nPast + tw0 + FN_WR;
    fn_u32 warpBegMax = 0, warpBegMin = 0;
#if WINDOW
    warpBegMax = warpEndMax > WINDOW ? warpEndMax - WINDOW : 0;
    warpBegMin = warpEndMin > WINDOW ? warpEndMin - WINDOW : 0;
#endif
#if CHUNK
    warpBegMax = max(warpBegMax, (warpEndMax - 1) / CHUNK * CHUNK);
    warpBegMin = max(warpBegMin, (warpEndMin - 1) / CHUNK * CHUNK);
#endif
    const bool warpLive = tw0 < qT;

    float o[FN_MR][FN_NO][4];
    float m[FN_MR][2], l[FN_MR][2];
#pragma unroll
    for (fn_u32 mr = 0; mr < FN_MR; mr++)
    {
#pragma unroll
        for (fn_u32 i = 0; i < FN_NO; i++)
        {
            o[mr][i][0] = o[mr][i][1] = o[mr][i][2] = o[mr][i][3] = 0.0f;
        }
        m[mr][0] = m[mr][1] = FN_NEG;
        l[mr][0] = l[mr][1] = 0.0f;
    }

    fn_u32 stage = 0;
    for (fn_u32 j0 = jFirst; j0 < jEnd; j0 += FN_BC)
    {
        // Tile j0 complete (the FN_STAGES - 2 later tiles may still be in flight); the slot of the previous tile, which every warp
        // has finished reading, takes the tile FN_STAGES - 1 ahead.
        asm volatile("cp.async.wait_group %0;\n" :: "n"(FN_STAGES - 2));
        __syncthreads();
        const fn_u32 ahead = j0 + (FN_STAGES - 1) * FN_BC;
        if (ahead < jEnd)
        {
            fn_load(p, sbase + ((stage + FN_STAGES - 1) % FN_STAGES) * FN_RSTAGE, kvBase, hk, ahead, jEnd, tid);
        }
        asm volatile("cp.async.commit_group;\n" ::);
#if FN_KVF8
        // Tile j0's raw rows into the f16 tile (every warp takes part, also those that skip the tile).
        {
            const unsigned char* raw = smem + stage * FN_RSTAGE;
            unsigned short* tile = (unsigned short*)(smem + FN_TILE0);
            fn_convert(raw, tile, D, FN_RB, tid);
            fn_convert(raw + FN_BC * FN_RB, tile + FN_KT, DV, FN_VB, tid);
        }
        __syncthreads();
#endif

        // Tiles none of the warp's rows sees: past the last row's end, or before the first row's window / chunk.
        if (!warpLive || j0 >= warpEndMax || j0 + FN_BC <= warpBegMin)
        {
            stage = (stage + 1) % FN_STAGES;
            continue;
        }

#if FN_KVF8
        const fn_u32 kbase = sbase + FN_TILE0;
#else
        const fn_u32 kbase = sbase + stage * FN_STAGE * 2;
#endif
        const fn_u32 vbase = kbase + FN_KT * 2;
        float s[FN_MR][FN_NB][4];
#pragma unroll
        for (fn_u32 mr = 0; mr < FN_MR; mr++)
        {
#pragma unroll
            for (fn_u32 i = 0; i < FN_NB; i++)
            {
                s[mr][i][0] = s[mr][i][1] = s[mr][i][2] = s[mr][i][3] = 0.0f;
            }
        }
        // S = Q · Kᵀ: matrices of ldmatrix.x4 m = lane / 8: keys nb * 16 + (m / 2) * 8 + lane % 8, chunk kc * 2 + m % 2.
#pragma unroll
        for (fn_u32 nb = 0; nb < FN_KK; nb++)
        {
            const fn_u32 r = nb * 16 + ((lane >> 4) << 3) + (lane & 7);
#pragma unroll
            for (fn_u32 kc = 0; kc < FN_ND; kc++)
            {
                const fn_u32 ch = part * (D / FN_DS / 8) + kc * 2 + ((lane >> 3) & 1);
                fn_u32 b0, b1, b2, b3;
                fn_ldsm4(kbase + (r * D + ((ch ^ (r & 7)) * 8)) * 2, b0, b1, b2, b3);
#pragma unroll
                for (fn_u32 mr = 0; mr < FN_MR; mr++)
                {
                    fn_mma(s[mr][2 * nb], qa[mr][kc], b0, b1);
                    fn_mma(s[mr][2 * nb + 1], qa[mr][kc], b2, b3);
                }
            }
        }

#if FN_DS == 2
        // The pair's partial scores: this warp's into its slot of the tile's buffer, then the partner's added (a + b alike in both).
        {
            float* xch = (float*)(smem + FN_STAGES * FN_STAGE * 2);
            float* mine = xch + warp * (FN_MR * FN_NB * 4 * 32);
            const float* other = xch + (warp ^ 1) * (FN_MR * FN_NB * 4 * 32);
#pragma unroll
            for (fn_u32 mr = 0; mr < FN_MR; mr++)
            {
#pragma unroll
                for (fn_u32 i = 0; i < FN_NB; i++)
                {
#pragma unroll
                    for (fn_u32 e = 0; e < 4; e++)
                    {
                        mine[((mr * FN_NB + i) * 4 + e) * 32 + lane] = s[mr][i][e];
                    }
                }
            }
            asm volatile("bar.sync %0, 64;\n" :: "r"(1 + rowGroup));
#pragma unroll
            for (fn_u32 mr = 0; mr < FN_MR; mr++)
            {
#pragma unroll
                for (fn_u32 i = 0; i < FN_NB; i++)
                {
#pragma unroll
                    for (fn_u32 e = 0; e < 4; e++)
                    {
                        const float x = other[((mr * FN_NB + i) * 4 + e) * 32 + lane];
                        s[mr][i][e] = part == 0 ? s[mr][i][e] + x : x + s[mr][i][e];
                    }
                }
            }
        }
#endif

        fn_u32 pf[FN_MR][FN_KK][4];   // probabilities as A fragments of P · V (16 keys per K step)
        const bool masked = j0 + FN_BC > warpEndMin || j0 < warpBegMax;
#pragma unroll
        for (fn_u32 mr = 0; mr < FN_MR; mr++)
        {
#if HAS_SOFTCAP
#pragma unroll
            for (fn_u32 i = 0; i < FN_NB; i++)
            {
#pragma unroll
                for (fn_u32 e = 0; e < 4; e++)
                {
                    s[mr][i][e] = FN_SOFTCAP * tanhf(s[mr][i][e] * (1.0f / FN_SOFTCAP));
                }
            }
#endif
            // Mask on tiles that cross a row's limits [beg, end).
            if (masked)
            {
                const fn_u32 endA = nPast + tw0 + mr * 16 + g + 1, endB = endA + 8;
                fn_u32 begA = 0, begB = 0;
#if WINDOW
                begA = endA > WINDOW ? endA - WINDOW : 0;
                begB = endB > WINDOW ? endB - WINDOW : 0;
#endif
#if CHUNK
                begA = max(begA, (endA - 1) / CHUNK * CHUNK);
                begB = max(begB, (endB - 1) / CHUNK * CHUNK);
#endif
#pragma unroll
                for (fn_u32 i = 0; i < FN_NB; i++)
                {
#pragma unroll
                    for (fn_u32 e = 0; e < 2; e++)
                    {
                        fn_u32 j = j0 + i * 8 + 2 * c4 + e;
                        if (j >= endA || j < begA)
                        {
                            s[mr][i][e] = FN_NEG;
                        }
                        if (j >= endB || j < begB)
                        {
                            s[mr][i][2 + e] = FN_NEG;
                        }
                    }
                }
            }

            // Online softmax of rows a and b (the four lanes of a row hold its FN_BC scores).
            float xa = s[mr][0][0], xb = s[mr][0][2];
#pragma unroll
            for (fn_u32 i = 0; i < FN_NB; i++)
            {
                xa = fmaxf(xa, fmaxf(s[mr][i][0], s[mr][i][1]));
                xb = fmaxf(xb, fmaxf(s[mr][i][2], s[mr][i][3]));
            }
            xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, 1));
            xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, 2));
            xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, 1));
            xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, 2));
            const float nA = fmaxf(m[mr][0], xa), nB = fmaxf(m[mr][1], xb);
            const float corrA = fn_exp2((m[mr][0] - nA) * FN_L2E), corrB = fn_exp2((m[mr][1] - nB) * FN_L2E);
            m[mr][0] = nA;
            m[mr][1] = nB;
            // A row without a visible key so far keeps the sentinel maximum: its masked scores must still give 0.
            const float sa = nA == FN_NEG ? 0.0f : nA * FN_L2E, sb = nB == FN_NEG ? 0.0f : nB * FN_L2E;
            float pa = 0.0f, pb = 0.0f;
#pragma unroll
            for (fn_u32 i = 0; i < FN_NB; i++)
            {
                float p0 = fn_exp2(s[mr][i][0] * FN_L2E - sa);
                float p1 = fn_exp2(s[mr][i][1] * FN_L2E - sa);
                float p2 = fn_exp2(s[mr][i][2] * FN_L2E - sb);
                float p3 = fn_exp2(s[mr][i][3] * FN_L2E - sb);
                pa += p0 + p1;
                pb += p2 + p3;
                pf[mr][i / 2][(i & 1) * 2] = fn_pack(p0, p1);
                pf[mr][i / 2][(i & 1) * 2 + 1] = fn_pack(p2, p3);
            }
            l[mr][0] = l[mr][0] * corrA + pa;
            l[mr][1] = l[mr][1] * corrB + pb;
#pragma unroll
            for (fn_u32 i = 0; i < FN_NO; i++)
            {
                o[mr][i][0] *= corrA;
                o[mr][i][1] *= corrA;
                o[mr][i][2] *= corrB;
                o[mr][i][3] *= corrB;
            }
        }

        // O += P · V: ldmatrix.x4.trans matrices m = lane / 8: keys kk * 16 + (m % 2) * 8 + lane % 8, chunk dn * 2 + m / 2.
#pragma unroll
        for (fn_u32 kk = 0; kk < FN_KK; kk++)
        {
            const fn_u32 r = kk * 16 + (((lane >> 3) & 1) << 3) + (lane & 7);
#pragma unroll
            for (fn_u32 dn = 0; dn < DV / FN_DS / 16; dn++)
            {
                const fn_u32 ch = part * (DV / FN_DS / 8) + dn * 2 + (lane >> 4);
                fn_u32 b0, b1, b2, b3;
                fn_ldsm4t(vbase + (r * DV + ((ch ^ (r & 7)) * 8)) * 2, b0, b1, b2, b3);
#pragma unroll
                for (fn_u32 mr = 0; mr < FN_MR; mr++)
                {
                    fn_mma(o[mr][2 * dn], pf[mr][kk], b0, b1);
                    fn_mma(o[mr][2 * dn + 1], pf[mr][kk], b2, b3);
                }
            }
        }
        stage = (stage + 1) % FN_STAGES;
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

#pragma unroll
    for (fn_u32 mr = 0; mr < FN_MR; mr++)
    {
        float lA = l[mr][0], lB = l[mr][1];
        lA += __shfl_xor_sync(0xffffffffu, lA, 1);
        lA += __shfl_xor_sync(0xffffffffu, lA, 2);
        lB += __shfl_xor_sync(0xffffffffu, lB, 1);
        lB += __shfl_xor_sync(0xffffffffu, lB, 2);
        float fA = 1.0f, fB = 1.0f;
#if HAS_SINKS
        {
            const float sk = p.sinks[h], mA = m[mr][0], mB = m[mr][1];
            const float xA = fmaxf(mA, sk), xB = fmaxf(mB, sk);
            fA = __expf(mA - xA);
            fB = __expf(mB - xB);
            lA = lA * fA + __expf(sk - xA);
            lB = lB * fB + __expf(sk - xB);
        }
#endif
        fA = lA > 0.0f ? fA / lA : 0.0f;
        fB = lB > 0.0f ? fB / lB : 0.0f;
        const fn_u32 ta = tw0 + mr * 16 + g, tb = ta + 8;
        float* oa = p.o + ((fn_u64)(qRow0 + ta) * HQ + h) * DV + part * (DV / FN_DS) + 2 * c4;
        float* ob = p.o + ((fn_u64)(qRow0 + tb) * HQ + h) * DV + part * (DV / FN_DS) + 2 * c4;
#pragma unroll
        for (fn_u32 i = 0; i < FN_NO; i++)
        {
            if (ta < qT)
            {
                *(float2*)(oa + i * 8) = make_float2(o[mr][i][0] * fA, o[mr][i][1] * fA);
            }
            if (tb < qT)
            {
                *(float2*)(ob + i * 8) = make_float2(o[mr][i][2] * fB, o[mr][i][3] * fB);
            }
        }
    }
}
