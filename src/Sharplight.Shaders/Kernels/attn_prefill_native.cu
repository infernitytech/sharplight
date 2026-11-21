// Causal prefill attention against an f16 KV cache written in CUDA C++ for the CUDA provider (DeviceFeatures.NativeAttention, compute
// capability 8.0 and later), in the FlashAttention-2 form: a block handles FN_BR = 16 * FN_MR * FN_WARPS consecutive queries of one head;
// each warp owns 16 * FN_MR query rows for the whole key loop, so its query fragments (Q · SCALE in f16), score accumulators,
// softmax statistics and output accumulators stay in registers. The block shares tiles of FN_BC = 64 keys and values, copied with cp.async into a two-stage
// ring of dynamic shared memory (rows of 16-byte chunks, XOR swizzle so the copies and ldmatrix are free of bank conflicts) while the
// previous tile is multiplied: one block barrier per tile. FN_STAGES = 3 keeps two tiles in flight (one block per SM at head size 128,
// where a tile's copy takes about as long as its multiplication). Per tile a warp computes S = Q · Kᵀ (mma.sync.m16n8k16, f32), applies the
// mask only on tiles that cross a row's limits (the diagonal, a window's start), runs the online softmax on the accumulators (row maxima
// and sums over the four lanes of a row, exp2 with log2(e) folded in), and multiplies the probabilities, converted to f16 in the
// accumulator layout that is the A-operand layout of the next MMA, with the value tile (ldmatrix.trans). Tiles a warp's rows cannot see
// are skipped by a warp-uniform branch. Blocks run the longest query tiles first.
//
// Semantics are those of `fa` in attention.slang for the forms it serves: query t (position n_past + t) sees cache keys [rowBeg, n_past +
// t] with rowBeg = 0, or n_past + t + 1 - WINDOW (sliding window), or the start of its CHUNK; HAS_SOFTCAP caps the scores; HAS_SINKS adds
// the per-head sink logit to the softmax denominator; GQA heads read KV head h / (HQ / HKV). MULTI_SEQ: grid z = runs (slot, first row,
// row count, first position), optionally PAGED; MIX_RUN skips runs shorter than MIX_RUN rows (mixed steps).
//
// Macros: HQ, HKV, D, DV (multiples of 64 up to 128), VRS (value row stride), QS0, QS1, SCALE, WINDOW, CHUNK, HAS_SOFTCAP, SOFTCAP,
// HAS_SINKS, MULTI_SEQ, SLOT_CTX, PAGED, PAGE_BS, PAGE_MB, RT_N_PAST, MIX_RUN, FN_WARPS (4 or 8), FN_MR (16-row fragments per
// warp, default 1: 2 measured the same at head size 64 and spilled at 128), FN_STAGES (2 or 3, default 2).
// Grid: x = HQ, y = query tiles (of the longest run), z = runs (1 without MULTI_SEQ).
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
#define FN_THREADS (FN_WARPS * 32)
#define FN_WR (FN_MR * 16)          // query rows per warp
#define FN_BR (FN_WARPS * FN_WR)
#define FN_BC 64
#define FN_KCH (D / 8)              // 16-byte chunks per key row
#define FN_VCH (DV / 8)
#define FN_KT (FN_BC * D)           // halves of a K tile
#define FN_VT (FN_BC * DV)
#define FN_STAGE (FN_KT + FN_VT)    // halves of one stage
#define FN_ND (D / 16)              // K steps of S = Q · Kᵀ
#define FN_NO (DV / 8)              // output fragments of a warp's rows
#define FN_L2E 1.4426950408889634f
#define FN_SCALE ((float)(SCALE))   // the macros are double literals
#define FN_SOFTCAP ((float)(SOFTCAP))
#define FN_NEG (-1.0e30f)

#if (D % 64) != 0 || (DV % 64) != 0 || D > 128 || DV > 128 || (HQ % HKV) != 0
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


extern "C" __global__ void __launch_bounds__(FN_THREADS) attn_prefill_native(FnParams p)
{
    extern __shared__ __align__(128) unsigned char smem[];
    const fn_u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const fn_u32 g = lane >> 2, c4 = lane & 3;
    const fn_u32 h = blockIdx.x, hk = h / (HQ / HKV);
    const fn_u32 blk = gridDim.y - 1 - blockIdx.y;   // longest query tiles first
    fn_u32 qT = p.T, qRow0 = 0, nPast;
    fn_u64 kvBase = 0;
#if MULTI_SEQ
    {
        const int* run = p.runs + blockIdx.z * 4;
        fn_u32 rows = (fn_u32)run[2];
        if (blk * FN_BR >= rows)
        {
            return;
        }
#if MIX_RUN > 0
        if (rows < MIX_RUN)
        {
            return;
        }
#endif
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
            fn_load(p, sbase + st * FN_STAGE * 2, kvBase, hk, jFirst + st * FN_BC, jEnd, tid);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    // This warp's rows: tw0 + 16 mr + g (a) and + 8 (b); Q · SCALE as A fragments.
    const fn_u32 tw0 = t0 + warp * FN_WR;
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
            fn_u32 d = kc * 16 + 2 * c4;
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
            fn_load(p, sbase + ((stage + FN_STAGES - 1) % FN_STAGES) * FN_STAGE * 2, kvBase, hk, ahead, jEnd, tid);
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // Tiles none of the warp's rows sees: past the last row's end, or before the first row's window / chunk.
        if (!warpLive || j0 >= warpEndMax || j0 + FN_BC <= warpBegMin)
        {
            stage = (stage + 1) % FN_STAGES;
            continue;
        }

        const fn_u32 kbase = sbase + stage * FN_STAGE * 2;
        const fn_u32 vbase = kbase + FN_KT * 2;
        float s[FN_MR][8][4];
#pragma unroll
        for (fn_u32 mr = 0; mr < FN_MR; mr++)
        {
#pragma unroll
            for (fn_u32 i = 0; i < 8; i++)
            {
                s[mr][i][0] = s[mr][i][1] = s[mr][i][2] = s[mr][i][3] = 0.0f;
            }
        }
        // S = Q · Kᵀ: matrices of ldmatrix.x4 m = lane / 8: keys nb * 16 + (m / 2) * 8 + lane % 8, chunk kc * 2 + m % 2.
#pragma unroll
        for (fn_u32 nb = 0; nb < 4; nb++)
        {
            const fn_u32 r = nb * 16 + ((lane >> 4) << 3) + (lane & 7);
#pragma unroll
            for (fn_u32 kc = 0; kc < FN_ND; kc++)
            {
                const fn_u32 ch = kc * 2 + ((lane >> 3) & 1);
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

        fn_u32 pf[FN_MR][4][4];   // probabilities as A fragments of P · V (16 keys per K step)
        const bool masked = j0 + FN_BC > warpEndMin || j0 < warpBegMax;
#pragma unroll
        for (fn_u32 mr = 0; mr < FN_MR; mr++)
        {
#if HAS_SOFTCAP
#pragma unroll
            for (fn_u32 i = 0; i < 8; i++)
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
                for (fn_u32 i = 0; i < 8; i++)
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

            // Online softmax of rows a and b (the four lanes of a row hold its 64 scores).
            float xa = s[mr][0][0], xb = s[mr][0][2];
#pragma unroll
            for (fn_u32 i = 0; i < 8; i++)
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
            for (fn_u32 i = 0; i < 8; i++)
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
        for (fn_u32 kk = 0; kk < 4; kk++)
        {
            const fn_u32 r = kk * 16 + (((lane >> 3) & 1) << 3) + (lane & 7);
#pragma unroll
            for (fn_u32 dn = 0; dn < DV / 16; dn++)
            {
                const fn_u32 ch = dn * 2 + (lane >> 4);
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
        float* oa = p.o + ((fn_u64)(qRow0 + ta) * HQ + h) * DV + 2 * c4;
        float* ob = p.o + ((fn_u64)(qRow0 + tb) * HQ + h) * DV + 2 * c4;
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
