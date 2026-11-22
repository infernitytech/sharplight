// Split-K decode attention on tensor cores against an f16 KV cache, written in CUDA C++ for the CUDA provider (DeviceFeatures.NativeAttention,
// compute capability 8.0 and later): the form of `decode` in attention.slang (same grid, same key ranges, same partials for decode_combine)
// with the HPW query heads of a KV group as the rows of mma.sync.m16n8k16 (rows HPW .. 15 are zero). A block handles one (KV group, query
// row, key range); its ND_WARPS warps take interleaved tiles of ND_TK keys and each runs its own online softmax in registers, so the key
// loop has no shared memory and no barriers:
//
//   * every lane group g (4 lanes) loads key and value rows g, g + 8, .. of the tile straight from the cache in 16-byte chunks (chunk
//     4 q + c of the row for lane c), and these chunks are the B operands of S = Q · Kᵀ as they are: the head dimensions are permuted
//     consistently between Q and K (slot 2c + e of K step kk holds the dimension of word 2kk of lane c's chunks, slot 2c + 8 + e that of
//     word 2kk + 1), which a dot product does not see;
//   * the scores' accumulators are the A operands of O += P · V (converted to f16, the FlashAttention-2 register reuse), and the value
//     words become B operands by an in-register 8 x 8 transpose (movmatrix), so value dimension word f of lane c is output column 2c of
//     fragment f; the outputs are stored in that permuted order's inverse.
//
// The warps' (m, l, O) are merged in shared memory at the end; the result is written directly (one key range, sinks in the denominator)
// or as the range's partial [split][t][h][DV + 4] (acc, m, l) for decode_combine. Q · SCALE and P are rounded to f16 as in
// attn_prefill_native.cu.
//
// Macros: HQ, HKV, D (= DV, 64 or 128), VRS, QS0, QS1, SCALE, WINDOW, CHUNK, HAS_SOFTCAP, SOFTCAP, HAS_SINKS, MULTI_SEQ, SLOT_CTX, PAGED,
// PAGE_BS, PAGE_MB, RT_N_PAST, MIX_RUN, HPW (heads per block, at most 8), DEC_BC, DEC_MIN_KEYS, DEC_MAX_SPLITS, DEC_TARGET_WG,
// DEC_INVARIANT_ROWS, DEC_SPLIT_ROWS (the split rule of decode_split_keys), ND_PF (L2 prefetch distance in tiles, 0: off). Grid: x = HKV * (HQ / HKV / HPW), y = query rows, z = key ranges.
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
#define VRS D
#endif
#ifndef ND_WARPS
#define ND_WARPS 4
#endif
#ifndef ND_PF
#define ND_PF 1                       // tiles ahead each warp prefetches into L2 (0: none)
#endif
#define ND_THREADS (ND_WARPS * 32)
#define ND_TK (D == 64 ? 32 : 16)     // keys per warp tile
#define ND_R (ND_TK / 8)              // key rows per lane group and tile
#define ND_NCH (D / 32)               // 16-byte chunks per lane and row
#define ND_NW (D / 8)                 // 32-bit words per lane and row (each two f16 values)
#define ND_KS (D / 16)                // K steps of S = Q · Kᵀ
#define ND_G (HQ / HKV)
#define ND_GROUPS (HKV * (ND_G / HPW))
#define ND_PS (D + 4)                 // floats per partial (DEC_PS)
#define ND_LPR (D / 64)               // 128-byte lines per key / value row
#define ND_SCALE ((float)(SCALE))
#define ND_SOFTCAP ((float)(SOFTCAP))
#define ND_NEG (-1.0e30f)

#if (D != 64 && D != 128) || HPW > 8 || (ND_G % HPW) != 0 || (HQ % HKV) != 0
#error "attn_decode_native: unsupported shape"
#endif

typedef unsigned int nd_u32;
typedef unsigned long long nd_u64;

// The argument block of `decode` (attention.slang Params) for the forms this kernel serves.
struct NdParams
{
    const float* q;
    const unsigned short* kc;
    const unsigned short* vc;
    float* o;
    float* part;
    const float* sinks;
    const nd_u32* rt;
#if MULTI_SEQ
    const int* seq;
    const int* runs;
#endif
#if PAGED
    const int* pages;
#endif
    nd_u32* arrivals;
    nd_u32 T;
    nd_u32 row0;
};

__device__ __forceinline__ void nd_mma(float* c, nd_u32 a0, nd_u32 a2, nd_u32 b0, nd_u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

// The 8 x 8 matrix of b16 values whose row l / 4 holds lane l's word at column 2 (l % 4), transposed.
__device__ __forceinline__ nd_u32 nd_trans(nd_u32 x)
{
    nd_u32 r;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(r) : "r"(x));
    return r;
}

// Two floats as f16x2 (lo in the low half), round to nearest.
__device__ __forceinline__ nd_u32 nd_pack(float lo, float hi)
{
    nd_u32 r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ nd_u32 nd_word(const uint4& v, nd_u32 i)
{
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// Cache row (x HKV) of key j: contiguous slots or the block table of a paged cache.
__device__ __forceinline__ nd_u64 nd_kv_row(const NdParams& p, nd_u64 base, nd_u32 j)
{
#if PAGED
    int block = p.pages[base * PAGE_MB + min(j / PAGE_BS, (nd_u32)(PAGE_MB - 1))];
    return ((nd_u64)(nd_u32)max(block, 0) * PAGE_BS + j % PAGE_BS) * HKV;
#else
    return base + (nd_u64)j * HKV;
#endif
}

// decode_max_splits and decode_split_keys of attention.slang (the combine derives the same range count from them).
__device__ __forceinline__ nd_u32 nd_max_splits(nd_u32 T)
{
#ifdef DEC_SPLIT_ROWS
    if (T > DEC_SPLIT_ROWS)
    {
        return 1;
    }
#endif
#ifdef DEC_INVARIANT_ROWS
    if (T <= DEC_INVARIANT_ROWS)
    {
        T = 1;
    }
#endif
    return min(max((nd_u32)(DEC_TARGET_WG) / max(T * (nd_u32)ND_GROUPS, 1u), 1u), (nd_u32)(DEC_MAX_SPLITS));
}

__device__ __forceinline__ nd_u32 nd_split_keys(nd_u32 span, nd_u32 T)
{
    nd_u32 ms = nd_max_splits(T);
#if DEC_MIN_KEYS > DEC_BC
    nd_u32 ns = min(max(span / (nd_u32)(DEC_MIN_KEYS), 1u), ms);
#ifdef DEC_INVARIANT_ROWS
    nd_u32 rows = T <= DEC_INVARIANT_ROWS ? 1u : T;
#else
    nd_u32 rows = T;
#endif
    if (rows * ND_GROUPS * ns < (nd_u32)(DEC_TARGET_WG) / 8)
    {
        ns = min(max((span + DEC_MIN_KEYS - 1) / (nd_u32)(DEC_MIN_KEYS), 1u), ms);
    }
    nd_u32 per = (span + ns - 1) / ns;
    return (per + DEC_BC - 1) / DEC_BC * DEC_BC;
#else
    nd_u32 per = (span + ms - 1) / ms;
    per = (per + DEC_BC - 1) / DEC_BC * DEC_BC;
    return max(per, (nd_u32)(DEC_MIN_KEYS));
#endif
}

extern "C" __global__ void __launch_bounds__(ND_THREADS) attn_decode_native(NdParams p)
{
    __shared__ float sM[ND_WARPS * HPW];
    __shared__ float sL[ND_WARPS * HPW];
    __shared__ __align__(16) float sO[ND_WARPS * HPW * D];

    const nd_u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const nd_u32 g = lane >> 2, c4 = lane & 3;
    const nd_u32 hk = blockIdx.x / (ND_G / HPW);
    const nd_u32 h0 = hk * ND_G + blockIdx.x % (ND_G / HPW) * HPW;
    const nd_u32 t = blockIdx.y;
#if MULTI_SEQ && MIX_RUN > 0
    if (p.seq[t * 4 + 2] >= MIX_RUN)
    {
        return;
    }
#endif

    // Key range of the row (decode_range) and of this block's split.
    nd_u32 L, jLo;
    nd_u64 kvBase = 0;
#if MULTI_SEQ
    L = (nd_u32)p.seq[t * 4 + 1] + 1;
#if PAGED
    kvBase = (nd_u32)p.seq[t * 4];
#else
    kvBase = (nd_u64)(nd_u32)p.seq[t * 4] * SLOT_CTX * HKV;
#endif
#else
    L = p.rt[RT_N_PAST] + t + 1;
#endif
#if WINDOW
    jLo = L > WINDOW ? L - WINDOW : 0;
#else
    jLo = 0;
#endif
#if CHUNK
    jLo = max(jLo, (L - 1) / CHUNK * CHUNK);
#endif
    const nd_u32 span = L - jLo;
    const nd_u32 per = nd_split_keys(span, p.T);
    const nd_u32 ns = (span + per - 1) / per;
    const nd_u32 split = blockIdx.z;
    if (split >= ns)
    {
        return;
    }

    const nd_u32 jBeg = jLo + split * per;
    const nd_u32 jEnd = min(L, jBeg + per);

    // Query fragments (rows: the block's heads; K step kk: the dimensions of words 2kk and 2kk + 1 of lane c's chunks), Q · SCALE in f16.
    nd_u32 qa[ND_KS][2];
    {
        const bool live = g < HPW;
        const float* qr = p.q + (nd_u64)t * QS0 + (nd_u64)(h0 + (live ? g : 0)) * QS1;
#pragma unroll
        for (int kk = 0; kk < ND_KS; kk++)
        {
#pragma unroll
            for (int w = 0; w < 2; w++)
            {
                nd_u32 idx = 2 * kk + w;
                nd_u32 dim = ((idx / 4) * 4 + c4) * 8 + 2 * (idx % 4);
                float2 v = live ? *(const float2*)(qr + dim) : make_float2(0.0f, 0.0f);
                qa[kk][w] = nd_pack(v.x * ND_SCALE, v.y * ND_SCALE);
            }
        }
    }

    float o[ND_NW][4];
#pragma unroll
    for (int f = 0; f < ND_NW; f++)
    {
#pragma unroll
        for (int e = 0; e < 4; e++)
        {
            o[f][e] = 0.0f;
        }
    }
    float m = ND_NEG, l = 0.0f;

    for (nd_u32 j0 = jBeg + warp * ND_TK; j0 < jEnd; j0 += ND_WARPS * ND_TK)
    {
#if ND_PF
        // The warp's tile ND_PF tiles ahead into L2: the key and value rows' 128-byte lines, one per lane of a row's four (D = 64: two
        // lines per row, lanes 0 and 1; D = 128: four). The loads below then wait on L2 instead of DRAM (RTX PRO 4500, gpt-oss-20b,
        // 4 rows of 130,048 keys: 1.44 → 1.31 ms per layer, 739 → 815 GB/s; benchmarks.md section 78).
        if (c4 < 2 * ND_LPR)
        {
            const nd_u32 jn = j0 + ND_WARPS * ND_TK * ND_PF;
            const unsigned short* cache = c4 < ND_LPR ? p.kc : p.vc;
            const nd_u64 rowElems = c4 < ND_LPR ? (nd_u64)D : (nd_u64)VRS;
            const nd_u32 line = c4 % ND_LPR;
#pragma unroll
            for (int r = 0; r < ND_R; r++)
            {
                nd_u32 j = jn + 8 * r + g;
                if (j < jEnd)
                {
                    const unsigned short* a = cache + (nd_kv_row(p, kvBase, j) + hk) * rowElems + line * 64;
                    asm volatile("prefetch.global.L2 [%0];\n" :: "l"(a));
                }
            }
        }
#endif
        // Key and value rows j0 + 8r + g (rows past the range are zero and masked).
        uint4 kw[ND_R][ND_NCH], vw[ND_R][ND_NCH];
#pragma unroll
        for (int r = 0; r < ND_R; r++)
        {
            nd_u32 j = j0 + 8 * r + g;
            bool in = j < jEnd;
            nd_u64 row = nd_kv_row(p, kvBase, in ? j : jBeg) + hk;
            const uint4* kr = (const uint4*)(p.kc + row * D);
            const uint4* vr = (const uint4*)(p.vc + row * VRS);
#pragma unroll
            for (int q = 0; q < ND_NCH; q++)
            {
                kw[r][q] = in ? kr[q * 4 + c4] : make_uint4(0, 0, 0, 0);
                vw[r][q] = in ? vr[q * 4 + c4] : make_uint4(0, 0, 0, 0);
            }
        }

        // S = Q · Kᵀ: n fragment r holds keys j0 + 8r + 2c, + 1 of head g in s[r][0], s[r][1].
        float s[ND_R][4];
#pragma unroll
        for (int r = 0; r < ND_R; r++)
        {
#pragma unroll
            for (int e = 0; e < 4; e++)
            {
                s[r][e] = 0.0f;
            }
#pragma unroll
            for (int kk = 0; kk < ND_KS; kk++)
            {
                nd_u32 i0 = 2 * kk, i1 = 2 * kk + 1;
                nd_mma(s[r], qa[kk][0], qa[kk][1], nd_word(kw[r][i0 / 4], i0 % 4), nd_word(kw[r][i1 / 4], i1 % 4));
            }
        }

        // Online softmax over the tile (row g's statistics are shared by its four lanes).
        float mx = ND_NEG;
#pragma unroll
        for (int r = 0; r < ND_R; r++)
        {
#pragma unroll
            for (int e = 0; e < 2; e++)
            {
                float v = s[r][e];
#if HAS_SOFTCAP
                v = ND_SOFTCAP * tanhf(v / ND_SOFTCAP);
#endif
                v = j0 + 8 * r + 2 * c4 + e < jEnd ? v : ND_NEG;
                s[r][e] = v;
                mx = fmaxf(mx, v);
            }
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
        float mNew = fmaxf(m, mx);
        float corr = __expf(m - mNew);
        m = mNew;
        float ps = 0.0f;
#pragma unroll
        for (int r = 0; r < ND_R; r++)
        {
#pragma unroll
            for (int e = 0; e < 2; e++)
            {
                float pe = s[r][e] > -1.0e29f ? __expf(s[r][e] - mNew) : 0.0f;
                s[r][e] = pe;
                ps += pe;
            }
        }
        l = l * corr + ps;
#pragma unroll
        for (int f = 0; f < ND_NW; f++)
        {
            o[f][0] *= corr;
            o[f][1] *= corr;
        }

        // O += P · V over the tile's 16-key steps: P from the score accumulators, V transposed in registers.
#pragma unroll
        for (int ks = 0; ks < ND_R / 2; ks++)
        {
            nd_u32 a0 = nd_pack(s[2 * ks][0], s[2 * ks][1]);
            nd_u32 a2 = nd_pack(s[2 * ks + 1][0], s[2 * ks + 1][1]);
#pragma unroll
            for (int f = 0; f < ND_NW; f++)
            {
                nd_u32 b0 = nd_trans(nd_word(vw[2 * ks][f / 4], f % 4));
                nd_u32 b1 = nd_trans(nd_word(vw[2 * ks + 1][f / 4], f % 4));
                nd_mma(o[f], a0, a2, b0, b1);
            }
        }
    }

    // The warp's state into shared memory: row sums over the four lanes of a row; output word f of lane c is dimensions
    // (4 (f / 4) + c) * 8 + 2 (f % 4), + 1.
    l += __shfl_xor_sync(0xFFFFFFFFu, l, 1);
    l += __shfl_xor_sync(0xFFFFFFFFu, l, 2);
    if (g < HPW)
    {
        if (c4 == 0)
        {
            sM[warp * HPW + g] = m;
            sL[warp * HPW + g] = l;
        }
        float* od = sO + (warp * HPW + g) * D;
#pragma unroll
        for (int f = 0; f < ND_NW; f++)
        {
            nd_u32 dim = ((f / 4) * 4 + c4) * 8 + 2 * (f % 4);
            *(float2*)(od + dim) = make_float2(o[f][0], o[f][1]);
        }
    }
    __syncthreads();

    // Output element i = (head, dimension): the warps' states merged, then normalized (direct) or written as the range's partial.
    const bool direct = ns == 1;
    for (nd_u32 i = tid; i < HPW * D; i += ND_THREADS)
    {
        nd_u32 hg = i / D, d = i % D, h = h0 + hg;
        float M = ND_NEG;
#pragma unroll
        for (int w = 0; w < ND_WARPS; w++)
        {
            M = fmaxf(M, sM[w * HPW + hg]);
        }
        float lsum = 0.0f, num = 0.0f;
#pragma unroll
        for (int w = 0; w < ND_WARPS; w++)
        {
            float wt = __expf(sM[w * HPW + hg] - M);
            lsum += wt * sL[w * HPW + hg];
            num += wt * sO[(w * HPW + hg) * D + d];
        }
        if (direct)
        {
#if HAS_SINKS
            float sk = p.sinks[h], mf = fmaxf(M, sk), cf = __expf(M - mf);
            lsum = lsum * cf + __expf(sk - mf);
            num *= cf;
#endif
            p.o[((nd_u64)t * HQ + h) * D + d] = lsum > 0.0f ? num / lsum : 0.0f;
        }
        else
        {
            float* dst = p.part + (((nd_u64)split * p.T + t) * HQ + h) * ND_PS;
            dst[d] = num;
            if (d == 0)
            {
                dst[D] = M;
                dst[D + 1] = lsum;
            }
        }
    }
}
