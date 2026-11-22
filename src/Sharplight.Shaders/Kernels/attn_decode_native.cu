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
#ifndef ND_PF_MIN_TILES
#define ND_PF_MIN_TILES 4             // tiles per warp from which a key range prefetches
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

#if (D != 64 && D != 128 && D != 256 && D != 512) || HPW > 8 || (ND_G % HPW) != 0 || (HQ % HKV) != 0
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

// Key range [jBeg, jEnd) of this block (decode_range and the split of decode_split_keys); false when the block's split is past the
// row's range count.
__device__ __forceinline__ bool nd_range(const NdParams& p, nd_u32 t, nd_u64& kvBase, nd_u32& jBeg, nd_u32& jEnd, nd_u32& ns)
{
    nd_u32 L, jLo;
    kvBase = 0;
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
    ns = (span + per - 1) / per;
    if (blockIdx.z >= ns)
    {
        return false;
    }

    jBeg = jLo + blockIdx.z * per;
    jEnd = min(L, jBeg + per);
    return true;
}

#if D <= 128
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
    nd_u64 kvBase;
    nd_u32 jBeg, jEnd, ns;
    if (!nd_range(p, t, kvBase, jBeg, jEnd, ns))
    {
        return;
    }
    const nd_u32 split = blockIdx.z;

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

#if ND_PF
    const bool prefetch = jEnd - jBeg >= (nd_u32)(ND_PF_MIN_TILES * ND_WARPS * ND_TK);
#endif
    for (nd_u32 j0 = jBeg + warp * ND_TK; j0 < jEnd; j0 += ND_WARPS * ND_TK)
    {
#if ND_PF
        // The warp's tile ND_PF tiles ahead into L2: the key and value rows' 128-byte lines, one per lane of a row's four (D = 64: two
        // lines per row, lanes 0 and 1; D = 128: four). The loads below then wait on L2 instead of DRAM (RTX PRO 4500, gpt-oss-20b,
        // 4 rows of 130,048 keys: 1.44 → 1.31 ms per layer, 739 → 815 GB/s; benchmarks.md section 78). Ranges of fewer than
        // ND_PF_MIN_TILES tiles per warp skip it: their few tiles gain little, and the prefetch's address work (the block table of a
        // paged cache) costs ~2 µs per launch (1 x 32,768 keys: 6.16 → 6.21 ms per gpt-oss-20b step without the limit).
        if (prefetch && c4 < 2 * ND_LPR)
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
#else
// ---------------------------------------------------------------------------------------------
// Head sizes 256 and 512 (Gemma 4's window and global layers, Qwen3.5 / 3.8): a warp's full-width key / value rows and output
// accumulators do not fit its registers (D = 512: 256 accumulator registers alone), so the block's ND_WARPS warps share each tile:
//
//   * the tile's NWD_TK keys (NWD_TK x D values: 32 KiB of keys and values per stage) are copied with cp.async into a ring of NWD_ST
//     stages of dynamic shared memory (rows of 16-byte chunks, XOR swizzle within each 128-byte group, so the copies and ldmatrix are
//     free of bank conflicts; rows past the range zero-filled), so a block keeps two tiles in flight while it computes the third;
//   * S = Q · Kᵀ: warp w computes key fragment w % NWD_NF (8 keys) over dimensions [w / NWD_NF * NWD_DQ, + NWD_DQ) with its query
//     fragments in registers (rows: the block's heads; rows HPW .. 15 zero) and K from ldmatrix, and leaves its partial scores in
//     shared memory; after a barrier every warp sums the partials of every key fragment in the same order, so all warps run the same
//     online softmax and hold the same (m, l);
//   * O += P · V: warp w owns output dimensions [w * NWD_DVW, + NWD_DVW) (V from ldmatrix.trans), so the warps' states need no merge.
//
// ND_KVQ8 (wide form only): q8_0 caches (32-value blocks of an f16 scale and 32 int8 quants, 34 bytes; a row of D values is 16-byte
// aligned at head sizes 256 / 512): the ring holds the tiles' raw rows (17 / 34 chunks of 16 bytes per row), and after the barrier
// that makes tile i visible the block converts it into one f16 tile in the swizzled layout above (each value d * q rounded to f16, as
// fa_mqa stages it), behind a third barrier, so the products are those of the f16 form. Half the bytes of an f16 cache per key.
//
// Two barriers per tile (three with ND_KVQ8). The result is written directly (one key range, sinks in the denominator) or as the range's partial
// [split][t][h][D + 4] (acc, m, l) for decode_combine, as the narrow form does; Q · SCALE and P are rounded to f16.
#ifndef NWD_ST
#define NWD_ST 3                                // ring stages
#endif
#ifndef ND_KVQ8
#define ND_KVQ8 0
#endif
#define NWD_TK (8192 / D)                       // keys per tile (32 at D = 256, 16 at 512): 32 KiB of keys and values
#define NWD_NF (NWD_TK / 8)                     // key fragments of S per tile
#define NWD_DS (ND_WARPS / NWD_NF)              // warps per key fragment (each a slice of the dimensions)
#define NWD_DQ (D / NWD_DS)                     // dimensions of a warp's partial scores
#define NWD_KSQ (NWD_DQ / 16)                   // K steps of a warp's partial scores
#define NWD_DVW (D / ND_WARPS)                  // output dimensions per warp
#define NWD_NO (NWD_DVW / 8)                    // output fragments per warp
#define NWD_KSV (NWD_TK / 16)                   // K steps of P · V per tile
#define NWD_CPR (D / 8)                         // 16-byte chunks per row
#define NWD_CPT (NWD_TK * NWD_CPR / ND_THREADS) // chunks per thread, keys or values, per tile
#define NWD_RPT (ND_THREADS / NWD_CPR)          // rows the block's threads cover per chunk round
#if NWD_DS * NWD_NF != ND_WARPS || NWD_KSQ < 2 || (NWD_KSQ % 2) != 0 || (NWD_NO % 2) != 0 || NWD_TK > 32 || (NWD_TK * NWD_CPR) % ND_THREADS != 0
#error "attn_decode_native: unsupported wide shape"
#endif
#define NWD_RB (D / 32 * 34)                    // ND_KVQ8: bytes of one cache row (16-byte multiple at D = 256 / 512)
#define NWD_RCH (NWD_RB / 16)                   // ND_KVQ8: 16-byte chunks of one cache row
#if ND_KVQ8
#define NWD_STAGE_BYTES (2 * NWD_TK * NWD_RB)   // raw keys and values of one tile
#define NWD_TILE_BYTES (2 * NWD_TK * D * 2)     // the converted f16 tile after the ring
#else
#define NWD_STAGE_BYTES (2 * NWD_TK * D * 2)
#define NWD_TILE_BYTES 0
#endif

__device__ __forceinline__ nd_u32 nd_smem(const void* ptr)
{
    return (nd_u32)__cvta_generic_to_shared(ptr);
}

// 16 bytes from global to shared; src_bytes 0 zero-fills.
__device__ __forceinline__ void nd_copy(nd_u32 dst, const void* src, nd_u32 srcBytes)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(srcBytes));
}

__device__ __forceinline__ void nd_ldsm4(nd_u32 addr, nd_u32& r0, nd_u32& r1, nd_u32& r2, nd_u32& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void nd_ldsm4t(nd_u32 addr, nd_u32& r0, nd_u32& r1, nd_u32& r2, nd_u32& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// Byte offset of chunk `ch` of row `r` in a tile of rows of NWD_CPR chunks (the chunk's low three bits XOR the row's).
__device__ __forceinline__ nd_u32 nd_swz(nd_u32 r, nd_u32 ch)
{
    return (r * NWD_CPR + (ch ^ (r & 7))) * 16;
}

// Copies the key and value rows of tile [j0, j0 + NWD_TK) into stage buffers sk / sv (shared addresses). Lane l of each warp looks
// up the cache row of key j0 + l (paged caches: one block-table read per lane), the copies take it by shuffle.
__device__ __forceinline__ void nd_load_tile(const NdParams& p, nd_u64 kvBase, nd_u32 hk, nd_u32 j0, nd_u32 jEnd, nd_u32 jSafe, nd_u32 sk, nd_u32 sv,
                                             nd_u32 tid)
{
    const nd_u32 lane = tid & 31;
    const nd_u32 jl = j0 + (lane % NWD_TK);
    const nd_u64 myRow = nd_kv_row(p, kvBase, jl < jEnd ? jl : jSafe) + hk;
#pragma unroll
    for (int i = 0; i < NWD_CPT; i++)
    {
        const nd_u32 e = tid + i * ND_THREADS;
        const nd_u32 r = e / NWD_CPR, ch = e % NWD_CPR;
        const nd_u64 row = __shfl_sync(0xFFFFFFFFu, myRow, r);
        const nd_u32 bytes = j0 + r < jEnd ? 16u : 0u;
        nd_copy(sk + nd_swz(r, ch), p.kc + row * D + ch * 8, bytes);
        nd_copy(sv + nd_swz(r, ch), p.vc + row * VRS + ch * 8, bytes);
    }
}

#if ND_KVQ8
// ND_KVQ8: copies the raw q8_0 rows of tile [j0, j0 + NWD_TK) into stage buffers sk / sv (rows of NWD_RB bytes, rows past the range
// zero-filled), the cache rows looked up as in nd_load_tile.
__device__ __forceinline__ void nd_load_tile_q8(const NdParams& p, nd_u64 kvBase, nd_u32 hk, nd_u32 j0, nd_u32 jEnd, nd_u32 jSafe, nd_u32 sk, nd_u32 sv,
                                                nd_u32 tid)
{
    const nd_u32 lane = tid & 31;
    const nd_u32 jl = j0 + (lane % NWD_TK);
    const nd_u64 myRow = nd_kv_row(p, kvBase, jl < jEnd ? jl : jSafe) + hk;
    const unsigned char* kc = (const unsigned char*)p.kc;
    const unsigned char* vc = (const unsigned char*)p.vc;
#pragma unroll
    for (int i = 0; i < (NWD_TK * NWD_RCH + ND_THREADS - 1) / ND_THREADS; i++)
    {
        const nd_u32 e = tid + i * ND_THREADS;
        const nd_u32 r = min(e / NWD_RCH, (nd_u32)NWD_TK - 1), ch = e % NWD_RCH;
        const nd_u64 row = __shfl_sync(0xFFFFFFFFu, myRow, r);
        if (e < NWD_TK * NWD_RCH)
        {
            const nd_u32 bytes = j0 + r < jEnd ? 16u : 0u;
            nd_copy(sk + r * NWD_RB + ch * 16, kc + row * NWD_RB + ch * 16, bytes);
            nd_copy(sv + r * NWD_RB + ch * 16, vc + row * (VRS / 32 * 34) + ch * 16, bytes);
        }
    }
}

// ND_KVQ8: converts the raw rows of a stage (keys, then values) into the f16 tile `tile` (keys at 0, values at NWD_TK * D * 2), 8 values
// per 16-byte chunk in the swizzled layout of nd_swz: d * q rounded to f16.
__device__ __forceinline__ void nd_convert_q8(const unsigned char* raw, unsigned char* tile, nd_u32 tid)
{
#pragma unroll 4
    for (nd_u32 e = tid; e < 2 * NWD_TK * NWD_CPR; e += ND_THREADS)
    {
        const nd_u32 kv = e / (NWD_TK * NWD_CPR), rest = e % (NWD_TK * NWD_CPR), r = rest / NWD_CPR, ch = rest % NWD_CPR;
        const unsigned char* blk = raw + kv * NWD_TK * NWD_RB + r * NWD_RB + (ch >> 2) * 34;
        float d;
        asm("cvt.f32.f16 %0, %1;\n" : "=f"(d) : "h"(*(const unsigned short*)blk));
        const unsigned short* qw = (const unsigned short*)(blk + 2 + (ch & 3) * 8);
        nd_u32 w[4];
#pragma unroll
        for (int k = 0; k < 4; k++)
        {
            const nd_u32 b = qw[k];
            w[k] = nd_pack(d * (float)(signed char)(b & 0xFF), d * (float)(signed char)(b >> 8));
        }
        *(uint4*)(tile + kv * NWD_TK * D * 2 + nd_swz(r, ch)) = make_uint4(w[0], w[1], w[2], w[3]);
    }
}
#endif

extern "C" __global__ void __launch_bounds__(ND_THREADS) attn_decode_native(NdParams p)
{
    extern __shared__ __align__(128) unsigned char nd_sm[];
    // Stages of [keys | values] (ND_KVQ8: raw rows, then the converted f16 tile), then the partial scores [warp][lane] (float2).
    float2* sS = (float2*)(nd_sm + NWD_ST * NWD_STAGE_BYTES + NWD_TILE_BYTES);

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
    nd_u64 kvBase;
    nd_u32 jBeg, jEnd, ns;
    if (!nd_range(p, t, kvBase, jBeg, jEnd, ns))
    {
        return;
    }

    const nd_u32 nt = (jEnd - jBeg + NWD_TK - 1) / NWD_TK;
    const nd_u32 base = nd_smem(nd_sm);
    const nd_u32 stageBytes = NWD_STAGE_BYTES;
#if ND_KVQ8
#define ND_LOAD_TILE(j0, st) nd_load_tile_q8(p, kvBase, hk, j0, jEnd, jBeg, st, st + NWD_TK * NWD_RB, tid)
#else
#define ND_LOAD_TILE(j0, st) nd_load_tile(p, kvBase, hk, j0, jEnd, jBeg, st, st + NWD_TK * D * 2, tid)
#endif
#pragma unroll
    for (int s = 0; s < NWD_ST - 1; s++)
    {
        if (s < (int)nt)
        {
            ND_LOAD_TILE(jBeg + s * NWD_TK, base + s * stageBytes);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    // Query fragments of the warp's dimension slice (K step kk: dimensions q0 + 16 kk + 2c, + 1 and + 8), Q · SCALE in f16.
    const nd_u32 nf = warp % NWD_NF, q0 = warp / NWD_NF * NWD_DQ;
    nd_u32 qa[NWD_KSQ][2];
    {
        const bool live = g < HPW;
        const float* qr = p.q + (nd_u64)t * QS0 + (nd_u64)(h0 + (live ? g : 0)) * QS1 + q0 + 2 * c4;
#pragma unroll
        for (int kk = 0; kk < NWD_KSQ; kk++)
        {
            float2 v0 = live ? *(const float2*)(qr + 16 * kk) : make_float2(0.0f, 0.0f);
            float2 v1 = live ? *(const float2*)(qr + 16 * kk + 8) : make_float2(0.0f, 0.0f);
            qa[kk][0] = nd_pack(v0.x * ND_SCALE, v0.y * ND_SCALE);
            qa[kk][1] = nd_pack(v1.x * ND_SCALE, v1.y * ND_SCALE);
        }
    }

    float o[NWD_NO][4];
#pragma unroll
    for (int f = 0; f < NWD_NO; f++)
    {
#pragma unroll
        for (int e = 0; e < 4; e++)
        {
            o[f][e] = 0.0f;
        }
    }
    float m = ND_NEG, l = 0.0f;

    for (nd_u32 i = 0; i < nt; i++)
    {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NWD_ST - 2));
        __syncthreads();   // tile i visible; every warp is done with tile i - 1 (its stage and the partial scores)
        {
            const nd_u32 nx = i + NWD_ST - 1;
            if (nx < nt)
            {
                const nd_u32 st = base + (nx % NWD_ST) * stageBytes;
                ND_LOAD_TILE(jBeg + nx * NWD_TK, st);
            }
            asm volatile("cp.async.commit_group;\n" ::);
        }
#if ND_KVQ8
        nd_convert_q8(nd_sm + (i % NWD_ST) * stageBytes, nd_sm + NWD_ST * stageBytes, tid);
        __syncthreads();   // the converted tile is visible
        const nd_u32 sk = base + NWD_ST * stageBytes, sv = sk + NWD_TK * D * 2;
#else
        const nd_u32 sk = base + (i % NWD_ST) * stageBytes, sv = sk + NWD_TK * D * 2;
#endif
        const nd_u32 j0 = jBeg + i * NWD_TK;

        // Partial S of key fragment nf over the warp's dimensions: ldmatrix.x4 matrices m = lane / 8: keys nf * 8 + lane % 8,
        // chunk q0 / 8 + 4 kp + m (K steps 2 kp and 2 kp + 1).
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int kp = 0; kp < NWD_KSQ / 2; kp++)
        {
            nd_u32 b0, b1, b2, b3;
            nd_ldsm4(sk + nd_swz(nf * 8 + (lane & 7), q0 / 8 + 4 * kp + (lane >> 3)), b0, b1, b2, b3);
            nd_mma(s, qa[2 * kp][0], qa[2 * kp][1], b0, b1);
            nd_mma(s, qa[2 * kp + 1][0], qa[2 * kp + 1][1], b2, b3);
        }
        sS[warp * 32 + lane] = make_float2(s[0], s[1]);
        __syncthreads();

        // Scores of row g at keys nf' * 8 + 2c, + 1 for every key fragment nf' (the slices summed in order), softcap, mask.
        float sc[NWD_NF][2];
        float mx = ND_NEG;
#pragma unroll
        for (int f = 0; f < NWD_NF; f++)
        {
            float2 v = sS[f * 32 + lane];
#pragma unroll
            for (int d = 1; d < NWD_DS; d++)
            {
                float2 w = sS[(d * NWD_NF + f) * 32 + lane];
                v.x += w.x;
                v.y += w.y;
            }
#pragma unroll
            for (int e = 0; e < 2; e++)
            {
                float x = e == 0 ? v.x : v.y;
#if HAS_SOFTCAP
                x = ND_SOFTCAP * tanhf(x / ND_SOFTCAP);
#endif
                x = j0 + f * 8 + 2 * c4 + e < jEnd ? x : ND_NEG;
                sc[f][e] = x;
                mx = fmaxf(mx, x);
            }
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
        const float mNew = fmaxf(m, mx);
        const float corr = __expf(m - mNew);
        m = mNew;
        float ps = 0.0f;
#pragma unroll
        for (int f = 0; f < NWD_NF; f++)
        {
#pragma unroll
            for (int e = 0; e < 2; e++)
            {
                float pe = sc[f][e] > -1.0e29f ? __expf(sc[f][e] - mNew) : 0.0f;
                sc[f][e] = pe;
                ps += pe;
            }
        }
        l = l * corr + ps;
#pragma unroll
        for (int f = 0; f < NWD_NO; f++)
        {
            o[f][0] *= corr;
            o[f][1] *= corr;
        }

        // O += P · V over the tile's 16-key steps: ldmatrix.x4.trans matrices m = lane / 8: keys 16 ks + (m % 2) * 8 + lane % 8,
        // chunk (warp * NWD_DVW + 16 fp) / 8 + m / 2 (output fragments 2 fp and 2 fp + 1).
#pragma unroll
        for (int ks = 0; ks < NWD_KSV; ks++)
        {
            const nd_u32 a0 = nd_pack(sc[2 * ks][0], sc[2 * ks][1]);
            const nd_u32 a2 = nd_pack(sc[2 * ks + 1][0], sc[2 * ks + 1][1]);
#pragma unroll
            for (int fp = 0; fp < NWD_NO / 2; fp++)
            {
                nd_u32 b0, b1, b2, b3;
                nd_ldsm4t(sv + nd_swz(16 * ks + (lane & 8) + (lane & 7), (warp * NWD_DVW + 16 * fp) / 8 + (lane >> 4)), b0, b1, b2, b3);
                nd_mma(o[2 * fp], a0, a2, b0, b1);
                nd_mma(o[2 * fp + 1], a0, a2, b2, b3);
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    // Row sums over the row's four lanes; every warp holds the same (m, l) and its own output dimensions.
    l += __shfl_xor_sync(0xFFFFFFFFu, l, 1);
    l += __shfl_xor_sync(0xFFFFFFFFu, l, 2);
    if (g < HPW)
    {
        const nd_u32 h = h0 + g;
        if (ns == 1)
        {
            float lsum = l, cf = 1.0f;
#if HAS_SINKS
            const float sk = p.sinks[h], mf = fmaxf(m, sk);
            cf = __expf(m - mf);
            lsum = lsum * cf + __expf(sk - mf);
#endif
            const float inv = lsum > 0.0f ? cf / lsum : 0.0f;
            float* dst = p.o + ((nd_u64)t * HQ + h) * D + warp * NWD_DVW + 2 * c4;
#pragma unroll
            for (int f = 0; f < NWD_NO; f++)
            {
                *(float2*)(dst + 8 * f) = make_float2(o[f][0] * inv, o[f][1] * inv);
            }
        }
        else
        {
            float* dst = p.part + (((nd_u64)blockIdx.z * p.T + t) * HQ + h) * ND_PS;
#pragma unroll
            for (int f = 0; f < NWD_NO; f++)
            {
                *(float2*)(dst + warp * NWD_DVW + 8 * f + 2 * c4) = make_float2(o[f][0], o[f][1]);
            }
            if (warp == 0 && c4 == 0)
            {
                dst[D] = m;
                dst[D + 1] = l;
            }
        }
    }
}
#endif
