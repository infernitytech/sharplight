#pragma once
// Skinny tensor-core GEMM for few activation rows (decode batches, speculative verification, 2-64 rows), included by
// matmul.slang (uses its Params, load16h and the packed-half decoders of the gemm_cm2 section).
//
// Workgroup = SK_SG subgroups that all own the same SK_FN x 16 weight rows and split K between them: subgroup s walks the
// K blocks s, s + SK_SG, ... and the partial accumulators are summed through shared memory at the end. So every weight is
// read from memory exactly once, by one workgroup, with no split-K scratch buffer and no reduction launch, and the grid
// has N / (16 SK_FN) workgroups however few rows there are. Token rows past T count as zeros and their results are dropped.
//
// Two ways to feed the MMAs:
//  * Staged (any device): each subgroup stages its own A (SK_FM x 16 token rows) and B slices of SK_BK values into a private
//    part of shared memory and loads the fragments from there, so the K loop has subgroup barriers only.
//  * Direct (SK_DIRECT, one token fragment, f16 x; the lowering uses it for block FP8 and Q8_0 weights): the operands go
//    from memory straight into the fragment registers, with no shared memory and no barriers in the K loop. A lane holds the K slots 2c, 2c + 1, 2c + 8, 2c + 9 (c = lane % 4) of
//    the A rows / B columns g and g + 8 (g = lane / 4), the element layout of the 16x8x16 MMA instructions behind the
//    16x16x16 fragments on NVIDIA GPUs. Since a dot product may visit K in any order, as long as A and B agree, each lane's
//    slots are mapped to 16 contiguous values of its rows instead (sk_direct), so a lane loads x and the weights with 16-byte
//    words and the four lanes of a row read 64 contiguous values. FP8 weights are prefetched as packed codes and decoded to
//    f16 right before their MMAs. The lowering picks this path only on devices with that layout (DeviceFeatures.
//    MatrixMmaLayout); the kernel verifies it once (sk_direct_layout) and stores NaN where it differs, so a wrong detection
//    fails loudly rather than computing garbage. (Both paths in one kernel cost the direct one 60 registers.)
//
// x: f16 rows (X_F16 = 1, row stride X_STRIDE halves) or f32 rows converted while staging (X_F16 = 0). DUAL: w and w2
// (same shape and format) give SK_FN fragments each, combined by epi2 per element before the store.
// Requires K % SK_BK == 0, and K % 64 == 0 with SK_DIRECT.
#ifdef KERNEL_GEMM_SK
using namespace linalg;
#ifndef SK_FM
#define SK_FM 1
#endif
#ifndef SK_FN
#define SK_FN 2
#endif
#ifndef SK_SG
#define SK_SG 4
#endif
#ifndef SK_BK
#define SK_BK 64
#endif
#ifndef X_F16
#define X_F16 0
#endif
#ifndef SK_DIRECT
#define SK_DIRECT 0
#endif
#if SK_DIRECT && (SK_FM != 1 || !X_F16)
#error "the direct skinny GEMM needs one token fragment and f16 activations"
#endif
#define SK_NB (SK_FN * (1 + DUAL))          // B fragments (16 weight rows each) per subgroup
#define SK_AS (SK_BK + 8)                   // staged row stride in halves (padding against bank conflicts)
#define SK_A_PER_LANE (SK_FM * 16 * SK_BK / 8 / 32)
#define SK_B_PER_LANE (SK_NB * 16 * SK_BK / 16 / 32)
#define SK_A_WORDS (SK_FM * 16 * SK_AS / 8)  // uint4 per subgroup
#define SK_B_WORDS (SK_NB * 16 * SK_AS / 8)

typealias SkA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias SkB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias SkC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

#if SK_DIRECT
// The subgroups' partial tiles of the final reduction (64 words per 16x16 f32 tile), and the layout probe (a 16x16 f16 matrix).
groupshared uint4 skB[SK_SG * (1 + DUAL) * 64];
groupshared uint skP[128];
#else
// Staged slices; the B slices also hold the partial tiles of the final reduction (SK_B_WORDS >= 64 (1 + DUAL)).
groupshared uint4 skA[SK_SG * SK_A_WORDS];
groupshared uint4 skB[SK_SG * SK_B_WORDS];
#endif

// The 8 activations [k, k + 8) of token row t as packed halves (zeros past T).
uint4 sk_load_x(float* x, uint t, uint k, uint T)
{
    if (t >= T)
    {
        return uint4(0, 0, 0, 0);
    }
#if X_F16
    return ((uint4*)((uint16_t*)x + uint64_t(t) * X_STRIDE + k))[0];
#else
    float4* xp = (float4*)(x + uint64_t(t) * X_STRIDE + k);
    float4 a = xp[0], b = xp[1];
    return uint4(f32tof16(a.x) | (f32tof16(a.y) << 16), f32tof16(a.z) | (f32tof16(a.w) << 16),
                 f32tof16(b.x) | (f32tof16(b.y) << 16), f32tof16(b.z) | (f32tof16(b.w) << 16));
#endif
}

#if !SK_DIRECT
// Staged path: accumulates the workgroup's share of K into acc.
void sk_staged(uint n0, uint sg, uint lane, Params p, inout SkC acc[SK_FM][SK_NB])
{
    uint aBase = sg * SK_A_WORDS, bBase = sg * SK_B_WORDS;
    uint4 xs[SK_A_PER_LANE];
    uint4 ws[SK_B_PER_LANE][2];
    #define SK_FETCH(k0)                                                                                \
    [ForceUnroll] for (uint it = 0; it < SK_A_PER_LANE; it++)                                           \
    {                                                                                                   \
        uint idx = lane + it * 32, r = idx / (SK_BK / 8), q = idx % (SK_BK / 8);                       \
        xs[it] = sk_load_x(p.x, r, (k0) + q * 8, p.T);                                                  \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < SK_B_PER_LANE; it++)                                           \
    {                                                                                                   \
        uint idx = lane + it * 32, r = idx / (SK_BK / 16), q = idx % (SK_BK / 16);                      \
        uint wr = n0 + r % (SK_FN * 16);                                                                \
        ws[it][0] = 0; ws[it][1] = 0;                                                                   \
        if (N % (SK_FN * 16) == 0 || wr < N)                                                            \
        {                                                                                               \
            if (!DUAL || r < SK_FN * 16) load16h(p.w, wr, (k0) / 16 + q, ws[it][0], ws[it][1]);        \
            SK_LOAD_W2(wr, (k0) / 16 + q, ws[it])                                                        \
        }                                                                                               \
    }
#if DUAL
    #define SK_LOAD_W2(wr, g16, dst) else { load16h(p.w2, wr, g16, dst[0], dst[1]); }
#else
    #define SK_LOAD_W2(wr, g16, dst)
#endif
    #define SK_STAGE()                                                                                  \
    [ForceUnroll] for (uint it = 0; it < SK_A_PER_LANE; it++)                                           \
    {                                                                                                   \
        uint idx = lane + it * 32, r = idx / (SK_BK / 8), q = idx % (SK_BK / 8);                       \
        skA[aBase + (r * SK_AS + q * 8) / 8] = xs[it];                                                  \
    }                                                                                                   \
    [ForceUnroll] for (uint it = 0; it < SK_B_PER_LANE; it++)                                           \
    {                                                                                                   \
        uint idx = lane + it * 32, r = idx / (SK_BK / 16), q = idx % (SK_BK / 16);                      \
        uint ob = bBase + (r * SK_AS + q * 16) / 8;                                                     \
        skB[ob] = ws[it][0]; skB[ob + 1] = ws[it][1];                                                   \
    }

    const uint blocks = K / SK_BK;
    if (sg < blocks)
    {
        SK_FETCH(sg * SK_BK)
    }
    for (uint kb = sg; kb < blocks; kb += SK_SG)
    {
        // The previous block's MMAs have read the slices; stage this block and prefetch the next one into registers.
        GroupMemoryBarrierWithWaveSync();
        SK_STAGE()
        GroupMemoryBarrierWithWaveSync();
        if (kb + SK_SG < blocks)
        {
            SK_FETCH((kb + SK_SG) * SK_BK)
        }

        [ForceUnroll]
        for (uint kk = 0; kk < SK_BK; kk += 16)
        {
            SkB b[SK_NB];
            [ForceUnroll] for (int j = 0; j < SK_NB; j++) b[j] = SkB.Load<CoopMatMatrixLayout.ColumnMajor>(skB, (bBase * 8 + j * 16 * SK_AS + kk) / 8, SK_AS / 8);
            [ForceUnroll]
            for (int i = 0; i < SK_FM; i++)
            {
                SkA a = SkA.Load<CoopMatMatrixLayout.RowMajor>(skA, (aBase * 8 + i * 16 * SK_AS + kk) / 8, SK_AS / 8);
                [ForceUnroll] for (int j = 0; j < SK_NB; j++) acc[i][j] = coopMatMulAdd<float, false>(a, b[j], acc[i][j]);
            }
        }
    }
    #undef SK_FETCH
    #undef SK_STAGE
    #undef SK_LOAD_W2
}
#else
// Word w (0-7) of the 16 packed halves lo, hi.
uint sk_word(uint4 lo, uint4 hi, uint w)
{
    return w < 4 ? lo[w] : hi[w - 4];
}

// Half h (0: low, 1: high) of a packed word.
half sk_half(uint word, uint h)
{
    half2 v = bit_cast<half2>(word);
    return h == 0 ? v.x : v.y;
}

// The 16 weights [16 g16, 16 g16 + 16) of row `row` as fetched for sk_decode_row: FP8 codes stay packed (one uint4, and the
// group multiplier in r1.x) until their MMAs, other formats are decoded to packed halves at once.
void sk_fetch_row(uint8_t* w, uint row, uint g16, out uint4 r0, out uint4 r1)
{
#ifdef WFMT_FP8
    float s;
    uint* q = (uint*)fp8_group(w, ROW_BYTES, row, g16 * 2, FP8_BLOCK, WFMT == FMT_F8E5M2, s);
#if FP8_SPLIT || FP8_BLOCK == 0
    r0 = *(uint4*)q;   // 16-byte aligned codes
#else
    r0 = uint4(q[0], q[1], q[2], q[3]);
#endif
    r1 = uint4(asuint(s), 0, 0, 0);
#else
    load16h(w, row, g16, r0, r1);
#endif
}

// Packed halves (lo: first 8) of what sk_fetch_row fetched; the same values as load16h.
void sk_decode_row(uint4 r0, uint4 r1, out uint4 lo, out uint4 hi)
{
#ifdef WFMT_FP8
    float s = asfloat(r1.x);
    lo = uint4(fp8x4_h(r0.x, s), fp8x4_h(r0.y, s));
    hi = uint4(fp8x4_h(r0.z, s), fp8x4_h(r0.w, s));
#else
    lo = r0;
    hi = r1;
#endif
}

// True when the device's 16x16 f16 fragments have the element layout the direct path assumes (uniform over the workgroup):
// element e of lane (g, c) is at row (A) / column (B) g + 8 (e >> 2) and K slot 2c + (e & 1) + 8 ((e >> 1) & 1). Loads both
// fragments from a matrix whose element (r, k) / (k, n) holds 16 r + k / 16 n + k.
bool sk_direct_layout(uint tid, uint lane)
{
    for (uint i = tid; i < 128; i += SK_SG * 32)
    {
        skP[i] = f32tof16(float(2 * i)) | (f32tof16(float(2 * i + 1)) << 16);
    }
    GroupMemoryBarrierWithGroupSync();
    SkA pa = SkA.Load<CoopMatMatrixLayout.RowMajor>(skP, 0, 8);
    SkB pb = SkB.Load<CoopMatMatrixLayout.ColumnMajor>(skP, 0, 8);
    uint g = lane / 4, c = lane % 4;
    bool match = true;
    [ForceUnroll]
    for (uint e = 0; e < 8; e++)
    {
        uint expected = (g + 8 * (e >> 2)) * 16 + 2 * c + (e & 1) + 8 * ((e >> 1) & 1);
        match = match && uint(float(pa[e])) == expected && uint(float(pb[e])) == expected;
    }
    // Every subgroup sees the same layout; the barrier keeps the probe intact until all have read it.
    match = WaveActiveAllTrue(match);
    GroupMemoryBarrierWithGroupSync();
    return match;
}

// Direct path: accumulates the workgroup's share of K into acc[0]. Lane (g, c) of subgroup sg walks the 64-value blocks
// sg, sg + SK_SG, ...; of block kb it reads values [64 kb + 16 c, 64 kb + 16 c + 16) of x rows g, g + 8 and of the weight
// rows g, g + 8 of every B fragment. MMA f (0-3) maps the lane's K slots 2c, 2c + 1, 2c + 8, 2c + 9 to its values 4f .. 4f + 3,
// the same for A and B, so the four MMAs cover the block once. The weights of the next block are prefetched into registers;
// x (the same rows for every workgroup, so mostly cache hits) is too.
void sk_direct(uint n0, uint sg, uint lane, Params p, inout SkC acc[SK_FM][SK_NB])
{
    uint g = lane / 4, c = lane % 4;
    bool rowLo = g < p.T, rowHi = g + 8 < p.T;
    uint4 xr[2][2], wr[SK_NB][2][2];
    #define SKD_FETCH(kb)                                                                               \
    {                                                                                                   \
        uint k_ = (kb) * 64 + c * 16;                                                                   \
        [ForceUnroll]                                                                                   \
        for (int h = 0; h < 2; h++)                                                                     \
        {                                                                                               \
            xr[h][0] = 0; xr[h][1] = 0;                                                                 \
            if (h == 0 ? rowLo : rowHi)                                                                 \
            {                                                                                           \
                uint4* px = (uint4*)((uint16_t*)p.x + uint64_t(g + 8 * h) * X_STRIDE + k_);             \
                xr[h][0] = px[0]; xr[h][1] = px[1];                                                     \
            }                                                                                           \
            [ForceUnroll]                                                                               \
            for (int j = 0; j < SK_NB; j++)                                                             \
            {                                                                                           \
                uint row = n0 + (j % SK_FN) * 16 + g + 8 * h;                                           \
                wr[j][h][0] = 0; wr[j][h][1] = 0;                                                       \
                if (N % 16 == 0 || row < N) sk_fetch_row(SKD_WPTR(j), row, k_ / 16, wr[j][h][0], wr[j][h][1]); \
            }                                                                                           \
        }                                                                                               \
    }
#if DUAL
    #define SKD_WPTR(j) ((j) >= SK_FN ? p.w2 : p.w)
#else
    #define SKD_WPTR(j) p.w
#endif

    const uint blocks = K / 64;
    if (sg < blocks)
    {
        SKD_FETCH(sg)
    }
    for (uint kb = sg; kb < blocks; kb += SK_SG)
    {
        uint4 xc[2][2], wc[SK_NB][2][2];
        [ForceUnroll]
        for (int h = 0; h < 2; h++)
        {
            xc[h][0] = xr[h][0];
            xc[h][1] = xr[h][1];
            [ForceUnroll] for (int j = 0; j < SK_NB; j++) { wc[j][h][0] = wr[j][h][0]; wc[j][h][1] = wr[j][h][1]; }
        }
        if (kb + SK_SG < blocks)
        {
            SKD_FETCH(kb + SK_SG)
        }

        uint4 wd[SK_NB][2][2];
        [ForceUnroll]
        for (int j = 0; j < SK_NB; j++)
        {
            [ForceUnroll] for (int h = 0; h < 2; h++) sk_decode_row(wc[j][h][0], wc[j][h][1], wd[j][h][0], wd[j][h][1]);
        }
        [ForceUnroll]
        for (uint f = 0; f < 4; f++)
        {
            SkA a;
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
                uint h = e >> 2, w = 2 * f + ((e >> 1) & 1);
                a[e] = sk_half(sk_word(xc[h][0], xc[h][1], w), e & 1);
            }
            [ForceUnroll]
            for (int j = 0; j < SK_NB; j++)
            {
                SkB b;
                [ForceUnroll]
                for (uint e = 0; e < 8; e++)
                {
                    uint h = e >> 2, w = 2 * f + ((e >> 1) & 1);
                    b[e] = sk_half(sk_word(wd[j][h][0], wd[j][h][1], w), e & 1);
                }
                acc[0][j] = coopMatMulAdd<float, false>(a, b, acc[0][j]);
            }
        }
    }
    #undef SKD_FETCH
    #undef SKD_WPTR
}
#endif

[shader("compute")]
[numthreads(SK_SG * 32, 1, 1)]
void gemm_sk(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    GRID_BIND(p);
    uint tid = ltid.x;
    uint sg = tid / 32, lane = tid % 32;
    uint n0 = gid.x * (SK_FN * 16);

    SkC acc[SK_FM][SK_NB];
    [ForceUnroll] for (int i = 0; i < SK_FM; i++) [ForceUnroll] for (int j = 0; j < SK_NB; j++) acc[i][j] = SkC(0.0);
#if SK_DIRECT
    if (sk_direct_layout(tid, lane))
    {
        sk_direct(n0, sg, lane, p, acc);
    }
    else
    {
        [ForceUnroll] for (int j = 0; j < SK_NB; j++) acc[0][j] = SkC(asfloat(0x7FC00000u));
    }
#else
    sk_staged(n0, sg, lane, p, acc);
#endif

    // Sum the subgroups' partial tiles fragment by fragment through the (now free) B slices: subgroup s stores its 16x16
    // f32 tile (and with DUAL its w2 partner) at uint4 offset 64 (2 s (+1)), every thread then adds up whole elements.
    [ForceUnroll]
    for (int i = 0; i < SK_FM; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < SK_FN; j++)
        {
            GroupMemoryBarrierWithGroupSync();
            acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(skB, (1 + DUAL) * sg * 64, 4u);
#if DUAL
            acc[i][j + SK_FN].Store<CoopMatMatrixLayout.RowMajor>(skB, 2 * sg * 64 + 64, 4u);
#endif
            GroupMemoryBarrierWithGroupSync();
            for (uint e = tid; e < 256; e += SK_SG * 32)
            {
                uint t = i * 16 + e / 16, n = n0 + j * 16 + e % 16;
                if (t < p.T && n < N)
                {
                    float s = 0;
                    [ForceUnroll] for (uint z = 0; z < SK_SG; z++) s += asfloat(skB[((1 + DUAL) * z * 256 + e) / 4][e % 4]);
#if DUAL
                    float s2 = 0;
                    [ForceUnroll] for (uint z = 0; z < SK_SG; z++) s2 += asfloat(skB[(2 * z * 256 + 256 + e) / 4][e % 4]);
                    s = epi2(s, s2);
#endif
                    p.y[uint64_t(t) * Y_STRIDE + n] = s;
                }
            }
        }
    }
}
#endif
