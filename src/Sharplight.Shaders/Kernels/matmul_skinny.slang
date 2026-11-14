#pragma once
// Skinny tensor-core GEMM for few activation rows (decode batches, speculative verification, 2-64 rows), included by
// matmul.slang (uses its Params, load16h and the packed-half decoders of the gemm_cm2 section).
//
// Workgroup = SK_SG subgroups that all own the same SK_FN x 16 weight rows and split K between them: subgroup s walks the
// K blocks s, s + SK_SG, ... of SK_BK values, and the partial accumulators are summed through shared memory at the end. So
// every weight is read from memory exactly once, by one workgroup, with no split-K scratch buffer and no reduction launch,
// and the grid has N / (16 SK_FN) workgroups however few rows there are. Each subgroup stages its own A (SK_FM x 16 token
// rows) and B slices into a private part of shared memory, so the K loop has subgroup barriers only. Token rows past T
// are staged as zeros and their results are dropped.
//
// x: f16 rows (X_F16 = 1, row stride X_STRIDE halves) or f32 rows converted while staging (X_F16 = 0). DUAL: w and w2
// (same shape and format) give SK_FN fragments each, combined by epi2 per element before the store.
// Requires K % SK_BK == 0.
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
#define SK_NB (SK_FN * (1 + DUAL))          // B fragments (16 weight rows each) per subgroup
#define SK_AS (SK_BK + 8)                   // staged row stride in halves (padding against bank conflicts)
#define SK_A_PER_LANE (SK_FM * 16 * SK_BK / 8 / 32)
#define SK_B_PER_LANE (SK_NB * 16 * SK_BK / 16 / 32)
#define SK_A_WORDS (SK_FM * 16 * SK_AS / 8)  // uint4 per subgroup
#define SK_B_WORDS (SK_NB * 16 * SK_AS / 8)

typealias SkA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias SkB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias SkC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

groupshared uint4 skA[SK_SG * SK_A_WORDS];
groupshared uint4 skB[SK_SG * SK_B_WORDS];

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

[shader("compute")]
[numthreads(SK_SG * 32, 1, 1)]
void gemm_sk(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform Params p)
{
    GRID_BIND(p);
    uint tid = ltid.x;
    uint sg = tid / 32, lane = tid % 32;
    uint n0 = gid.x * (SK_FN * 16);
    uint aBase = sg * SK_A_WORDS, bBase = sg * SK_B_WORDS;

    SkC acc[SK_FM][SK_NB];
    [ForceUnroll] for (int i = 0; i < SK_FM; i++) [ForceUnroll] for (int j = 0; j < SK_NB; j++) acc[i][j] = SkC(0.0);

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
