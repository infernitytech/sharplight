// Batched matrix multiply on strided f32 inputs with broadcast batch dims.
//  y[b, m, n] = ALPHA * sum_k A[b, m, k] * B[b, k, n]
// A element address: aoff(b) + m*SAM + k*SAK ; B: boff(b) + k*SBK + n*SBN ; y dense [B..., M, N].
// Macros: M N K NB (batch dims, may be 0) BSH BSA BSB (per batch dim: shape [outermost unused], A stride, B stride),
//         SAM SAK SBK SBN ALPHA M_DYN (M comes from the uniform) BM BN BK TM TN NT
//         SPLITK KPS WS_STRIDE PARTIAL: with PARTIAL (few output tiles and a long K, e.g. attention P.V), grid z is
//         batch * SPLITK; each split sums KPS values of k and writes its unscaled partial to y + split * WS_STRIDE (a
//         workspace); bmm_reduce sums the partials and applies ALPHA.
#include "common.slang"

#ifndef SPLITK
#define SPLITK 1
#define KPS K
#define WS_STRIDE 0
#define PARTIAL 0
#endif

#if PARTIAL
#define OUT_SCALE 1.0
#else
#define OUT_SCALE ALPHA
#endif

#ifdef KERNEL_BMM_REDUCE
struct BmmReduceParams { float* ws; float* y; uint total; };

[shader("compute")]
[numthreads(256, 1, 1)]
void bmm_reduce(uint3 dtid : SV_DispatchThreadID, uniform BmmReduceParams p)
{
    uint i = dtid.x;
    if (i >= p.total) return;
    float s = 0.0;
    [ForceUnroll]
    for (uint k = 0; k < SPLITK; k++) s += p.ws[uint64_t(k) * WS_STRIDE + i];
    p.y[i] = s * ALPHA;
}
#endif

#ifdef KERNEL_BMM

struct BmmParams { float* a; float* b; float* y; uint dyn; };

#if NB > 0
static const int Q_BSH[NB] = { BSH };
static const int Q_BSA[NB] = { BSA };
static const int Q_BSB[NB] = { BSB };
#endif

groupshared float As[BK][BM + 1];
groupshared float Bs[BK][BN + 1];

[shader("compute")]
[numthreads(NT, 1, 1)]
void bmm(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform BmmParams p)
{
    const uint tid = ltid.x;
#if M_DYN
    const uint Mv = p.dyn;
#else
    const uint Mv = M;
#endif
    uint bi = gid.z / SPLITK;
    const uint split = gid.z % SPLITK;
    const uint kBegin = split * KPS;
    const uint kEnd = min(uint(K), kBegin + KPS);
    uint64_t aoff = 0, boff = 0;
#if NB > 0
    uint64_t yb = uint64_t(bi) * uint64_t(Mv) * N;
    [ForceUnroll]
    for (int d = NB - 1; d >= 1; d--)
    {
        uint c = bi % uint(Q_BSH[d]);
        bi /= uint(Q_BSH[d]);
        aoff += uint64_t(c) * uint64_t(Q_BSA[d]);
        boff += uint64_t(c) * uint64_t(Q_BSB[d]);
    }
    aoff += uint64_t(bi) * uint64_t(Q_BSA[0]);
    boff += uint64_t(bi) * uint64_t(Q_BSB[0]);
#else
    uint64_t yb = 0;
#endif
    yb += uint64_t(split) * WS_STRIDE;
    const uint mBase = gid.y * BM;
    const uint nBase = gid.x * BN;
    float* ap = p.a + aoff;
    float* bp = p.b + boff;

    float acc[TM][TN];
    [ForceUnroll] for (int i = 0; i < TM; i++)
        [ForceUnroll] for (int j = 0; j < TN; j++) acc[i][j] = 0.0;
    const uint tx = tid % (BN / TN);
    const uint ty = tid / (BN / TN);

    for (uint k0 = kBegin; k0 < kEnd; k0 += BK)
    {
                for (uint i = tid; i < BM * BK; i += NT)
        {
#if SAK == 1
            uint kk = i % BK, m = i / BK;
#else
            uint m = i % BM, kk = i / BM;
#endif
            float v = 0.0;
            if (mBase + m < Mv && k0 + kk < kEnd) v = ap[uint64_t(mBase + m) * SAM + uint64_t(k0 + kk) * SAK];
            As[kk][m] = v;
        }
                for (uint i = tid; i < BK * BN; i += NT)
        {
#if SBN == 1
            uint n = i % BN, kk = i / BN;
#else
            uint kk = i % BK, n = i / BK;
#endif
            float v = 0.0;
            if (nBase + n < N && k0 + kk < kEnd) v = bp[uint64_t(k0 + kk) * SBK + uint64_t(nBase + n) * SBN];
            Bs[kk][n] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint kk = 0; kk < BK; kk++)
        {
            float a[TM], b[TN];
            [ForceUnroll] for (int i = 0; i < TM; i++) a[i] = As[kk][ty * TM + i];
            [ForceUnroll] for (int j = 0; j < TN; j++) b[j] = Bs[kk][tx * TN + j];
            [ForceUnroll] for (int i = 0; i < TM; i++)
                [ForceUnroll] for (int j = 0; j < TN; j++) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    [ForceUnroll]
    for (int i = 0; i < TM; i++)
    {
        uint m = mBase + ty * TM + i;
        if (m >= Mv) continue;
#if N % 4 == 0 && TN % 4 == 0
        // Each thread's TN outputs of a row are contiguous: 16-byte stores keep the warp's writes in whole sectors.
        [ForceUnroll]
        for (int j = 0; j < TN; j += 4)
        {
            uint n = nBase + tx * TN + j;
            if (n < N)
            {
                *(float4*)(p.y + yb + uint64_t(m) * N + n) = float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]) * OUT_SCALE;
            }
        }
        continue;
#endif
        [ForceUnroll]
        for (int j = 0; j < TN; j++)
        {
            uint n = nBase + tx * TN + j;
            if (n < N) p.y[yb + uint64_t(m) * N + n] = acc[i][j] * OUT_SCALE;
        }
    }
}
#endif
