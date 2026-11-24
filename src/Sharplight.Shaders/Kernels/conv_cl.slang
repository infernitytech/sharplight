// Convolutions over channels-last 16-bit activations (reduced precision).
//  conv_cl       : implicit GEMM on cooperative matrices (16x16x16, f32 accumulate): C[o, pix] = sum_k W[g*OG + o, k] *
//                  im2col(x)[k, pix] per group g, pix over N*HO*WO, the reduction tap-major (k = (kh * KW + kw) * CG + c).
//                  x is channels-last 16-bit (XS_C = 1, f16 or with XBF16 bf16): a staged item of 8 k is 8 channels of one
//                  input pixel, one 16-byte load and one bounds check; w is dense [O][KTOT] of the same type, tap-major. A
//                  workgroup of WARPS_M x WARPS_N subgroups computes (WARPS_M * FRAG_M * 16) o x (WARPS_N * FRAG_N * 16) pix;
//                  the shared tiles are double buffered and the next K step's global loads are in registers while this
//                  step's products run, so there is one barrier per K step of TBK = 32. y dense [N, O, HO, WO] f32 (+ bias,
//                  fused epilogue); SPLITK > 1 writes partials [SPLITK][N][O][HO*WO] for conv_splitk_reduce (conv.slang).
//                  Needs CG % 8 == 0 and XS_N, XS_H, XS_W multiples of 8 (16-byte aligned items).
//  conv_cl_stage : converts x [N, C, H, W] (strides XS_*, XFMT f32 / f16 / bf16) to the channels-last 16-bit layout, channel
//                  range [C_OFF, C_OFF + C) of a tensor with strides OS_N, OS_H, OS_W (channel stride 1): one convolution's
//                  input, or one part of a channel concatenation that only convolutions read. Each thread converts V (8 or 16)
//                  channels of one pixel (lanes along the pixels, so the reads coalesce) and stores them as one or two 16-byte
//                  words.
#include "common.slang"
#include "conv_epilogue.slang"
using namespace linalg;

#ifndef XBF16
#define XBF16 0
#endif

// Packs 8 floats into 16-bit operand words (round to nearest even).
uint4 pack16(float v[8])
{
    uint e[8];
    [ForceUnroll]
    for (int j = 0; j < 8; j++)
    {
#if XBF16
        uint u = asuint(v[j]);
        e[j] = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
#else
        e[j] = f32tof16(v[j]);
#endif
    }
    return uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16), e[4] | (e[5] << 16), e[6] | (e[7] << 16));
}

#ifdef KERNEL_CONV_CL_STAGE
struct StageParams
{
    uint8_t* x;
    uint4* xs;
    uint pixels;   // N * H * W
    uint total;    // pixels * C / V
};

float stage_load(uint8_t* x, uint64_t index)
{
#if XFMT == FMT_F32
    return ((float*)x)[index];
#elif XFMT == FMT_F16
    return h2f(((uint16_t*)x)[index]);
#else
    return bf2f(((uint16_t*)x)[index]);
#endif
}

[shader("compute")]
[numthreads(256, 1, 1)]
void conv_cl_stage(uint3 dtid : SV_DispatchThreadID, uniform StageParams p)
{
    uint i = dtid.x;
    if (i >= p.total)
    {
        return;
    }

    uint pix = i % p.pixels;
    uint c0 = (i / p.pixels) * V;
    uint n = pix / (H * W);
    uint r = pix - n * (H * W);
    uint h = r / W;
    uint w = r - h * W;
    uint64_t src = uint64_t(n) * XS_N + uint64_t(c0) * XS_C + uint64_t(h) * XS_H + uint64_t(w) * XS_W;
    uint64_t dst = (uint64_t(n) * OS_N + uint64_t(h) * OS_H + uint64_t(w) * OS_W + C_OFF + c0) / 8;
    [ForceUnroll]
    for (uint part = 0; part < V / 8; part++)
    {
        float v[8];
        [ForceUnroll]
        for (uint j = 0; j < 8; j++)
        {
            v[j] = stage_load(p.x, src + uint64_t(part * 8 + j) * XS_C);
        }
        p.xs[dst + part] = pack16(v);
    }
}
#endif

#ifdef KERNEL_CONV_CL
#define KHW (KH * KW)
#define KTOT (CG * KH * KW)
#define TBK 32
// 8-value k groups per K step.
#define KG (TBK / 8)
// Row stride of the shared tiles in 16-bit values: 32 k plus 8 of padding (conflict-free fragment loads).
#define SAS (TBK + 8)
#define NT (WARPS_M * WARPS_N * 32)
#define TBM (WARPS_M * FRAG_M * 16)
#define TBN (WARPS_N * FRAG_N * 16)
#define A_ITEMS (TBM * KG / NT)
#define B_ITEMS (TBN * KG / NT)

#if XBF16
typealias Elem = BFloat16;
#else
typealias Elem = half;
#endif
#if SL_TARGET_CUDA
#include "cuda_mma.slang"
#if XBF16
typealias MatA = CuA16bf;
typealias MatB = CuB16bf;
#else
typealias MatA = CuA16;
typealias MatB = CuB16;
#endif
typealias Acc = CuAccF;
#else
typealias MatA = CoopMat<Elem, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<Elem, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias Acc = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;
#endif

struct ConvParams
{
    uint4* x;
    uint4* w;
    float* bias;
    float* y;
#if EPI_RES
    float* res;
#endif
    uint N;
};

// A: weights [o][k] (row-major), B: im2col [pix][k] (column-major B); 8 packed operands per uint4, two buffers each.
groupshared uint4 sA[2 * TBM * SAS / 8];
groupshared uint4 sB[2 * TBN * SAS / 8];
groupshared float sC[WARPS_M * WARPS_N * 256];

Acc mma(MatA a, MatB b, Acc c) { return coopMatMulAdd<float, false>(a, b, c); }

// Global loads of the K step at k0 into registers; zero outside the tensors and at or past kEnd. A item it: row
// (tid + it * NT) / KG; B item it: pixel (tid + it * NT) / KG; both at the thread's k group q = tid % KG.
void fetch(ConvParams p, uint g, uint mBase, uint k0, uint kEnd, uint tid, uint64_t bbase[B_ITEMS], int ih0[B_ITEMS], int iw0[B_ITEMS],
    out uint4 a[A_ITEMS], out uint4 b[B_ITEMS])
{
    uint kq = k0 + (tid % KG) * 8;
    bool kin = kq < kEnd;
    [ForceUnroll]
    for (uint it = 0; it < A_ITEMS; it++)
    {
        uint r = (tid + it * NT) / KG;
        a[it] = 0;
        if (kin && (OG % TBM == 0 || mBase + r < OG))
        {
            a[it] = p.w[(uint64_t(g * OG + mBase + r) * KTOT + kq) / 8];
        }
    }

    uint tap = kq / CG;
    uint c = kq - tap * CG;
    uint kh = tap / KW;
    int dh = int(kh) * DH;
    int dw = int(tap - kh * KW) * DW;
    uint64_t koff = uint64_t(int64_t(dh) * XS_H + int64_t(dw) * XS_W) + g * CG + c;
    [ForceUnroll]
    for (uint it = 0; it < B_ITEMS; it++)
    {
        int ih = ih0[it] + dh;
        int iw = iw0[it] + dw;
        b[it] = 0;
#if NOPAD
        if (kin && ih0[it] != 0x40000000)
#else
        if (kin && uint(ih) < uint(H) && uint(iw) < uint(W))
#endif
        {
            b[it] = p.x[(bbase[it] + koff) / 8];
        }
    }
}

void stage(uint buf, uint tid, uint4 a[A_ITEMS], uint4 b[B_ITEMS])
{
    [ForceUnroll]
    for (uint it = 0; it < A_ITEMS; it++)
    {
        uint idx = tid + it * NT;
        sA[(buf * TBM + idx / KG) * (SAS / 8) + idx % KG] = a[it];
    }
    [ForceUnroll]
    for (uint it = 0; it < B_ITEMS; it++)
    {
        uint idx = tid + it * NT;
        sB[(buf * TBN + idx / KG) * (SAS / 8) + idx % KG] = b[it];
    }
}

[shader("compute")]
[numthreads(NT, 1, 1)]
void conv_cl(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID, uniform ConvParams p)
{
    const uint tid = ltid.x;
    const uint sg = tid / 32;
    const uint g = gid.z / SPLITK;
    const uint split = gid.z % SPLITK;
    const uint HW = HO * WO;
    const uint NC = p.N * HW;
    const uint mBase = gid.y * TBM;
    const uint colBase = gid.x * TBN;
    const uint kBeg = split * KPS;
    const uint kEnd = min(uint(KTOT), kBeg + KPS);
    const uint wm = (sg / WARPS_N) * (FRAG_M * 16);
    const uint wn = (sg % WARPS_N) * (FRAG_N * 16);

    // The input pixels of the thread's B items: image base and the top-left tap's row and column (rows far outside the map
    // mark pixels past the batch, so the bounds check rejects them without another flag).
    uint64_t bbase[B_ITEMS];
    int ih0[B_ITEMS];
    int iw0[B_ITEMS];
    [ForceUnroll]
    for (uint it = 0; it < B_ITEMS; it++)
    {
        uint bj = colBase + (tid + it * NT) / KG;
        uint bn = bj / HW;
        uint bpix = bj - bn * HW;
        uint boh = bpix / WO;
        uint bow = bpix - boh * WO;
        bool valid = bj < NC;
        bbase[it] = valid ? uint64_t(bn) * XS_N : 0;
        ih0[it] = valid ? int(boh) * SH - PH : 0x40000000;
        iw0[it] = int(bow) * SW - PW;
        bbase[it] += uint64_t(int64_t(ih0[it] == 0x40000000 ? 0 : ih0[it]) * XS_H + int64_t(iw0[it]) * XS_W);
    }

    Acc acc[FRAG_M][FRAG_N];
    [ForceUnroll]
    for (int i = 0; i < FRAG_M; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j++)
        {
            acc[i][j] = Acc(0.0);
        }
    }

    uint4 a[A_ITEMS];
    uint4 b[B_ITEMS];
    fetch(p, g, mBase, kBeg, kEnd, tid, bbase, ih0, iw0, a, b);
    stage(0, tid, a, b);
    GroupMemoryBarrierWithGroupSync();
    uint buf = 0;
    for (uint k0 = kBeg; k0 < kEnd; k0 += TBK)
    {
        bool more = k0 + TBK < kEnd;
        if (more)
        {
            fetch(p, g, mBase, k0 + TBK, kEnd, tid, bbase, ih0, iw0, a, b);
        }

        [ForceUnroll]
        for (uint kk = 0; kk < TBK; kk += 16)
        {
            MatB bf[FRAG_N];
            [ForceUnroll]
            for (int j = 0; j < FRAG_N; j++)
            {
                bf[j] = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sB, ((buf * TBN + wn + j * 16) * SAS + kk) / 8, SAS / 8);
            }
            [ForceUnroll]
            for (int i = 0; i < FRAG_M; i++)
            {
                MatA af = MatA.Load<CoopMatMatrixLayout.RowMajor>(sA, ((buf * TBM + wm + i * 16) * SAS + kk) / 8, SAS / 8);
                [ForceUnroll]
                for (int j = 0; j < FRAG_N; j++)
                {
                    acc[i][j] = mma(af, bf[j], acc[i][j]);
                }
            }
        }

        if (more)
        {
            stage(buf ^ 1, tid, a, b);
        }
        GroupMemoryBarrierWithGroupSync();
        buf ^= 1;
    }

    // ---- epilogue: each subgroup stages one accumulator at a time in its shared slice; guarded store (+ bias) ----
    uint64_t outBase = uint64_t(split) * uint64_t(p.N) * (O * HW);
    const uint lane = tid % 32;
    [ForceUnroll]
    for (int i = 0; i < FRAG_M; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j++)
        {
            acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(sC, sg * 256, 16u);
            GroupMemoryBarrierWithGroupSync();
            [ForceUnroll]
            for (uint e = 0; e < 8; e++)
            {
                uint w = lane + e * 32;
                uint m = mBase + wm + i * 16 + w / 16;
                uint col = colBase + wn + j * 16 + w % 16;
                if (m < OG && col < NC)
                {
                    uint n = col / HW;
                    uint pix = col - n * HW;
                    uint o = g * OG + m;
                    float v = sC[sg * 256 + w];
                    uint64_t idx = outBase + uint64_t(n) * (O * HW) + uint64_t(o) * HW + pix;
#if SPLITK == 1
#if HAS_BIAS
                    v += p.bias[o];
#endif
                    CONV_EPI(v, idx);
#endif
                    p.y[idx] = v;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
#endif
