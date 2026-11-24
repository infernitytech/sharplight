// Convolutions over channels-last 16-bit activations (reduced precision).
//  conv_cl       : implicit GEMM on cooperative matrices (16x16x16, f32 accumulate): C[o, pix] = sum_k W[g*OG + o, k] *
//                  im2col(x)[k, pix] per group g, pix over N*HO*WO, the reduction tap-major (k = (kh * KW + kw) * CG + c).
//                  x is channels-last 16-bit (XS_C = 1, f16 or with XBF16 bf16): a staged item of 8 k is 8 channels of one
//                  input pixel, one 16-byte load and one bounds check; w is dense [O][KTOT] of the same type, tap-major. A
//                  workgroup of WARPS_M x WARPS_N subgroups computes (WARPS_M * FRAG_M * 16) o x (WARPS_N * FRAG_N * 16) pix,
//                  one barrier per K step of TBK = 32: the shared tiles are double buffered with the next step's global loads
//                  in registers during this step's products, or (CL_ASYNC, CUDA) a ring of CL_STAGES buffers filled by
//                  asynchronous copies. y dense [N, O, HO, WO] f32 (+ bias, fused epilogue), stored along the pixels through
//                  the A tiles; SPLITK > 1 writes partials [SPLITK][N][O][HO*WO] for conv_splitk_reduce (conv.slang).
//                  Needs CG % 8 == 0 and XS_N, XS_H, XS_W multiples of 8 (16-byte aligned items). X_PARTS > 1: x is a channel
//                  concatenation read from its parts in place (see ConvParams).
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
#define M_TILES ((OG + TBM - 1) / TBM)
// CL_ASYNC (CUDA, compute capability 8.0 and later): the tiles stream into a ring of CL_STAGES shared buffers by asynchronous
// copies (cp.async, zero-filled outside the tensors), so the loads of the next CL_STAGES - 1 steps are in flight during each
// step's products without passing through registers. Otherwise two buffers and a register prefetch of the next step.
#ifndef CL_ASYNC
#define CL_ASYNC 0
#endif
#if CL_ASYNC
#define NBUF CL_STAGES
#else
#define NBUF 2
#endif

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

// X_PARTS > 1: x is the channel concatenation of X_PARTS tensors of PART_C channels each with the same strides (x, x1, x2, x3),
// e.g. the stacked frame windows of a causal 3-D convolution, read in place (one group).
#ifndef X_PARTS
#define X_PARTS 1
#endif
#if X_PARTS > 1 && G != 1
#error "conv_cl: a concatenated input takes one group"
#endif

struct ConvParams
{
    uint4* x;
    uint4* w;
    float* bias;
    float* y;
#if X_PARTS > 1
    uint4* x1;
#endif
#if X_PARTS > 2
    uint4* x2;
#endif
#if X_PARTS > 3
    uint4* x3;
#endif
#if EPI_RES
    float* res;
#endif
    uint N;
};

// A: weights [o][k] (row-major), B: im2col [pix][k] (column-major B); 8 packed operands per uint4, NBUF buffers each. The epilogue
// reuses the A tiles for one row of f32 results per subgroup (16 x FRAG_N * 16 values).
#if NBUF * TBM * SAS * 2 < WARPS_M * WARPS_N * FRAG_N * 1024
#error "conv_cl: the A tiles cannot hold the epilogue's results"
#endif
groupshared uint4 sA[NBUF * TBM * SAS / 8];
groupshared uint4 sB[NBUF * TBN * SAS / 8];

Acc mma(MatA a, MatB b, Acc c) { return coopMatMulAdd<float, false>(a, b, c); }

// The thread's input pixels (B items): image base plus the top-left tap's offset, and that tap's row and column (rows far outside
// the map mark pixels past the batch, so the bounds check rejects them without another flag).
struct Pixels
{
    uint64_t base[B_ITEMS];
    int ih0[B_ITEMS];
    int iw0[B_ITEMS];
};

// The K step at k0 for the thread's k group q = tid % KG: the address of the input tensor (part) it reads, the element offset of
// the tap and channel within a pixel, the tap's row and column offsets, and whether k0 + 8 q is inside the split.
struct KStep
{
    uint64_t x;
    uint64_t offset;
    int dh;
    int dw;
    bool inside;
};

KStep k_step(ConvParams p, uint g, uint k0, uint kEnd, uint tid)
{
    KStep s;
    uint kq = k0 + (tid % KG) * 8;
    uint tap = kq / CG;
    uint c = kq - tap * CG;
    uint kh = tap / KW;
    s.dh = int(kh) * DH;
    s.dw = int(tap - kh * KW) * DW;
#if X_PARTS > 1
    uint part = c / PART_C;
    c -= part * PART_C;
    s.x = uint64_t(p.x);
    s.x = part == 1 ? uint64_t(p.x1) : s.x;
#if X_PARTS > 2
    s.x = part == 2 ? uint64_t(p.x2) : s.x;
#endif
#if X_PARTS > 3
    s.x = part == 3 ? uint64_t(p.x3) : s.x;
#endif
#else
    s.x = uint64_t(p.x);
#endif
    s.offset = uint64_t(int64_t(s.dh) * XS_H + int64_t(s.dw) * XS_W) + g * CG + c;
    s.inside = kq < kEnd;
    return s;
}

bool a_inside(uint mBase, uint tid, uint it, KStep s)
{
    return s.inside && (OG % TBM == 0 || mBase + (tid + it * NT) / KG < OG);
}

// Index (in uint4s) of A item it's 8 weights.
uint64_t a_index(uint g, uint mBase, uint k0, uint tid, uint it)
{
    return (uint64_t(g * OG + mBase + (tid + it * NT) / KG) * KTOT + k0 + (tid % KG) * 8) / 8;
}

bool b_inside(Pixels px, uint it, KStep s)
{
    int ih = px.ih0[it] + s.dh;
    int iw = px.iw0[it] + s.dw;
#if NOPAD
    return s.inside && px.ih0[it] != 0x40000000;
#else
    return s.inside && uint(ih) < uint(H) && uint(iw) < uint(W);
#endif
}

// Shared index (in uint4s) of item it of buffer buf: A row / B pixel (tid + it * NT) / KG, k group tid % KG.
uint a_slot(uint buf, uint tid, uint it) { return (buf * TBM + (tid + it * NT) / KG) * (SAS / 8) + tid % KG; }
uint b_slot(uint buf, uint tid, uint it) { return (buf * TBN + (tid + it * NT) / KG) * (SAS / 8) + tid % KG; }

#if CL_ASYNC
// Issues the copies of the K step at k0 into buffer buf.
void issue(ConvParams p, uint g, uint mBase, uint k0, uint kEnd, uint tid, Pixels px, uint buf)
{
    KStep s = k_step(p, g, k0, kEnd, tid);
    [ForceUnroll]
    for (uint it = 0; it < A_ITEMS; it++)
    {
        bool in = a_inside(mBase, tid, it, s);
        cu_cp_async16(sA, a_slot(buf, tid, it), uint64_t(p.w) + (in ? a_index(g, mBase, k0, tid, it) * 16 : 0), in);
    }
    [ForceUnroll]
    for (uint it = 0; it < B_ITEMS; it++)
    {
        bool in = b_inside(px, it, s);
        cu_cp_async16(sB, b_slot(buf, tid, it), s.x + (in ? (px.base[it] + s.offset) * 2 : 0), in);
    }
}
#else
// Global loads of the K step at k0 into registers; zero outside the tensors and at or past kEnd.
void fetch(ConvParams p, uint g, uint mBase, uint k0, uint kEnd, uint tid, Pixels px, out uint4 a[A_ITEMS], out uint4 b[B_ITEMS])
{
    KStep s = k_step(p, g, k0, kEnd, tid);
    [ForceUnroll]
    for (uint it = 0; it < A_ITEMS; it++)
    {
        a[it] = 0;
        if (a_inside(mBase, tid, it, s))
        {
            a[it] = p.w[a_index(g, mBase, k0, tid, it)];
        }
    }
    [ForceUnroll]
    for (uint it = 0; it < B_ITEMS; it++)
    {
        b[it] = 0;
        if (b_inside(px, it, s))
        {
            b[it] = *(uint4*)(s.x + (px.base[it] + s.offset) * 2);
        }
    }
}

void stage(uint buf, uint tid, uint4 a[A_ITEMS], uint4 b[B_ITEMS])
{
    [ForceUnroll]
    for (uint it = 0; it < A_ITEMS; it++)
    {
        sA[a_slot(buf, tid, it)] = a[it];
    }
    [ForceUnroll]
    for (uint it = 0; it < B_ITEMS; it++)
    {
        sB[b_slot(buf, tid, it)] = b[it];
    }
}
#endif

// The products of the K step in buffer buf.
void products(uint buf, uint wm, uint wn, inout Acc acc[FRAG_M][FRAG_N])
{
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
    // The output-channel tiles of a pixel tile are adjacent workgroups, so they read its input while it is in L2.
    const uint mBase = (gid.x % M_TILES) * TBM;
    const uint colBase = (gid.x / M_TILES) * TBN;
    const uint kBeg = split * KPS;
    const uint kEnd = min(uint(KTOT), kBeg + KPS);
    const uint wm = (sg / WARPS_N) * (FRAG_M * 16);
    const uint wn = (sg % WARPS_N) * (FRAG_N * 16);

    Pixels px;
    [ForceUnroll]
    for (uint it = 0; it < B_ITEMS; it++)
    {
        uint bj = colBase + (tid + it * NT) / KG;
        uint bn = bj / HW;
        uint bpix = bj - bn * HW;
        uint boh = bpix / WO;
        uint bow = bpix - boh * WO;
        bool valid = bj < NC;
        px.ih0[it] = valid ? int(boh) * SH - PH : 0x40000000;
        px.iw0[it] = int(bow) * SW - PW;
        px.base[it] = valid ? uint64_t(bn) * XS_N + uint64_t(int64_t(px.ih0[it]) * XS_H + int64_t(px.iw0[it]) * XS_W) : 0;
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

#if CL_ASYNC
    const uint steps = (kEnd - kBeg + TBK - 1) / TBK;
    [ForceUnroll]
    for (uint s = 0; s + 1 < NBUF; s++)
    {
        if (s < steps)
        {
            issue(p, g, mBase, kBeg + s * TBK, kEnd, tid, px, s);
        }
        cu_cp_commit();
    }
    for (uint step = 0; step < steps; step++)
    {
#if NBUF == 2
        cu_cp_wait0();
#elif NBUF == 3
        cu_cp_wait1();
#elif NBUF == 4
        cu_cp_wait2();
#else
        cu_cp_wait3();
#endif
        GroupMemoryBarrierWithGroupSync();
        uint next = step + NBUF - 1;
        if (next < steps)
        {
            issue(p, g, mBase, kBeg + next * TBK, kEnd, tid, px, next % NBUF);
        }
        cu_cp_commit();
        products(step % NBUF, wm, wn, acc);
    }
    cu_cp_wait0();
#else
    uint4 a[A_ITEMS];
    uint4 b[B_ITEMS];
    fetch(p, g, mBase, kBeg, kEnd, tid, px, a, b);
    stage(0, tid, a, b);
    GroupMemoryBarrierWithGroupSync();
    uint buf = 0;
    for (uint k0 = kBeg; k0 < kEnd; k0 += TBK)
    {
        bool more = k0 + TBK < kEnd;
        if (more)
        {
            fetch(p, g, mBase, k0 + TBK, kEnd, tid, px, a, b);
        }

        products(buf, wm, wn, acc);
        if (more)
        {
            stage(buf ^ 1, tid, a, b);
        }
        GroupMemoryBarrierWithGroupSync();
        buf ^= 1;
    }
#endif

    // ---- epilogue: per row of fragments, each subgroup stages its 16 x (FRAG_N * 16) results in the (now free) A tiles and
    // stores them along the pixels, 32 consecutive outputs per store (+ bias, fused epilogue) ----
    GroupMemoryBarrierWithGroupSync();
    uint64_t outBase = uint64_t(split) * uint64_t(p.N) * (O * HW);
    const uint lane = tid % 32;
    const uint slice = sg * (FRAG_N * 256);
    [ForceUnroll]
    for (int i = 0; i < FRAG_M; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < FRAG_N; j++)
        {
            acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(sA, (slice + j * 16) / 4, FRAG_N * 4);
        }
        GroupMemoryBarrierWithGroupSync();
        // All loads (staged results, bias, residual) before the first store: the compiler cannot tell that y does not alias them,
        // so a load after a store would wait for it (a fused residual add ran 3.5 x slower with the loads between the stores).
        float v[FRAG_N * 8];
        uint64_t at[FRAG_N * 8];
        bool inside[FRAG_N * 8];
        [ForceUnroll]
        for (uint e = 0; e < FRAG_N * 8; e++)
        {
            uint w = lane + e * 32;
            uint m = mBase + wm + i * 16 + w / (FRAG_N * 16);
            uint col = colBase + wn + w % (FRAG_N * 16);
            inside[e] = m < OG && col < NC;
            uint n = col / HW;
            uint o = g * OG + m;
            at[e] = outBase + uint64_t(n) * (O * HW) + uint64_t(o) * HW + (col - n * HW);
            uint staged = slice + w;
            v[e] = asfloat(sA[staged / 4][staged % 4]);
#if SPLITK == 1 && HAS_BIAS
            v[e] += inside[e] ? p.bias[o] : 0.0;
#endif
#if SPLITK == 1 && EPI && EPI_RES
            v[e] = conv_epi(v[e], inside[e] ? p.res[at[e]] : 0.0);
#elif SPLITK == 1 && EPI
            v[e] = conv_epi(v[e], 0.0);
#endif
        }
        [ForceUnroll]
        for (uint e = 0; e < FRAG_N * 8; e++)
        {
            if (inside[e])
            {
                p.y[at[e]] = v[e];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
#endif
