// K-quant decode GEMM written in CUDA C++ for the CUDA provider (DeviceFeatures.NativeGemmKQuant, compute capability 8.0 and later):
// y[t, n] = sum_k x[t, k] * W[n, k] for batches of at most KQ_BM token rows per token tile (decode steps of 9-64 sequences), with f16
// activations (the generated GEMMs' f16 copy of x) and Q4_K / Q5_K / Q6_K weights in the GGUF block layout (KQ_FMT 4 / 5 / 6), f32
// accumulation and output. Compiled by NVRTC from this file.
//
// A block covers KQ_BM (16, 32 or 64) token rows and 64 weight rows; each of its eight warps takes 8 weight rows and every token row,
// so each weight value is dequantized once. A K step is one 256-value superblock: cp.async copies the step's activation rows (four
// 64-value sub-tiles of 128-byte rows, XOR swizzle, read by ldmatrix) and the raw blocks of the 64 weight rows (each row's 16-byte-aligned window around its block) into a ring of KQ_STAGES
// stages in dynamic shared memory. Each thread dequantizes the weight values of its mma.sync.m16n8k16 B fragments straight from the
// raw blocks into f16 pairs (exact integer codes times the f16 sub-block scale, minus the f16 sub-block minimum for Q4_K / Q5_K), so
// the weights are read from memory once and never written back as f16.
//
// KQ_SPLIT > 1 splits the superblocks between the blocks of grid z: each writes its partial sums to slice z of part ([KQ_SPLIT][T][N]),
// publishes them and counts its arrival in arrivals[tile]; the last block of a tile sums the slices in slice order (z = 0 first, from
// zero, so the result does not depend on which block arrives last), stores the outputs and resets the counter (every launch leaves
// the counters zero). Blocks never wait for each other.
//
// MULTI (2 or 3): the multi-output form over the concatenated rows of two or three matrices of one format and K (q / k / v), each with
// its own output (dense rows of its width) and, with H_BIAS, its own bias: rows [0, MULTI_O1) are w's into y, [MULTI_O1, MULTI_O2)
// w_1's into y_1, the rest w_2's into y_2 (MULTI_O1, MULTI_O2 multiples of 64, so a block's weight tile lies in one matrix).
//
// H_BIAS: the store adds bias[n]. H_XRS: x is a row-ranged f16 copy whose scale words start at byte H_XRS of x (common.slang,
// f16_range_scale); the store multiplies each row's sums by the inverse of its scale (before the bias).
//
// Specialization macros: K (a multiple of 256), N (all rows with MULTI), Y_STRIDE, KQ_FMT, MULTI, MULTI_O1, MULTI_O2, KQ_BM (16, 32 or
// 64), KQ_STAGES, KQ_SPLIT, H_BIAS, H_XRS. Requires 16-byte aligned x and weights; weight rows may start at any even byte (Q6_K rows of 210-byte blocks). Grid: x = token tiles,
// y = weight tiles (64 rows), z = K splits.
#ifndef H_BIAS
#define H_BIAS 0
#endif
#ifndef MULTI
#define MULTI 0
#endif
#ifndef KQ_DUAL
#define KQ_DUAL 0
#endif
#if KQ_DUAL && (MULTI || H_BIAS)
#error "gemm_kq_native: the dual form has neither members nor biases"
#endif
#ifndef KQ_SPLIT
#define KQ_SPLIT 1
#endif
#ifndef KQ_BM
#define KQ_BM 64
#endif
#ifndef KQ_STAGES
#define KQ_STAGES 2
#endif
#ifndef KQ_PF
#define KQ_PF 0
#endif
#ifndef KQ_PFB
#define KQ_PFB 0
#endif
#if KQ_FMT == 4
#define KQ_BB 144
#elif KQ_FMT == 5
#define KQ_BB 176
#elif KQ_FMT == 6
#define KQ_BB 210
#else
#error "gemm_kq_native: KQ_FMT 4, 5 or 6"
#endif
#define KQ_BN 64
#define KQ_THREADS 256
#define KQ_FN 1
#define KQ_TM KQ_BM
#define KQ_FM (KQ_TM / 16)
#define KQ_SB (K / 256)
#define KQ_ROWB ((unsigned long long)KQ_SB * KQ_BB)
// 16-byte chunks of one weight row's window per stage: the block itself when every block starts 16-byte aligned, else the block
// plus up to 14 bytes before it.
#if KQ_BB % 16 == 0 && (KQ_SB * KQ_BB) % 16 == 0
#define KQ_WCH (KQ_BB / 16)
#else
#define KQ_WCH ((KQ_BB + 14 + 15) / 16)
#endif
#define KQ_XBYTES (KQ_BM * 512)
#define KQ_STAGE_BYTES (KQ_XBYTES + KQ_BN * KQ_WCH * 16)
#define KQ_XRS_BYTES ((unsigned long long)K * 2)
// Columns of the partial slices: the dual form's interleaved rows of both matrices.
#if KQ_DUAL
#define KQ_PN (2 * N)
#else
#define KQ_PN N
#endif

typedef unsigned int kq_u32;
typedef unsigned long long kq_u64;

struct KqParams
{
    const unsigned char* x;
    const unsigned char* w;
    float* y;
#if H_BIAS
    const float* bias;
#endif
#if MULTI || KQ_DUAL
    const unsigned char* w_1;
#endif
#if MULTI
#if MULTI > 2
    const unsigned char* w_2;
#endif
    float* y_1;
#if MULTI > 2
    float* y_2;
#endif
#if H_BIAS
    const float* bias_1;
#if MULTI > 2
    const float* bias_2;
#endif
#endif
#endif
    float* part;        // KQ_SPLIT > 1: the partial slices
    kq_u32* arrivals;   // KQ_SPLIT > 1: one counter per output tile
    kq_u32 T;
};

static __device__ __forceinline__ void kq_cp16(kq_u32 dst, const void* src, kq_u32 bytes)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes));
}

static __device__ __forceinline__ void kq_ldsm4(kq_u32 addr, kq_u32& r0, kq_u32& r1, kq_u32& r2, kq_u32& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

static __device__ __forceinline__ void kq_mma(float* c, const kq_u32* a, kq_u32 b0, kq_u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Byte offset of 16-byte chunk c of row r in an activation sub-tile of 128-byte rows (chunk c ^ (r % 8)).
static __device__ __forceinline__ kq_u32 kq_swz(kq_u32 r, kq_u32 c)
{
    return r * 128u + ((c ^ (r & 7u)) << 4);
}

static __device__ __forceinline__ kq_u32 kq_ld16(const unsigned char* p)
{
    return *(const unsigned short*)p;
}

// Two code bytes (low byte first) spread to the two 16-bit halves: [b0, 0, b1, 0].
static __device__ __forceinline__ kq_u32 kq_spread(kq_u32 v)
{
    return __byte_perm(v, 0u, 0x4140u);
}

// f16 pairs as 32-bit words (lo in the low half); NVRTC compiles this file without the CUDA headers.
// The f16 pair of two integer codes c0, c1 < 1024 held in the low bits of the halves of v, minus `bias` (an f16 pair): exact.
static __device__ __forceinline__ kq_u32 kq_codes(kq_u32 v, kq_u32 bias)
{
    kq_u32 r;
    asm("sub.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(v | 0x64006400u), "r"(bias));
    return r;
}

// Both halves set to f, round to nearest.
static __device__ __forceinline__ kq_u32 kq_splat(float f)
{
    kq_u32 r;
    asm("cvt.rn.f16x2.f32 %0, %1, %1;\n" : "=r"(r) : "f"(f));
    return r;
}

static __device__ __forceinline__ kq_u32 kq_fma2(kq_u32 a, kq_u32 b, kq_u32 c)
{
    kq_u32 r;
    asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

static __device__ __forceinline__ kq_u32 kq_mul2(kq_u32 a, kq_u32 b)
{
    kq_u32 r;
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// The f16 at p (2-byte aligned) as a float.
static __device__ __forceinline__ float kq_half(const unsigned char* p)
{
    float f;
    asm("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(*(const unsigned short*)p));
    return f;
}

// The matrix a block's weight tile lies in (MULTI: its member's), with that member's output, bias, row count and output row stride.
struct KqMember
{
    const unsigned char* w;
    float* y;
#if H_BIAS
    const float* bias;
#endif
    kq_u32 rows;
    kq_u32 stride;
};

// The member of weight row n0 (a tile's first), and n0 made relative to it.
static __device__ __forceinline__ KqMember kq_member(const KqParams& p, kq_u32& n0)
{
#if H_BIAS
    KqMember m = { p.w, p.y, p.bias, (kq_u32)N, (kq_u32)Y_STRIDE };
#else
    KqMember m = { p.w, p.y, (kq_u32)N, (kq_u32)Y_STRIDE };
#endif
#if MULTI
    m.rows = MULTI_O1;
    if (n0 >= MULTI_O1)
    {
#if MULTI > 2
        const bool third = n0 >= MULTI_O2;
        m.w = third ? p.w_2 : p.w_1;
        m.y = third ? p.y_2 : p.y_1;
#if H_BIAS
        m.bias = third ? p.bias_2 : p.bias_1;
#endif
        m.rows = third ? N - MULTI_O2 : MULTI_O2 - MULTI_O1;
        n0 -= third ? MULTI_O2 : MULTI_O1;
#else
        m.w = p.w_1;
        m.y = p.y_1;
#if H_BIAS
        m.bias = p.bias_1;
#endif
        m.rows = N - MULTI_O1;
        n0 -= MULTI_O1;
#endif
    }
    m.stride = m.rows;
#endif
    return m;
}

// The first byte of tile row r's weight row (rows past the matrix clamped to its last) and the end of that row's matrix. The dual
// form interleaves its matrices: even tile rows are w's rows o0 + r / 2, odd ones w_1's.
static __device__ __forceinline__ const unsigned char* kq_row(const KqParams& p, const KqMember& m, kq_u32 n0, kq_u32 r, const unsigned char*& end)
{
#if KQ_DUAL
    const unsigned char* w = (r & 1) ? p.w_1 : p.w;
    end = w + (kq_u64)N * KQ_ROWB;
    return w + (kq_u64)min(n0 / 2 + (r >> 1), (kq_u32)N - 1) * KQ_ROWB;
#else
    end = m.w + (kq_u64)m.rows * KQ_ROWB;
    return m.w + (kq_u64)min(n0 + r, m.rows - 1) * KQ_ROWB;
#endif
}

#if KQ_FMT != 6
// The 6-bit scales and minimums of a Q4_K / Q5_K block's 12 scale bytes q (2-byte aligned), one byte each: out[0] = sc[0..3],
// out[1] = sc[4..7], out[2] = m[0..3], out[3] = m[4..7].
static __device__ __forceinline__ void kq_scales_mins(const unsigned char* q, kq_u32* out)
{
    const kq_u32 u0 = kq_ld16(q) | (kq_ld16(q + 2) << 16), u1 = kq_ld16(q + 4) | (kq_ld16(q + 6) << 16), u2 = kq_ld16(q + 8) | (kq_ld16(q + 10) << 16);
    out[0] = u0 & 0x3F3F3F3Fu;
    out[1] = (u2 & 0x0F0F0F0Fu) | (((u0 >> 6) & 0x03030303u) << 4);
    out[2] = u1 & 0x3F3F3F3Fu;
    out[3] = ((u2 >> 4) & 0x0F0F0F0Fu) | (((u1 >> 6) & 0x03030303u) << 4);
}
#endif

extern "C" __global__ void __launch_bounds__(KQ_THREADS) gemm_kq_native(KqParams p)
{
    extern __shared__ __align__(128) unsigned char smem[];
    const kq_u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const kq_u32 wm = 0, wn = warp;
    const kq_u32 g = lane >> 2, c4 = lane & 3;
    const kq_u32 t0 = blockIdx.x * KQ_BM, ng = blockIdx.y * KQ_BN;
    kq_u32 n0 = ng;
    const KqMember m = kq_member(p, n0);
    const kq_u32 sb0 = blockIdx.z * KQ_SB / KQ_SPLIT, sb1 = (blockIdx.z + 1) * KQ_SB / KQ_SPLIT, steps = sb1 - sb0;

    auto load = [&](kq_u32 s, kq_u32 sb) {
        const kq_u32 base = (kq_u32)__cvta_generic_to_shared(smem + s * KQ_STAGE_BYTES);
#pragma unroll
        for (kq_u32 i = 0; i < KQ_BM * 32 / KQ_THREADS; i++)
        {
            kq_u32 idx = tid + i * KQ_THREADS, r = idx >> 5, c = idx & 31;
            kq_cp16(base + (c >> 3) * (KQ_BM * 128) + kq_swz(r, c & 7), p.x + (kq_u64)min(t0 + r, p.T - 1) * KQ_XRS_BYTES + sb * 512 + c * 16, 16);
        }
        for (kq_u32 idx = tid; idx < KQ_BN * KQ_WCH; idx += KQ_THREADS)
        {
            kq_u32 r = idx / KQ_WCH, c = idx % KQ_WCH;
            const unsigned char* wend;
            const unsigned char* block = kq_row(p, m, n0, r, wend) + (kq_u64)sb * KQ_BB;
            const unsigned char* src = (const unsigned char*)((kq_u64)block & ~15ull) + c * 16;
            long long left = wend - src;
            kq_cp16(base + KQ_XBYTES + (r * KQ_WCH + c) * 16, left > 0 ? src : m.w, left >= 16 ? 16u : left > 0 ? (kq_u32)left : 0u);
        }
#if KQ_PF > 0
        if (tid < KQ_BN && sb + KQ_PF < sb1)
        {
            const unsigned char* wend;
            const unsigned char* block = kq_row(p, m, n0, tid, wend) + (kq_u64)(sb + KQ_PF) * KQ_BB;
            const unsigned char* src = (const unsigned char*)((kq_u64)block & ~15ull);
#if KQ_PFB
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(src), "r"((kq_u32)(KQ_WCH * 16)));
#else
#pragma unroll
            for (kq_u32 o = 0; o < KQ_WCH * 16; o += 64)
            {
                asm volatile("prefetch.global.L2 [%0];\n" :: "l"(src + o));
            }
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(src + KQ_WCH * 16 - 1));
#endif
        }
#endif
    };

    float acc[KQ_FM][KQ_FN][4];
#pragma unroll
    for (int i = 0; i < KQ_FM; i++)
    {
#pragma unroll
        for (int j = 0; j < KQ_FN; j++)
        {
#pragma unroll
            for (int e = 0; e < 4; e++)
            {
                acc[i][j][e] = 0.0f;
            }
        }
    }

#pragma unroll
    for (int s = 0; s < KQ_STAGES - 1; s++)
    {
        if (s < (int)steps)
        {
            load(s, sb0 + s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    for (kq_u32 step = 0; step < steps; step++)
    {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(KQ_STAGES - 2));
        __syncthreads();
        kq_u32 next = step + KQ_STAGES - 1;
        if (next < steps)
        {
            load(next % KQ_STAGES, sb0 + next);
        }
        asm volatile("cp.async.commit_group;\n" ::);

        const unsigned char* stage = smem + (step % KQ_STAGES) * KQ_STAGE_BYTES;
        const kq_u32 sa = (kq_u32)__cvta_generic_to_shared(stage);
        const unsigned char* blk[KQ_FN];
#pragma unroll
        for (int j = 0; j < KQ_FN; j++)
        {
            kq_u32 r = wn * 8 * KQ_FN + j * 8 + g;
            const unsigned char* wend;
            const unsigned char* block = kq_row(p, m, n0, r, wend) + (kq_u64)(sb0 + step) * KQ_BB;
            blk[j] = stage + KQ_XBYTES + r * KQ_WCH * 16 + ((kq_u64)block & 15ull);
        }

        // The A fragments of chunk q (16 values of the superblock) and the products with the B fragments b[j][0..1].
        auto mma = [&](int q, const kq_u32 (*b)[2]) {
#pragma unroll
            for (int i = 0; i < KQ_FM; i++)
            {
                kq_u32 a[4];
                kq_u32 r = wm * KQ_TM + i * 16 + (lane & 15);
                kq_ldsm4(sa + (q >> 2) * (KQ_BM * 128) + kq_swz(r, (q & 3) * 2 + (lane >> 4)), a[0], a[1], a[2], a[3]);
#pragma unroll
                for (int j = 0; j < KQ_FN; j++)
                {
                    kq_mma(acc[i][j], a, b[j][0], b[j][1]);
                }
            }
        };

#if KQ_FMT == 6
        float d[KQ_FN];
#pragma unroll
        for (int j = 0; j < KQ_FN; j++)
        {
            d[j] = kq_half(blk[j] + 208);
        }
#pragma unroll
        for (int jj = 0; jj < 2; jj++)
        {
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
                kq_u32 ql[KQ_FN][2][2], qh[KQ_FN][2];
#pragma unroll
                for (int j = 0; j < KQ_FN; j++)
                {
                    const kq_u32 l = 16 * h + 2 * c4;
#pragma unroll
                    for (int o = 0; o < 2; o++)
                    {
                        ql[j][0][o] = kq_spread(kq_ld16(blk[j] + 64 * jj + l + 8 * o));
                        ql[j][1][o] = kq_spread(kq_ld16(blk[j] + 64 * jj + 32 + l + 8 * o));
                        qh[j][o] = kq_spread(kq_ld16(blk[j] + 128 + 32 * jj + l + 8 * o));
                    }
                }
#pragma unroll
                for (int qq = 0; qq < 4; qq++)
                {
                    const int q = 8 * jj + 2 * qq + h;
                    kq_u32 b[KQ_FN][2];
#pragma unroll
                    for (int j = 0; j < KQ_FN; j++)
                    {
                        const kq_u32 scale = kq_splat(d[j] * (float)(signed char)blk[j][192 + q]);
#pragma unroll
                        for (int o = 0; o < 2; o++)
                        {
                            kq_u32 code = ((ql[j][qq & 1][o] >> (4 * (qq >> 1))) & 0x000F000Fu) | (((qh[j][o] >> (2 * qq)) & 0x00030003u) << 4);
                            b[j][o] = kq_mul2(kq_codes(code, 0x64206420u), scale);   // (q - 32) * d * sc
                        }
                    }
                    mma(q, b);
                }
            }
        }
#else
#if KQ_FMT == 5
        const kq_u32 qsOffset = 48;
#else
        const kq_u32 qsOffset = 16;
#endif
        // The eight 6-bit scales and minimums of each row's block, unpacked four to a word (sc[0..3], sc[4..7], m[0..3], m[4..7]).
        kq_u32 scales[KQ_FN][4];
        float dd[KQ_FN], dm[KQ_FN];
#pragma unroll
        for (int j = 0; j < KQ_FN; j++)
        {
            dd[j] = kq_half(blk[j]);
            dm[j] = kq_half(blk[j] + 2);
            kq_scales_mins(blk[j] + 4, scales[j]);
        }
#pragma unroll
        for (int jj = 0; jj < 4; jj++)
        {
            kq_u32 dsc[KQ_FN][2], dmn[KQ_FN][2];
#pragma unroll
            for (int j = 0; j < KQ_FN; j++)
            {
#pragma unroll
                for (int hi = 0; hi < 2; hi++)
                {
                    const int sub = 2 * jj + hi;
                    const kq_u32 sc = (scales[j][sub >> 2] >> (8 * (sub & 3))) & 0xFFu, mn = (scales[j][2 + (sub >> 2)] >> (8 * (sub & 3))) & 0xFFu;
                    dsc[j][hi] = kq_splat(dd[j] * (float)sc);
                    dmn[j][hi] = kq_splat(-dm[j] * (float)mn);
                }
            }
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
                kq_u32 qs[KQ_FN][2];
#if KQ_FMT == 5
                kq_u32 qh[KQ_FN][2];
#endif
#pragma unroll
                for (int j = 0; j < KQ_FN; j++)
                {
                    const kq_u32 l = 16 * h + 2 * c4;
#pragma unroll
                    for (int o = 0; o < 2; o++)
                    {
                        qs[j][o] = kq_spread(kq_ld16(blk[j] + qsOffset + 32 * jj + l + 8 * o));
#if KQ_FMT == 5
                        qh[j][o] = kq_spread(kq_ld16(blk[j] + 16 + l + 8 * o));
#endif
                    }
                }
#pragma unroll
                for (int hi = 0; hi < 2; hi++)
                {
                    kq_u32 b[KQ_FN][2];
#pragma unroll
                    for (int j = 0; j < KQ_FN; j++)
                    {
#pragma unroll
                        for (int o = 0; o < 2; o++)
                        {
                            kq_u32 code = (qs[j][o] >> (4 * hi)) & 0x000F000Fu;
#if KQ_FMT == 5
                            code |= ((qh[j][o] >> (2 * jj + hi)) & 0x00010001u) << 4;
#endif
                            b[j][o] = kq_fma2(kq_codes(code, 0x64006400u), dsc[j][hi], dmn[j][hi]);   // q * d * sc - dmin * m
                        }
                    }
                    mma(4 * jj + 2 * hi + h, b);
                }
            }
        }
#endif
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

#if KQ_SPLIT > 1
    __shared__ kq_u32 kqLast;
    const kq_u32 tile = blockIdx.y * gridDim.x + blockIdx.x;
#pragma unroll
    for (int i = 0; i < KQ_FM; i++)
    {
#pragma unroll
        for (int hrow = 0; hrow < 2; hrow++)
        {
            kq_u32 t = t0 + wm * KQ_TM + i * 16 + g + hrow * 8;
            if (t >= p.T)
            {
                continue;
            }

            float* pr = p.part + ((kq_u64)blockIdx.z * p.T + t) * KQ_PN;
#pragma unroll
            for (int j = 0; j < KQ_FN; j++)
            {
                kq_u32 n = ng + wn * 8 * KQ_FN + j * 8 + c4 * 2;
                if (n + 1 < KQ_PN)
                {
                    __stcg((float2*)(pr + n), make_float2(acc[i][j][hrow * 2], acc[i][j][hrow * 2 + 1]));
                }
                else if (n < KQ_PN)
                {
                    __stcg(pr + n, acc[i][j][hrow * 2]);
                }
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0)
    {
        kq_u32 arrived = atomicAdd(p.arrivals + tile, 1u);
        bool last = arrived + 1 == KQ_SPLIT;
        if (last)
        {
            p.arrivals[tile] = 0;
        }
        kqLast = last ? 1u : 0u;
    }
    __syncthreads();
    if (kqLast == 0)
    {
        return;
    }
    __threadfence();
#endif

#pragma unroll
    for (int i = 0; i < KQ_FM; i++)
    {
#pragma unroll
        for (int hrow = 0; hrow < 2; hrow++)
        {
            kq_u32 t = t0 + wm * KQ_TM + i * 16 + g + hrow * 8;
            if (t >= p.T)
            {
                continue;
            }

            float* yr = m.y + (kq_u64)t * m.stride;
#ifdef H_XRS
            const kq_u32 scaleWord = *(const kq_u32*)(p.x + H_XRS + (kq_u64)t * 4);
            const float rowInverse = scaleWord == 0 ? 1.0f : __uint_as_float(scaleWord);
#endif
#pragma unroll
            for (int j = 0; j < KQ_FN; j++)
            {
                kq_u32 n = n0 + wn * 8 * KQ_FN + j * 8 + c4 * 2;
#if KQ_SPLIT > 1
                const kq_u32 column = ng + wn * 8 * KQ_FN + j * 8 + c4 * 2;
                float v0 = 0.0f, v1 = 0.0f;
                for (kq_u32 z = 0; z < KQ_SPLIT; z++)
                {
                    const float* pr = p.part + ((kq_u64)z * p.T + t) * KQ_PN;
                    v0 += __ldcg(pr + min(column, (kq_u32)KQ_PN - 1));
                    v1 += __ldcg(pr + min(column + 1, (kq_u32)KQ_PN - 1));
                }
#else
                float v0 = acc[i][j][hrow * 2], v1 = acc[i][j][hrow * 2 + 1];
#endif
#ifdef H_XRS
                v0 *= rowInverse;
                v1 *= rowInverse;
#endif
#if H_BIAS
                v0 += m.bias[min(n, m.rows - 1)];
                v1 += m.bias[min(n + 1, m.rows - 1)];
#endif
#if KQ_DUAL
                // v0 is w's row n / 2, v1 w_1's: the pair's activation product.
                if (n / 2 < (kq_u32)N)
                {
                    yr[n / 2] = KQ_EPI2(v0, v1);
                }
#else
                if (n + 1 < m.rows)
                {
                    *(float2*)(yr + n) = make_float2(v0, v1);
                }
                else if (n < m.rows)
                {
                    yr[n] = v0;
                }
#endif
            }
        }
    }
}
