// Expert-grouped MXFP4 GEMM written in CUDA C++ for the CUDA provider (DeviceFeatures.NativeGemmMoe, compute capability 8.0 and
// later): the grouped form of a MatMulId with MXFP4R expert weights (WeightRepack.Mxfp4Split) for batches with many rows per
// expert. Each block computes one routing tile (moe_route in moe.slang: [0] = tile count, [1 + 3i ..] = tile i {expert, row
// offset, rows}, [1 + 3 MAXTILES + o] = item at sorted position o) against NM_BN weight rows of the tile's expert:
//
//   y[item(o), n] = sum_k x[o, k] * W[expert, n, k]      (x: the expert-sorted f16 rows of moe_gather_x16, X_STRIDE = K)
//
// cp.async copies the activations (64-value K steps, 128-byte rows, XOR swizzle) and the raw MXFP4 codes and E8M0 scales (4.25
// bits per weight) into a ring of NM_STAGES stages of dynamic shared memory; each warp decodes the codes it multiplies right before
// its MMAs (mma.sync.m16n8k16, f32 accumulation). The decode is the arithmetic of gemm_cm2's (matmul.slang: the doubled E2M1 code
// in half bits times the two half factors of the block scale) and the MMAs run in the same K order on the same 16-value chunks, so
// the products are those of the generated grouped GEMM on this device.
//
// DUAL: two expert matrices of one shape (w, w2: gate and up, or the two halves of a fused gate / up matrix, w2 = the upper half's
// first row) share the activations; the B tile's 8-row groups alternate between them and the store applies NM_EPI2(m0, m1, e, n)
// (generated: the expert group's elementwise epilogue, which may read per-expert rows of the epilogue tables). Single: the store
// applies NM_EPI1(m, e, n, t, s) when defined (t, s: token and slot of the item), else stores the product.
// X_RS: x is row-ranged (common.slang, f16_range_scale; the scale word of sorted row o at byte X_RS + 4 o): rows are multiplied by
// their inverse scale before the epilogue.
//
// Specialization macros: K, N (outputs), EXPERT_BYTES (bytes per expert of w), ROW_BYTES (17 K / 32), S, MAXTILES, NM_BM (32, 64 or
// 128 token rows), NM_BN (weight rows per tile; outputs NM_BN / 2 with DUAL), NM_WM / NM_WN (warps), NM_STAGES, NM_MINB, NM_TABLE, DUAL, X_RS,
// NM_EPI2 / NM_EPI1 and NM_EPI_FIELDS (the tables they read). Requires K % 64 == 0, N % 16 == 0. Grid: x = column tiles, y = routing
// tiles (blocks past the tile count return).
#ifndef DUAL
#define DUAL 0
#endif
#ifndef NM_BM
#define NM_BM 128
#endif
#ifndef NM_BN
#define NM_BN 256
#endif
#ifndef NM_WM
#define NM_WM (NM_BM / 64)
#endif
#ifndef NM_WN
#define NM_WN 4
#endif
#ifndef NM_STAGES
#define NM_STAGES 4
#endif
#define NM_THREADS (NM_WM * NM_WN * 32)
// NM_TABLE 1: the E8M0 factors come from a 256-entry shared table built per block (1 KB); 0: converted at each decode, which
// leaves room for a fourth stage of the 64-row dual tile at two blocks per SM.
#ifndef NM_TABLE
#define NM_TABLE 1
#endif
// Resident blocks per SM the register allocation is bounded for (__launch_bounds__): 2 for the short tiles whose stages fit twice.
#ifndef NM_MINB
#define NM_MINB 1
#endif
#define NM_TM (NM_BM / NM_WM)
#define NM_TN (NM_BN / NM_WN)
#define NM_FM (NM_TM / 16)
#define NM_FN (NM_TN / 8)
#if DUAL
#define NM_BOUT (NM_BN / 2)
#define NM_MATS 2
#else
#define NM_BOUT NM_BN
#define NM_MATS 1
#endif
// Stage: [NM_BM rows of 128 bytes][2 blocks][NM_BN rows of 16 code bytes][2 blocks][NM_MATS][NM_BOUT / 16 groups of 16 scales]
#define NM_A_BYTES (NM_BM * 128)
#define NM_C_BYTES (2 * NM_BN * 16)
#define NM_S_BYTES (2 * NM_MATS * (NM_BOUT / 16) * 16)
#define NM_STAGE_BYTES (NM_A_BYTES + NM_C_BYTES + NM_S_BYTES)
#define NM_XRS ((unsigned long long)K * 2)
#define NM_CODE_BYTES ((unsigned long long)K / 2)

typedef unsigned int nm_u32;
typedef unsigned long long nm_u64;

#if (NM_BOUT % 16) != 0 || (DUAL && (NM_FN % 2) != 0) || (NM_FM * 16 * NM_WM) != NM_BM || (NM_TN % 8) != 0
#error "gemm_moe_native: unsupported tile"
#endif

struct NmParams
{
    const unsigned char* x;
    const unsigned char* w;
#if DUAL
    const unsigned char* w2;
#endif
    const nm_u32* scratch;
    float* y;
#ifdef NM_EPI_FIELDS
    NM_EPI_FIELDS
#endif
    nm_u32 T;
};

static __device__ __forceinline__ void nm_cp16(nm_u32 dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}

static __device__ __forceinline__ void nm_ldsm4(nm_u32 addr, nm_u32& r0, nm_u32& r1, nm_u32& r2, nm_u32& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

static __device__ __forceinline__ uint4 nm_lds128(nm_u32 addr)
{
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

// The copy of weight codes and scales: with an L2 prefetch of the surrounding NM_PF bytes (a row's next K steps), so the weights
// stream from memory in whole bursts although a stage takes only 32 bytes of each row (and of each 16-row group's scale rows).
#ifndef NM_PF
#define NM_PF 256
#endif
static __device__ __forceinline__ void nm_cp16w(nm_u32 dst, const void* src)
{
#if NM_PF == 256
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
#elif NM_PF == 128
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
#else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
#endif
}

static __device__ __forceinline__ nm_u32 nm_lds8(nm_u32 addr)
{
    nm_u32 v;
    asm volatile("ld.shared.u8 %0, [%1];\n" : "=r"(v) : "r"(addr));
    return v;
}

static __device__ __forceinline__ void nm_mma(float* c, const nm_u32* a, nm_u32 b0, nm_u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ nm_u32 nm_prmt(nm_u32 a, nm_u32 b, nm_u32 sel)
{
    nm_u32 r;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

static __device__ __forceinline__ nm_u32 nm_hmul2(nm_u32 a, nm_u32 b)
{
    nm_u32 r;
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// The E2M1 codes in the high nibbles of bytes 1 and 3 of x as packed halves in the bits of fp4_h2 (matmul.slang): each code's
// magnitude at half bits 9..11 and its sign at bit 15 (the other bytes of x are ignored). One shift and one three-input logic
// operation per pair, where masking each code into the low bits first took five.
static __device__ __forceinline__ nm_u32 nm_e2m1_hi(nm_u32 x)
{
    return (x & 0x80008000u) | ((x >> 3) & 0x0E000E00u);
}

// The E2M1 codes in the low nibbles of bytes 1 and 3 of x, as nm_e2m1_hi.
static __device__ __forceinline__ nm_u32 nm_e2m1_lo(nm_u32 x)
{
    return ((x << 4) & 0x80008000u) | ((x << 1) & 0x0E000E00u);
}

// The low and the high half of v, each in both halves. Written as a half-pair move, which ptxas folds into the half-broadcast
// operands (.H0_H0 / .H1_H1) of the multiplications that use them instead of issuing two byte permutes.
static __device__ __forceinline__ void nm_halves(nm_u32 v, nm_u32& lo, nm_u32& hi)
{
    asm("{\n .reg .b16 l, h;\n mov.b32 {l, h}, %2;\n mov.b32 %0, {l, l};\n mov.b32 %1, {h, h};\n}\n" : "=r"(lo), "=r"(hi) : "r"(v));
}

// f32 to f16 bits, rounded to nearest even (the kernel is compiled without cuda_fp16.h).
static __device__ __forceinline__ nm_u32 nm_f2h(float v)
{
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;\n" : "=h"(h) : "f"(v));
    return h;
}

// The two half factors of an E8M0 scale byte (e8m0_half(e) * 32768 split at half's range, as matmul.slang), each in both halves.
static __device__ __forceinline__ void nm_scale(nm_u32 e, nm_u32& s1, nm_u32& s2)
{
    float s = __uint_as_float(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23)) * 32768.0f;
    nm_u32 a = nm_f2h(fminf(s, 32768.0f));
    nm_u32 b = nm_f2h(fmaxf(s / 32768.0f, 1.0f));
    s1 = a * 0x10001u;
    s2 = b * 0x10001u;
}

// Byte offset of 16-byte chunk c of row r in a 128-byte-row tile (chunk c ^ (r % 8)).
static __device__ __forceinline__ nm_u32 nm_swz(nm_u32 r, nm_u32 c)
{
    return r * 128u + ((c ^ (r & 7u)) << 4);
}

// The MMAs of the first F row fragments of a warp (rows r0 ..) for 32-value block b of a stage at sa, with the block's decoded B
// fragments of the warp's columns; dec(j) (the decode of unit j of the next block) is issued between them, so its shared-memory
// latency and arithmetic fill the tensor pipe's gaps instead of stalling both warps of a scheduler at once.
template <int F, typename Dec>
static __device__ __forceinline__ void nm_mmas(float (&acc)[NM_FM][NM_FN][4], nm_u32 sa, nm_u32 r0, nm_u32 b, nm_u32 lane,
                                               const nm_u32 (&bf)[NM_FN][4], Dec dec)
{
#pragma unroll
    for (int h = 0; h < 2; h++)
    {
#pragma unroll
        for (int i = 0; i < F; i++)
        {
            nm_u32 a[4];
            nm_u32 r = r0 + i * 16 + (lane & 15);
            nm_ldsm4(sa + nm_swz(r, b * 4 + h * 2 + (lane >> 4)), a[0], a[1], a[2], a[3]);
#pragma unroll
            for (int j = 0; j < NM_FN; j++)
            {
                nm_mma(acc[i][j], a, bf[j][h * 2], bf[j][h * 2 + 1]);
            }
            // Unit u of the next block after fragment u's MMAs (the first half), the rest after the last fragment.
            if (h == 0 && i < NM_FN)
            {
                dec(i);
            }
        }
        if (h == 0)
        {
#pragma unroll
            for (int j = F; j < NM_FN; j++)
            {
                dec(j);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(NM_THREADS, NM_MINB) gemm_moe_native(NmParams p)
{
    extern __shared__ __align__(128) unsigned char smem[];
#if NM_TABLE
    // E8M0 byte -> its two half factors (s1 in the low half, s2 in the high one), built once per block.
    __shared__ nm_u32 sScale[256];
#endif
    const nm_u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const nm_u32 wm = warp / NM_WN, wn = warp % NM_WN;
    const nm_u32 tile = blockIdx.y;
    if (tile >= p.scratch[0])
    {
        return;
    }

    const nm_u32 expert = p.scratch[1 + 3 * tile], rowOff = p.scratch[2 + 3 * tile], tileRows = p.scratch[3 + 3 * tile];
    const nm_u32 n0 = blockIdx.x * NM_BOUT;
#if NM_TABLE
    for (nm_u32 e = tid; e < 256; e += NM_THREADS)
    {
        nm_u32 s1, s2;
        nm_scale(e, s1, s2);
        sScale[e] = (s1 & 0xFFFFu) | (s2 << 16);
    }
#endif

    const unsigned char* wb = p.w + (nm_u64)expert * EXPERT_BYTES;
#if DUAL
    const unsigned char* wb2 = p.w2 + (nm_u64)expert * EXPERT_BYTES;
#endif
    const nm_u32 nk = K / 64;

    // Copies per thread and K step: activations (NM_BM rows x 8 chunks; rows past the tile are not copied: their products are
    // never stored), codes (NM_BN rows x 2 blocks, stored [block][row]) and scales (per block and matrix, the 16-byte scale rows
    // of the tile's 16-row groups). Source addresses advance by a constant per K step.
#define NM_A_PER ((NM_BM * 8 + NM_THREADS - 1) / NM_THREADS)
#define NM_C_PER ((NM_BN * 2 + NM_THREADS - 1) / NM_THREADS)
    const unsigned char* aSrc[NM_A_PER];
    nm_u32 aDst[NM_A_PER];
    bool aLive[NM_A_PER];
#pragma unroll
    for (nm_u32 i = 0; i < NM_A_PER; i++)
    {
        nm_u32 idx = tid + i * NM_THREADS, r = idx >> 3, c = idx & 7;
        aLive[i] = idx < NM_BM * 8 && r < tileRows;
        aSrc[i] = p.x + (nm_u64)(rowOff + (aLive[i] ? r : 0)) * NM_XRS + c * 16;
        aDst[i] = nm_swz(r, c);
    }
    const unsigned char* cSrc[NM_C_PER];
#pragma unroll
    for (nm_u32 i = 0; i < NM_C_PER; i++)
    {
        nm_u32 idx = min(tid + i * NM_THREADS, (nm_u32)NM_BN * 2 - 1), r = idx >> 1, b = idx & 1;
#if DUAL
        const unsigned char* m = ((r >> 3) & 1) ? wb2 : wb;
        nm_u32 n = min(n0 + (r >> 4) * 8 + (r & 7), (nm_u32)N - 1);
#else
        const unsigned char* m = wb;
        nm_u32 n = min(n0 + r, (nm_u32)N - 1);
#endif
        cSrc[i] = m + (nm_u64)(n >> 4) * 16 * ROW_BYTES + (nm_u64)(n & 15) * NM_CODE_BYTES + b * 16;
    }
    const bool sLive = tid < 2 * NM_MATS * (NM_BOUT / 16);
    const unsigned char* sSrc;
    {
        nm_u32 q = tid % (NM_BOUT / 16), m = tid / (NM_BOUT / 16) % NM_MATS, b = tid / (NM_BOUT / 16) / NM_MATS % 2;
#if DUAL
        const unsigned char* mat = m ? wb2 : wb;
#else
        const unsigned char* mat = wb;
#endif
        nm_u32 gi = min(n0 / 16 + q, (nm_u32)N / 16 - 1);
        sSrc = mat + (nm_u64)gi * 16 * ROW_BYTES + 16 * NM_CODE_BYTES + b * 16;
    }

    auto load = [&](nm_u32 s, nm_u32 kb) {
        nm_u32 base = (nm_u32)__cvta_generic_to_shared(smem + s * NM_STAGE_BYTES);
#pragma unroll
        for (nm_u32 i = 0; i < NM_A_PER; i++)
        {
            if (aLive[i])
            {
                nm_cp16(base + aDst[i], aSrc[i] + kb * 128);
            }
        }
#pragma unroll
        for (nm_u32 i = 0; i < NM_C_PER; i++)
        {
            if ((NM_BN * 2) % NM_THREADS == 0 || tid + i * NM_THREADS < NM_BN * 2)
            {
                nm_u32 idx = tid + i * NM_THREADS;
                nm_cp16w(base + NM_A_BYTES + ((idx & 1) * NM_BN + (idx >> 1)) * 16, cSrc[i] + kb * 32);
            }
        }
        if (sLive)
        {
            nm_cp16w(base + NM_A_BYTES + NM_C_BYTES + tid * 16, sSrc + kb * 32);
        }
    };

    float acc[NM_FM][NM_FN][4];
#pragma unroll
    for (int i = 0; i < NM_FM; i++)
    {
#pragma unroll
        for (int j = 0; j < NM_FN; j++)
        {
#pragma unroll
            for (int e = 0; e < 4; e++)
            {
                acc[i][j][e] = 0.0f;
            }
        }
    }

#pragma unroll
    for (int s = 0; s < NM_STAGES - 1; s++)
    {
        if (s < (int)nk)
        {
            load(s, s);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    const nm_u32 g = lane >> 2, c4 = lane & 3;
    // The thread's code bytes 2c, 2c + 1, 2c + 8, 2c + 9 of a 16-byte block (values 2c.. of both 16-value halves, the B fragment
    // layout of m16n8k16): words (c & 2) / 2 and 2 + (c & 2) / 2, byte pair c & 1 of each.
    const nm_u32 sel = (c4 & 1) ? 0x7632u : 0x5410u;
    const bool upper = (c4 & 2) != 0;
    // 16-row fragments of the warp that hold tile rows (a partial tile's MMAs past its rows are skipped).
    const int liveFrags = (int)min((nm_u32)NM_FM, (tileRows > wm * NM_TM ? tileRows - wm * NM_TM + 15 : 0) / 16);
    const nm_u32 smem0 = (nm_u32)__cvta_generic_to_shared(smem);

    // Unit j of the B fragments of 32-value block b of stage s (two 16-value chunks of 8 weight rows), decoded once for all the
    // warp's row fragments.
    auto decodeUnit = [&](nm_u32 s, nm_u32 b, int j, nm_u32 (&bf)[NM_FN][4]) {
        const nm_u32 sc = smem0 + s * NM_STAGE_BYTES + NM_A_BYTES, ss = sc + NM_C_BYTES;
        nm_u32 row = wn * NM_TN + j * 8 + g;
        uint4 raw = nm_lds128(sc + (b * NM_BN + row) * 16);
        nm_u32 u = nm_prmt(upper ? raw.y : raw.x, upper ? raw.w : raw.z, sel);
#if DUAL
        nm_u32 f = row >> 3, m = f & 1, orow = (f >> 1) * 8 + g;
#else
        nm_u32 m = 0, orow = row;
#endif
        nm_u32 e8 = nm_lds8(ss + ((b * NM_MATS + m) * (NM_BOUT / 16) + orow / 16) * 16 + orow % 16);
        nm_u32 s1, s2;
#if NM_TABLE
        nm_halves(sScale[e8], s1, s2);
#else
        nm_scale(e8, s1, s2);
#endif
        // Code bytes 2c, 2c + 1 (and 2c + 8, 2c + 9) into bytes 1 and 3, one per half; the low nibbles are the first 16-value chunk.
        nm_u32 x01 = nm_prmt(u, 0u, 0x1404u), x23 = nm_prmt(u, 0u, 0x3424u);
        bf[j][0] = nm_hmul2(nm_e2m1_lo(x01), s1);
        bf[j][1] = nm_hmul2(nm_e2m1_lo(x23), s1);
        bf[j][2] = nm_hmul2(nm_e2m1_hi(x01), s1);
        bf[j][3] = nm_hmul2(nm_e2m1_hi(x23), s1);
        // s2 is exactly 1 up to E8M0 127 (a factor of 1): the second multiplication only above, so the halves are those of the two
        // multiplications. The branch is uniform where a block's scales are (gpt-oss's are all below 128); the decode is most of a
        // decode-sized step's instructions, and the step runs at the power limit (RTX PRO 4500, 200 W, ~1.8 GHz).
        if (e8 > 127u)
        {
#pragma unroll
            for (int q = 0; q < 4; q++)
            {
                bf[j][q] = nm_hmul2(bf[j][q], s2);
            }
        }
    };
    auto decode = [&](nm_u32 s, nm_u32 b, nm_u32 (&bf)[NM_FN][4]) {
#pragma unroll
        for (int j = 0; j < NM_FN; j++)
        {
            decodeUnit(s, b, j, bf);
        }
    };
    // The MMAs of 32-value block b of stage s with its decoded B fragments, for the warp's live row fragments only: a partial tile's
    // fragments past its rows are skipped by a warp-uniform branch (predicated-off MMAs still take the tensor pipe's time).
    auto mmas = [&](nm_u32 s, nm_u32 b, const nm_u32 (&bf)[NM_FN][4], auto dec) {
        const nm_u32 sa = smem0 + s * NM_STAGE_BYTES;
        switch (liveFrags)
        {
#define NM_CASE(F) case F: nm_mmas<F>(acc, sa, wm * NM_TM, b, lane, bf, dec); break;
        NM_CASE(1)
#if NM_FM > 1
        NM_CASE(2)
#endif
#if NM_FM > 2
        NM_CASE(3)
        NM_CASE(4)
#endif
#if NM_FM > 4
        NM_CASE(5)
        NM_CASE(6)
        NM_CASE(7)
        NM_CASE(8)
#endif
#undef NM_CASE
        default:
#pragma unroll
            for (int j = 0; j < NM_FN; j++)
            {
                dec(j);
            }
            break;
        }
    };

    // Software pipeline: the copies wait for two stages ahead (stage kb + 1 is complete at step kb), so the decode of the next
    // block (block 1 of this stage, then block 0 of the next one) is issued before the MMAs of the current block and its shared
    // memory latency hides under them.
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NM_STAGES - 3));
    __syncthreads();
    nm_u32 bf0[NM_FN][4], bf1[NM_FN][4];
    decode(0, 0, bf0);
    for (nm_u32 kb = 0; kb < nk; kb++)
    {
        const nm_u32 cur = kb % NM_STAGES;
        mmas(cur, 0, bf0, [&](int j) { decodeUnit(cur, 1, j, bf1); });
        // Stage kb + 1 complete (own copies, then every thread's) before its decode; the slot of stage kb - 1 is free for
        // the copies of stage kb + NM_STAGES - 1.
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NM_STAGES - 3));
        __syncthreads();
        nm_u32 next = kb + NM_STAGES - 1;
        if (next < nk)
        {
            load(next % NM_STAGES, next);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        const nm_u32 nextStage = (kb + 1) % NM_STAGES;
        const bool more = kb + 1 < nk;
        mmas(cur, 1, bf1, [&](int j) { if (more) { decodeUnit(nextStage, 0, j, bf0); } });
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

#pragma unroll
    for (int i = 0; i < NM_FM; i++)
    {
#pragma unroll
        for (int h = 0; h < 2; h++)
        {
            nm_u32 t = wm * NM_TM + i * 16 + g + h * 8;
            if (t >= tileRows)
            {
                continue;
            }

            nm_u32 item = p.scratch[1 + 3 * MAXTILES + rowOff + t];
            float* yr = p.y + (nm_u64)item * N;
#ifdef X_RS
            // f16_range_inverse (common.slang): 0 marks an unscaled row, else the word is the inverse factor.
            nm_u32 word = *(const nm_u32*)(p.x + X_RS + (nm_u64)(rowOff + t) * 4);
            float inv = word == 0u ? 1.0f : __uint_as_float(word);
#else
            float inv = 1.0f;
#endif
#if DUAL
#pragma unroll
            for (int j = 0; j < NM_FN; j += 2)
            {
                nm_u32 n = n0 + (wn * NM_TN) / 2 + (j / 2) * 8 + c4 * 2;
                if (n >= N)
                {
                    continue;
                }

                float v0 = NM_EPI2(acc[i][j][h * 2] * inv, acc[i][j + 1][h * 2] * inv, expert, n);
                float v1 = NM_EPI2(acc[i][j][h * 2 + 1] * inv, acc[i][j + 1][h * 2 + 1] * inv, expert, n + 1);
#else
#pragma unroll
            for (int j = 0; j < NM_FN; j++)
            {
                nm_u32 n = n0 + wn * NM_TN + j * 8 + c4 * 2;
                if (n >= N)
                {
                    continue;
                }

#ifdef NM_EPI1
                float v0 = NM_EPI1(acc[i][j][h * 2] * inv, expert, n, item / S, item % S);
                float v1 = NM_EPI1(acc[i][j][h * 2 + 1] * inv, expert, n + 1, item / S, item % S);
#else
                float v0 = acc[i][j][h * 2] * inv, v1 = acc[i][j][h * 2 + 1] * inv;
#endif
#endif
                *(float2*)(yr + n) = make_float2(v0, v1);
            }
        }
    }
}
