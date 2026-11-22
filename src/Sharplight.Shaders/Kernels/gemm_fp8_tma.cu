// FP8 GEMM written in CUDA C++ for the CUDA provider with tensor copies (DeviceFeatures.NativeGemmFp8 and TensorCopies, compute
// capability 9.0 and later): y[t, n] = sum_k x[t, k] * W[n, k] with E4M3 activations and weights in 128-value scale groups along K,
// on the operands of gemm_fp8_native.cu (x: quantize_fp8 rows followed by the dense scale copy; W, W2: the 128-row groups of
// WeightRepack.Fp8B128Grouped). Persistent and warp-specialized:
//  * gridDim.x blocks (at most one per SM) walk the 128 x 128 output tiles blockIdx.x, + gridDim.x, ... in the grouped order of
//    gemm_fp8_native.cu (groups of 16 token tiles walk all weight tiles), so the copies of a tile's first K blocks overlap the
//    previous tile's last blocks and its stores.
//  * A producer warp (warp 8, one lane) issues the copies: per 128-value K block one tensor copy of the 128 activation rows and one
//    (DUAL: two of 64 rows) of the weight rows, 128-byte swizzled by the copy engine into a three-stage ring of 32 KB, and bulk
//    copies of the block's 128 activation and 128 weight scales into a two-slot ring. Full barriers (transaction counts) tell the
//    consumers a stage has landed; empty barriers (one arrival per consumer warp) tell the producer a stage is free. No block-wide
//    barrier in the main loop: the consumer warps drift freely within the ring's slack.
//  * Eight consumer warps (2 x 4) compute the tile as gemm_fp8_native.cu does: per K block the B fragments once, then per 16-row
//    fragment the block's MMAs in a fresh accumulator, added with one fused multiply-add scaled by sx * sw. The results equal
//    gemm_fp8_native.cu's bit for bit.
//
// The tensor maps (CUtensorMap, 128 bytes each) follow the pointers and the row count in the parameter struct; the provider encodes
// them from the header line "sharplight-tensormaps" that the lowering adds (unsigned 8-bit elements, 128-byte swizzle):
//  * mx over x: [XCAP rows][K bytes] with row stride K + K / 32, box 128 x 128;
//  * mw (and mw2) over W: [N / 128 groups][128 rows][K bytes] with strides K and 128 (K + K / 32), box 128 bytes x 128 rows (DUAL: 64).
// DUAL: the B tile holds 64 rows of w, then the same 64 rows of w2; fragment j of a warp column reads w (j even) or w2 (j odd), so
// each thread holds both results of an output and stores F8N_EPI2(a, b).
//
// F8N_BLOCK_SCALED (DeviceFeatures.BlockScaledFp8, compute capability 12.x, compiled for the family-specific architecture
// compute_XYf, SL_FAMILY_ARCH): every scale must be a power of two (the activation rows of quantize_fp8 / the E4M3-writing norms with
// E4M3_UE8M0, the weights of WeightRepack.Fp8B128GroupedUe8m0). The consumer warps pass the scales' exponent fields as UE8M0 scale
// vectors to the block-scaled MMA (mma.sync ... kind::mxf8f6f4.block_scale.scale_vec::1X: one scale per 32 values of K for each row of
// A and each column of B; ours repeat per 128), which applies them in the tensor core, so the products of the whole K range accumulate
// in one f32 accumulator per output: no fresh accumulator and no multiply-add promotion per K block.
//
// QOUT (DUAL only): instead of f32 outputs, y points to E4M3 activation rows of the outputs as quantize_fp8 writes them
// ([QCAP rows of N codes + N / 32 bytes of f32 scales][N / 128][QCAP] dense scales), the next FP8 GEMM's x: the codes and scales
// quantize_fp8 would compute from the f32 outputs, bit for bit (amax / 448, or with F8N_BLOCK_SCALED its power-of-two rounding as
// E4M3_UE8M0 gives it). A 128-value scale group spans two 64-output tiles, so a block walks pairs of tiles (n0, n0 + 64) of one token
// tile back to back: the first tile's outputs stay in registers, each row's absolute maximum is combined in shared memory (two
// barriers of the computing warps per pair), and the second tile stores both. The lowering uses it with F8N_BLOCK_SCALED or F8N_WG:
// with the promotion's accumulators the 32 held outputs exceed the 168 registers of a 288-thread block and spill.
//
// F8N_WG (DeviceFeatures.RegisterReallocation, setmaxnreg, compiled for compute_XYf with SL_FAMILY_ARCH): the copy warp is a whole
// warpgroup (warps 8-11, 384 threads; one lane issues the copies, the others exit) that gives up its registers (setmaxnreg.dec to
// F8N_COPY_REGS) to the computing warps (setmaxnreg.inc to F8N_MMA_REGS): 232 registers instead of 168, since a block's registers are
// split by the warps per scheduler (9 warps take 3 slots on one, like 12). The budget must balance what the block was given at launch
// (384 x 168 = 128 x 40 + 256 x 232), or the increase waits for registers that are never released. Same results bit for bit.
//
// F8N_WUNI: the weight scales are uniform over each 128-row group per K block (128 x 128 block checkpoints): the promotion multiplies
// by one product of the row's activation scale and the group's weight scale (the same product as the per-row form, bit for bit).
//
// MULTI (2 or 3, not with DUAL or QOUT): the multi-output form over the concatenated rows of two or three matrices of one K (q / k / v),
// as gemm_fp8_native.cu's: rows [0, MULTI_O1) are w's into y, [MULTI_O1, MULTI_O2) w_1's into y_1, the rest w_2's into y_2 (dense rows
// of each member's width; N all rows, MULTI_O1 and MULTI_O2 multiples of 128), each with its own tensor map (mw_1, mw_2): a tile lies
// in one member, so each output is the product the member's own launch computes.
//
// Specialization macros: K, N, Y_STRIDE, XCAP, DUAL, F8N_EPI2 (DUAL), F8N_BLOCK_SCALED, F8N_WG, F8N_WUNI, MULTI, MULTI_O1, MULTI_O2, QOUT and QCAP. Requires K % 512 == 0, N % 128 == 0, XCAP % 128 == 0 and an
// even Y_STRIDE. Grid: x = persistent blocks (at most the number of tiles), 288 threads (F8N_WG: 384), F8N_SHARED bytes of dynamic shared memory.
#ifndef DUAL
#define DUAL 0
#endif
#ifndef F8N_BLOCK_SCALED
#define F8N_BLOCK_SCALED 0
#endif
#ifndef F8N_WUNI
#define F8N_WUNI 0
#endif
#ifndef MULTI
#define MULTI 0
#endif
#if MULTI && (DUAL || QOUT)
#error "gemm_fp8_tma: MULTI with DUAL or QOUT"
#endif
#define F8N_BM 128
#define F8N_BN 128
#define F8N_STAGES 3
#define F8N_WM 2
#define F8N_WN 4
#define F8N_CONSUMERS (F8N_WM * F8N_WN * 32)
#ifndef F8N_WG
#define F8N_WG 0
#endif
#if F8N_WG
// A copy warpgroup (warps 8-11) gives its registers to the computing warps (setmaxnreg).
#define F8N_THREADS (F8N_CONSUMERS + 128)
#define F8N_COPY_REGS 40
#define F8N_MMA_REGS 232
#else
#define F8N_THREADS (F8N_CONSUMERS + 32)
#endif
#define F8N_TM (F8N_BM / F8N_WM)
#define F8N_TN (F8N_BN / F8N_WN)
#define F8N_FM (F8N_TM / 16)
#define F8N_FN (F8N_TN / 8)
#define F8N_STAGE_BYTES ((F8N_BM + F8N_BN) * 128)
#define F8N_SCALE_BYTES ((F8N_BM + F8N_BN) * 4)
// Three code stages, two scale slots, eight barriers (full[3], empty[3], scales[2]).
#ifndef QOUT
#define QOUT 0
#endif
#if QOUT
// The rows' absolute maxima of a tile pair (128 floats) after the barriers.
#define F8N_SHARED (F8N_STAGES * F8N_STAGE_BYTES + 2 * F8N_SCALE_BYTES + 64 + 512)
// A tile pair covers 128 outputs: the walk's unit.
#define F8N_WALK 128
#define F8N_QRS ((unsigned long long)N + N / 32)
#else
#define F8N_SHARED (F8N_STAGES * F8N_STAGE_BYTES + 2 * F8N_SCALE_BYTES + 64)
#define F8N_WALK F8N_BOUT
#endif
#define F8N_RS ((unsigned long long)K + K / 32)
#define F8N_GROUP_BYTES (128ull * F8N_RS)
#define F8N_TILE_ORDER 16
#if DUAL
#define F8N_BOUT (F8N_BN / 2)
#else
#define F8N_BOUT F8N_BN
#endif

typedef unsigned char f8n_u8;
typedef unsigned int f8n_u32;
typedef unsigned long long f8n_u64;

struct __align__(64) F8nTensorMap
{
    f8n_u64 opaque[16];
};

struct F8nParams
{
    const f8n_u8* x;
    const f8n_u8* w;
#if DUAL
    const f8n_u8* w2;
#endif
    float* y;
#if MULTI
    const f8n_u8* w_1;
#if MULTI > 2
    const f8n_u8* w_2;
#endif
    float* y_1;
#if MULTI > 2
    float* y_2;
#endif
#endif
    f8n_u32 T;
    F8nTensorMap mx;
    F8nTensorMap mw;
#if DUAL
    F8nTensorMap mw2;
#endif
#if MULTI
    F8nTensorMap mw_1;
#if MULTI > 2
    F8nTensorMap mw_2;
#endif
#endif
};

// The member of a MULTI launch whose rows hold output column n0 of the concatenation: its index, first column and width.
static __device__ __forceinline__ f8n_u32 f8n_member(f8n_u32 n0, f8n_u32& first, f8n_u32& width)
{
#if MULTI
    if (n0 >= MULTI_O1)
    {
#if MULTI > 2
        if (n0 >= MULTI_O2)
        {
            first = MULTI_O2;
            width = N - MULTI_O2;
            return 2;
        }

        first = MULTI_O1;
        width = MULTI_O2 - MULTI_O1;
        return 1;
#else
        first = MULTI_O1;
        width = N - MULTI_O1;
        return 1;
#endif
    }

    first = 0;
    width = MULTI_O1;
    return 0;
#else
    first = 0;
    width = N;
    return 0;
#endif
}

static __device__ __forceinline__ f8n_u32 f8n_smem(const void* p)
{
    return (f8n_u32)__cvta_generic_to_shared(p);
}

static __device__ __forceinline__ void f8n_ldsm4(f8n_u32 addr, f8n_u32& r0, f8n_u32& r1, f8n_u32& r2, f8n_u32& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

#if F8N_BLOCK_SCALED
// c += (A * 2^(sa - 127)) (B * 2^(sb - 127)): the block-scaled E4M3 MMA of sm_120 (kind::mxf8f6f4, one UE8M0 scale per 32 values
// of K for each row of A and each column of B; scale_vec::1X, byte 0 of the scale registers, thread pairs 0 / 1 of each quad).
static __device__ __forceinline__ void f8n_mma(float* c, const f8n_u32* a, f8n_u32 b0, f8n_u32 b1, f8n_u32 sa, f8n_u32 sb)
{
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "r"(sa), "h"((unsigned short)0), "h"((unsigned short)0), "r"(sb), "h"((unsigned short)0), "h"((unsigned short)0));
}
#else
static __device__ __forceinline__ void f8n_mma(float* c, const f8n_u32* a, f8n_u32 b0, f8n_u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#endif

static __device__ __forceinline__ void f8n_mbar_init(f8n_u32 bar, f8n_u32 count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count));
}

static __device__ __forceinline__ void f8n_mbar_expect(f8n_u32 bar, f8n_u32 bytes)
{
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void f8n_mbar_arrive(f8n_u32 bar)
{
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}

static __device__ __forceinline__ void f8n_mbar_wait(f8n_u32 bar, f8n_u32 parity)
{
    asm volatile("{\n .reg .pred p;\n F8N_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra F8N_WAIT_%=;\n}\n"
                 :: "r"(bar), "r"(parity) : "memory");
}

// Destination space of the bulk and tensor copies: the block's own shared memory (PTX 8.6, CUDA 12.8). The cluster form that older
// compilers take makes ptxas check at run time that the address is the block's, with a fallback call that needs local memory.
#if __CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)
#define F8N_DST "shared::cta"
#else
#define F8N_DST "shared::cluster"
#endif

static __device__ __forceinline__ void f8n_tma2(f8n_u32 dst, const F8nTensorMap* map, int c0, int c1, f8n_u32 bar)
{
    asm volatile("cp.async.bulk.tensor.2d." F8N_DST ".global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 :: "r"(dst), "l"((f8n_u64)map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

static __device__ __forceinline__ void f8n_tma3(f8n_u32 dst, const F8nTensorMap* map, int c0, int c1, int c2, f8n_u32 bar)
{
    asm volatile("cp.async.bulk.tensor.3d." F8N_DST ".global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
                 :: "r"(dst), "l"((f8n_u64)map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}

static __device__ __forceinline__ void f8n_bulk(f8n_u32 dst, const void* src, f8n_u32 bytes, f8n_u32 bar)
{
    asm volatile("{\n .reg .u64 g;\n cvta.to.global.u64 g, %1;\n cp.async.bulk." F8N_DST ".global.mbarrier::complete_tx::bytes [%0], [g], %2, [%3];\n}\n"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// Byte offset of 16-byte chunk c of row r in a stage tile of 128-byte rows, as the 128-byte swizzle of the tensor copies places it
// (chunk c ^ (r % 8); the tiles are 1024-byte aligned).
static __device__ __forceinline__ f8n_u32 f8n_swz(f8n_u32 r, f8n_u32 c)
{
    return r * 128u + ((c ^ (r & 7u)) << 4);
}

// Scales of weight row n at K block 0 (the 128 rows of a group are adjacent per block, blocks 512 bytes apart).
static __device__ __forceinline__ const f8n_u8* f8n_wscales(const f8n_u8* w, f8n_u32 n)
{
    return w + (n >> 7) * F8N_GROUP_BYTES + 128ull * K + (n & 127) * 4u;
}

// Token-tile and weight-tile origin of tile id (groups of F8N_TILE_ORDER token tiles walk all weight tiles).
static __device__ __forceinline__ void f8n_tile(f8n_u32 id, f8n_u32 tilesX, f8n_u32& t0, f8n_u32& n0)
{
    const f8n_u32 tilesY = N / F8N_WALK, groupBlocks = F8N_TILE_ORDER * tilesY;
    const f8n_u32 first = id / groupBlocks * F8N_TILE_ORDER, groupRows = min(tilesX - first, (f8n_u32)F8N_TILE_ORDER);
    t0 = (first + id % groupBlocks % groupRows) * F8N_BM;
    n0 = id % groupBlocks / groupRows * F8N_WALK;
}

// Origin of this block's local tile v: tile blockIdx.x + v gridDim.x of the walk, or (QOUT) half v % 2 of tile pair v / 2.
static __device__ __forceinline__ void f8n_local(f8n_u32 v, f8n_u32 tilesX, f8n_u32& t0, f8n_u32& n0)
{
#if QOUT
    f8n_tile(blockIdx.x + (v >> 1) * gridDim.x, tilesX, t0, n0);
    n0 += (v & 1) * F8N_BOUT;
#else
    f8n_tile(blockIdx.x + v * gridDim.x, tilesX, t0, n0);
#endif
}

#if QOUT
// Scale of a 128-value group with absolute maximum amax as e4m3_group_scale computes it (e4m3.slang): amax / 448, or with
// F8N_BLOCK_SCALED (E4M3_UE8M0) 2^ceil(log2(amax / 448)) with 1 for an all-zero group.
static __device__ __forceinline__ float f8n_qscale(float amax)
{
#if F8N_BLOCK_SCALED
    if (!(amax > 0.0f))
    {
        return 1.0f;
    }

    const f8n_u32 bits = __float_as_uint(amax / 448.0f);
    const f8n_u32 e = ((bits >> 23) & 0xFFu) + ((bits & 0x7FFFFFu) != 0 ? 1u : 0u);
    return __uint_as_float(min(max(e, 1u), 254u) << 23);
#else
    return amax / 448.0f;
#endif
}

// e4m3_code of e4m3.slang: the E4M3 code of |v| <= 448 (subnormals below 2^-6 rounded as round() does there), saturated at 448.
static __device__ __forceinline__ f8n_u32 f8n_e4m3(float v)
{
    const f8n_u32 sign = v < 0.0f ? 0x80u : 0u;
    const float a = fabsf(v);
    f8n_u32 code;
    if (a < 0.015625f)
    {
        code = (f8n_u32)roundf(a * 512.0f);
    }
    else
    {
        f8n_u32 bits = __float_as_uint(a);
        bits += 0x7FFFFu + ((bits >> 20) & 1u);
        code = min((bits >> 20) - (120u << 3), 0x7Eu);
    }

    return sign | code;
}

// The codes of a and b (a in the low byte).
static __device__ __forceinline__ unsigned short f8n_e4m3x2(float a, float b)
{
    return (unsigned short)(f8n_e4m3(a) | (f8n_e4m3(b) << 8));
}
#endif

// B tile row of the first row of fragment j of warp column wn.
static __device__ __forceinline__ f8n_u32 f8n_brow(f8n_u32 wn, f8n_u32 j)
{
#if DUAL
    return (j & 1) * (F8N_BN / 2) + wn * (F8N_TN / 2) + (j >> 1) * 8;
#else
    return wn * F8N_TN + j * 8;
#endif
}

extern "C" __global__ void __launch_bounds__(F8N_THREADS, 1) gemm_fp8_tma(const __grid_constant__ F8nParams p)
{
    extern __shared__ __align__(1024) f8n_u8 smem[];
    const f8n_u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const f8n_u32 tilesX = (p.T + F8N_BM - 1) / F8N_BM, tiles = tilesX * (N / F8N_WALK);
    const f8n_u32 nk = K / 128;
    const f8n_u32 myTiles = (blockIdx.x < tiles ? (tiles - blockIdx.x + gridDim.x - 1) / gridDim.x : 0) * (QOUT ? 2 : 1);
    const f8n_u32 total = myTiles * nk;
    const f8n_u8* sx = p.x + (f8n_u64)XCAP * F8N_RS;
    const f8n_u32 scaleRing = f8n_smem(smem + F8N_STAGES * F8N_STAGE_BYTES);
    // full[s] at bars + 8 s, empty[s] at bars + 24 + 8 s, scales[s] at bars + 48 + 8 s.
    const f8n_u32 bars = scaleRing + 2 * F8N_SCALE_BYTES;
    if (tid == 0)
    {
        for (int s = 0; s < F8N_STAGES; s++)
        {
            f8n_mbar_init(bars + s * 8, 1);
            f8n_mbar_init(bars + 24 + s * 8, F8N_CONSUMERS / 32);
        }
        f8n_mbar_init(bars + 48, 1);
        f8n_mbar_init(bars + 56, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
#if QOUT
    float* rowMax = (float*)(smem + F8N_STAGES * F8N_STAGE_BYTES + 2 * F8N_SCALE_BYTES + 64);
    if (tid < F8N_BM)
    {
        rowMax[tid] = 0.0f;
    }
#endif
    __syncthreads();

    if (warp >= F8N_WM * F8N_WN)
    {
#if F8N_WG
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(F8N_COPY_REGS));
#endif
        if (warp != F8N_WM * F8N_WN || lane != 0)
        {
            return;
        }

        // Producer: the codes of iteration m once iteration m - 3 released its stage, with them the scales of iteration m - 1
        // (their slot was last read in iteration m - 3 as well).
        f8n_u32 ct0 = 0, cn0 = 0, codeLt = 0, codeKb = 0, st0 = 0, sn0 = 0, scaleLt = 0, scaleKb = 0;
        for (f8n_u32 m = 0; m <= total; m++)
        {
            if (m >= F8N_STAGES)
            {
                f8n_mbar_wait(bars + 24 + (m % F8N_STAGES) * 8, ((m / F8N_STAGES) - 1) & 1);
            }

            if (m < total)
            {
                if (codeKb == 0)
                {
                    f8n_local(codeLt, tilesX, ct0, cn0);
                }

                const f8n_u32 s = m % F8N_STAGES, dst = f8n_smem(smem + s * F8N_STAGE_BYTES), bar = bars + s * 8;
                f8n_mbar_expect(bar, F8N_STAGE_BYTES);
                f8n_tma2(dst, &p.mx, codeKb * 128, ct0, bar);
#if DUAL
                f8n_tma3(dst + F8N_BM * 128, &p.mw, codeKb * 128, cn0 & 127, cn0 >> 7, bar);
                f8n_tma3(dst + (F8N_BM + F8N_BN / 2) * 128, &p.mw2, codeKb * 128, cn0 & 127, cn0 >> 7, bar);
#elif MULTI
                {
                    f8n_u32 first, width;
                    const f8n_u32 mi = f8n_member(cn0, first, width);
                    const F8nTensorMap* map = mi == 0 ? &p.mw : &p.mw_1;
#if MULTI > 2
                    map = mi == 2 ? &p.mw_2 : map;
#endif
                    f8n_tma3(dst + F8N_BM * 128, map, codeKb * 128, 0, (cn0 - first) >> 7, bar);
                }
#else
                f8n_tma3(dst + F8N_BM * 128, &p.mw, codeKb * 128, 0, cn0 >> 7, bar);
#endif
                if (++codeKb == nk)
                {
                    codeKb = 0;
                    codeLt++;
                }
            }

            if (m >= 1)
            {
                if (scaleKb == 0)
                {
                    f8n_local(scaleLt, tilesX, st0, sn0);
                }

                const f8n_u32 slot = (m - 1) & 1, dst = scaleRing + slot * F8N_SCALE_BYTES, bar = bars + 48 + slot * 8;
                f8n_mbar_expect(bar, F8N_SCALE_BYTES);
                f8n_bulk(dst, sx + ((f8n_u64)scaleKb * XCAP + st0) * 4, F8N_BM * 4, bar);
#if DUAL
                f8n_bulk(dst + F8N_BM * 4, f8n_wscales(p.w, sn0) + scaleKb * 512, F8N_BN * 2, bar);
                f8n_bulk(dst + F8N_BM * 4 + F8N_BN * 2, f8n_wscales(p.w2, sn0) + scaleKb * 512, F8N_BN * 2, bar);
#elif MULTI
                {
                    f8n_u32 first, width;
                    const f8n_u32 mi = f8n_member(sn0, first, width);
                    const f8n_u8* wp = mi == 0 ? p.w : p.w_1;
#if MULTI > 2
                    wp = mi == 2 ? p.w_2 : wp;
#endif
                    f8n_bulk(dst + F8N_BM * 4, f8n_wscales(wp, sn0 - first) + scaleKb * 512, F8N_BN * 4, bar);
                }
#else
                f8n_bulk(dst + F8N_BM * 4, f8n_wscales(p.w, sn0) + scaleKb * 512, F8N_BN * 4, bar);
#endif
                if (++scaleKb == nk)
                {
                    scaleKb = 0;
                    scaleLt++;
                }
            }
        }

        return;
    }

#if F8N_WG
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(F8N_MMA_REGS));
#endif
    const f8n_u32 wm = warp / F8N_WN, wn = warp % F8N_WN;
    const f8n_u32 g = lane >> 2, c4 = lane & 3;
    float acc[F8N_FM][F8N_FN][4];
#pragma unroll
    for (int i = 0; i < F8N_FM; i++)
    {
#pragma unroll
        for (int j = 0; j < F8N_FN; j++)
        {
#pragma unroll
            for (int e = 0; e < 4; e++)
            {
                acc[i][j][e] = 0.0f;
            }
        }
    }

#if QOUT
    // The first tile's outputs of the current pair.
    float held[F8N_FM][2][F8N_FN / 2][2];
#endif
    for (f8n_u32 lt = 0; lt < myTiles; lt++)
    {
        f8n_u32 t0, n0;
        f8n_local(lt, tilesX, t0, n0);
        for (f8n_u32 kb = 0; kb < nk; kb++)
        {
            const f8n_u32 it = lt * nk + kb;
            f8n_mbar_wait(bars + 48 + (it & 1) * 8, (it >> 1) & 1);
            const float* xs = (const float*)(smem + F8N_STAGES * F8N_STAGE_BYTES + (it & 1) * F8N_SCALE_BYTES);
            const float* ws = xs + F8N_BM;
#if F8N_BLOCK_SCALED
            // UE8M0 scales: the exponent fields of the power-of-two f32 scales. A: lanes 4g, 4g + 2 give row g, 4g + 1, 4g + 3 row
            // g + 8 of a fragment; B: every lane of quad g column g.
            f8n_u32 xe[F8N_FM], we[F8N_FN];
#pragma unroll
            for (int i = 0; i < F8N_FM; i++)
            {
                xe[i] = __float_as_uint(xs[wm * F8N_TM + i * 16 + g + (c4 & 1) * 8]) >> 23;
            }
#pragma unroll
            for (int j = 0; j < F8N_FN; j++)
            {
                we[j] = __float_as_uint(ws[f8n_brow(wn, j) + g]) >> 23;
            }

#elif F8N_WUNI
            // One weight scale per 128 rows and K block (and per matrix of the pair): the promotion's factors are per row of x.
            float xw[F8N_FM][2][1 + DUAL];
#pragma unroll
            for (int i = 0; i < F8N_FM; i++)
            {
#pragma unroll
                for (int h = 0; h < 2; h++)
                {
                    const float xh = xs[wm * F8N_TM + i * 16 + g + h * 8];
#pragma unroll
                    for (int d = 0; d <= DUAL; d++)
                    {
                        xw[i][h][d] = xh * ws[d * (F8N_BN / 2)];
                    }
                }
            }

#else
            float xv[F8N_FM][2];
            float2 wv[F8N_FN];
#pragma unroll
            for (int i = 0; i < F8N_FM; i++)
            {
                xv[i][0] = xs[wm * F8N_TM + i * 16 + g];
                xv[i][1] = xs[wm * F8N_TM + i * 16 + g + 8];
            }
#pragma unroll
            for (int j = 0; j < F8N_FN; j++)
            {
#if DUAL
                wv[j] = *(const float2*)(ws + (j & 1) * (F8N_BN / 2) + wn * (F8N_TN / 2) + (j >> 1) * 8 + c4 * 2);
#else
                wv[j] = *(const float2*)(ws + wn * F8N_TN + j * 8 + c4 * 2);
#endif
            }

#endif
            const f8n_u32 s = it % F8N_STAGES;
            f8n_mbar_wait(bars + s * 8, (it / F8N_STAGES) & 1);
            const f8n_u32 sa = f8n_smem(smem + s * F8N_STAGE_BYTES), sb = sa + F8N_BM * 128;
            f8n_u32 b[4][F8N_FN][2];
#pragma unroll
            for (int kk = 0; kk < 4; kk++)
            {
#pragma unroll
                for (int j = 0; j < F8N_FN; j += 2)
                {
                    f8n_u32 r = f8n_brow(wn, j + (lane >> 4)) + (lane & 7);
                    f8n_ldsm4(sb + f8n_swz(r, kk * 2 + ((lane >> 3) & 1)), b[kk][j][0], b[kk][j][1], b[kk][j + 1][0], b[kk][j + 1][1]);
                }
            }

#if F8N_BLOCK_SCALED
#pragma unroll
            for (int i = 0; i < F8N_FM; i++)
            {
#pragma unroll
                for (int kk = 0; kk < 4; kk++)
                {
                    f8n_u32 a[4];
                    f8n_u32 r = wm * F8N_TM + i * 16 + (lane & 15);
                    f8n_ldsm4(sa + f8n_swz(r, kk * 2 + (lane >> 4)), a[0], a[1], a[2], a[3]);
#pragma unroll
                    for (int j = 0; j < F8N_FN; j++)
                    {
                        f8n_mma(acc[i][j], a, b[kk][j][0], b[kk][j][1], xe[i], we[j]);
                    }
                }
            }

#else
#pragma unroll
            for (int i = 0; i < F8N_FM; i++)
            {
                float part[F8N_FN][4];
#pragma unroll
                for (int j = 0; j < F8N_FN; j++)
                {
#pragma unroll
                    for (int e = 0; e < 4; e++)
                    {
                        part[j][e] = 0.0f;
                    }
                }
#pragma unroll
                for (int kk = 0; kk < 4; kk++)
                {
                    f8n_u32 a[4];
                    f8n_u32 r = wm * F8N_TM + i * 16 + (lane & 15);
                    f8n_ldsm4(sa + f8n_swz(r, kk * 2 + (lane >> 4)), a[0], a[1], a[2], a[3]);
#pragma unroll
                    for (int j = 0; j < F8N_FN; j++)
                    {
                        f8n_mma(part[j], a, b[kk][j][0], b[kk][j][1]);
                    }
                }
#if F8N_WUNI
#pragma unroll
                for (int j = 0; j < F8N_FN; j++)
                {
                    const int d = DUAL ? (j & 1) : 0;
                    acc[i][j][0] = fmaf(part[j][0], xw[i][0][d], acc[i][j][0]);
                    acc[i][j][1] = fmaf(part[j][1], xw[i][0][d], acc[i][j][1]);
                    acc[i][j][2] = fmaf(part[j][2], xw[i][1][d], acc[i][j][2]);
                    acc[i][j][3] = fmaf(part[j][3], xw[i][1][d], acc[i][j][3]);
                }
#else
#pragma unroll
                for (int j = 0; j < F8N_FN; j++)
                {
                    acc[i][j][0] = fmaf(part[j][0], xv[i][0] * wv[j].x, acc[i][j][0]);
                    acc[i][j][1] = fmaf(part[j][1], xv[i][0] * wv[j].y, acc[i][j][1]);
                    acc[i][j][2] = fmaf(part[j][2], xv[i][1] * wv[j].x, acc[i][j][2]);
                    acc[i][j][3] = fmaf(part[j][3], xv[i][1] * wv[j].y, acc[i][j][3]);
                }
#endif
            }

#endif
            // Stage s and scale slot it % 2 are read (the fragments and scales are in registers).
            __syncwarp();
            if (lane == 0)
            {
                f8n_mbar_arrive(bars + 24 + s * 8);
            }
        }

#if QOUT
        // The tile's outputs (half lt % 2 of the pair) and each row's absolute maximum over them, combined across the quad and the
        // warp columns in rowMax.
        float out[F8N_FM][2][F8N_FN / 2][2];
#pragma unroll
        for (int i = 0; i < F8N_FM; i++)
        {
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
                float m = 0.0f;
#pragma unroll
                for (int j = 0; j < F8N_FN; j += 2)
                {
                    out[i][h][j / 2][0] = F8N_EPI2(acc[i][j][h * 2], acc[i][j + 1][h * 2]);
                    out[i][h][j / 2][1] = F8N_EPI2(acc[i][j][h * 2 + 1], acc[i][j + 1][h * 2 + 1]);
                    m = fmaxf(m, fmaxf(fabsf(out[i][h][j / 2][0]), fabsf(out[i][h][j / 2][1])));
                    acc[i][j][h * 2] = acc[i][j][h * 2 + 1] = acc[i][j + 1][h * 2] = acc[i][j + 1][h * 2 + 1] = 0.0f;
                }

                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                if (c4 == 0)
                {
                    atomicMax((unsigned int*)&rowMax[wm * F8N_TM + i * 16 + g + h * 8], __float_as_uint(m));
                }
            }
        }

        if ((lt & 1) == 0)
        {
#pragma unroll
            for (int i = 0; i < F8N_FM; i++)
            {
#pragma unroll
                for (int h = 0; h < 2; h++)
                {
#pragma unroll
                    for (int jj = 0; jj < F8N_FN / 2; jj++)
                    {
                        held[i][h][jj][0] = out[i][h][jj][0];
                        held[i][h][jj][1] = out[i][h][jj][1];
                    }
                }
            }

            continue;
        }

        // Both halves' maxima are in (the first half's atomics precede every warp's second), then read; the next pair's first
        // half cannot reach its atomics before every warp has passed this pair's stores (its K loop needs the stages they free).
        asm volatile("bar.sync 1, %0;\n" :: "r"(F8N_CONSUMERS) : "memory");
        float scale[F8N_FM][2];
#pragma unroll
        for (int i = 0; i < F8N_FM; i++)
        {
            scale[i][0] = f8n_qscale(rowMax[wm * F8N_TM + i * 16 + g]);
            scale[i][1] = f8n_qscale(rowMax[wm * F8N_TM + i * 16 + g + 8]);
        }

        asm volatile("bar.sync 1, %0;\n" :: "r"(F8N_CONSUMERS) : "memory");
        if (tid < F8N_BM)
        {
            rowMax[tid] = 0.0f;
        }

        const f8n_u32 pair0 = n0 - F8N_BOUT;
        f8n_u8* q = (f8n_u8*)p.y;
#pragma unroll
        for (int i = 0; i < F8N_FM; i++)
        {
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
                const f8n_u32 t = t0 + wm * F8N_TM + i * 16 + g + h * 8;
                if (t < p.T)
                {
                    f8n_u8* qr = q + (f8n_u64)t * F8N_QRS;
                    const float inv = scale[i][h] > 0.0f ? 1.0f / scale[i][h] : 0.0f;
#pragma unroll
                    for (int jj = 0; jj < F8N_FN / 2; jj++)
                    {
                        const f8n_u32 n = wn * (F8N_TN / 2) + jj * 8 + c4 * 2;
                        *(unsigned short*)(qr + pair0 + n) = f8n_e4m3x2(held[i][h][jj][0] * inv, held[i][h][jj][1] * inv);
                        *(unsigned short*)(qr + n0 + n) = f8n_e4m3x2(out[i][h][jj][0] * inv, out[i][h][jj][1] * inv);
                    }

                    if (wn == 0 && c4 == 0)
                    {
                        *(float*)(qr + N + (pair0 / 128) * 4) = scale[i][h];
                        *(float*)(q + (f8n_u64)QCAP * F8N_QRS + ((f8n_u64)(pair0 / 128) * QCAP + t) * 4) = scale[i][h];
                    }
                }
            }
        }
#else
        // The output and its row stride (MULTI: the member the tile's columns lie in, dense rows of its width).
        float* yb = p.y;
        f8n_u32 ys = Y_STRIDE;
#if MULTI
        {
            f8n_u32 first, width;
            const f8n_u32 mi = f8n_member(n0, first, width);
            yb = mi == 0 ? p.y : p.y_1;
#if MULTI > 2
            yb = mi == 2 ? p.y_2 : yb;
#endif
            yb -= first;
            ys = width;
        }
#endif
#pragma unroll
        for (int i = 0; i < F8N_FM; i++)
        {
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
                const f8n_u32 t = t0 + wm * F8N_TM + i * 16 + g + h * 8;
                if (t < p.T)
                {
                    float* yr = yb + (f8n_u64)t * ys;
#if DUAL
#pragma unroll
                    for (int j = 0; j < F8N_FN; j += 2)
                    {
                        const f8n_u32 n = n0 + wn * (F8N_TN / 2) + (j / 2) * 8 + c4 * 2;
                        *(float2*)(yr + n) = make_float2(F8N_EPI2(acc[i][j][h * 2], acc[i][j + 1][h * 2]),
                            F8N_EPI2(acc[i][j][h * 2 + 1], acc[i][j + 1][h * 2 + 1]));
                    }
#else
#pragma unroll
                    for (int j = 0; j < F8N_FN; j++)
                    {
                        const f8n_u32 n = n0 + wn * F8N_TN + j * 8 + c4 * 2;
                        *(float2*)(yr + n) = make_float2(acc[i][j][h * 2], acc[i][j][h * 2 + 1]);
                    }
#endif
                }

#pragma unroll
                for (int j = 0; j < F8N_FN; j++)
                {
                    acc[i][j][h * 2] = 0.0f;
                    acc[i][j][h * 2 + 1] = 0.0f;
                }
            }
        }
#endif
    }
}
