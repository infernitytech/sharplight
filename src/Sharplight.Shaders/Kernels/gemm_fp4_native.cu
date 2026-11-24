// FP4 GEMM written in CUDA C++ for the CUDA provider (DeviceFeatures.BlockScaledFp4, compute capability 12.x, compiled for the
// family-specific compute_XYf with SL_FAMILY_ARCH): y[t, n] = sum_k x[t, k] * W[n, k] with both operands in E2M1 codes and a
// UE4M3 scale per 16 values, multiplied by the block-scaled MMA (mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X m16n8k64:
// the tensor cores apply the 16-value scales of both operands and accumulate all of K in f32), times a per-row f32 factor of each
// operand (the activation row's global scale and the weight row's second-level scale) in the store.
//
// x: the FP4 activation rows of quantize_fp4.cu (per 256 values of K a 160-byte record [128 code bytes][16 UE4M3 scales][16 bytes
// padding], then 32 bytes with the row's f32 global scale; each 16-value group's code bytes hold value i in the low nibble of byte i
// and value i + 8 in the high one, the layout of the weights' sub-blocks). W as NVFP4 blocks { u8 d[4]; u8 qs[32] } (36 bytes per 64 values) or, with F4_GS,
//    NVFP4_GS blocks { f32 g; u8 d[4]; u8 qs[32] } (40 bytes; g is the same in every block of a row and is read once per row).
//    Requires K % 256 == 0.
//
// Tile: F4_BM tokens x F4_BN weight rows per block of F4_WM x F4_WN warps, F4_STAGES stages of one 256-value K step:
// [F4_BM rows of 128 code bytes, 16-byte chunks XOR-swizzled by row for ldmatrix][F4_BM rows of 32 scale bytes][F4_BN rows of
// 160 weight bytes]. Copies keep the two 16-byte halves of every 32-byte global sector in one 32-byte granule of shared memory
// (benchmarks.md section 97): activation records are sector-aligned, NVFP4_GS steps (160 bytes) too, and NVFP4 steps (144 bytes, which
// start on either half of a sector) land 16 bytes into their row when they start on the second half. A warp owns
// (F4_BM / F4_WM) x (F4_BN / F4_WN) outputs as m16n8k64 fragments: A by ldmatrix, B and both scale words by 32-bit shared loads.
//
// DUAL: two weight matrices of the same shape (w, w2; gate and up) share x; the B tile's 8-row groups alternate between them, so each
// thread holds both results of an output and stores F4N_EPI2(a, b) (the generated epilogue); a block covers F4_BN / 2 outputs.
//
// Activations are read with cp.async.cg (through L2, coherent with the launch that wrote them), the per-row factors with plain loads
// after griddepcontrol.wait; the weight codes are copied before the wait (no launch of a step writes them).
//
// F4_MOE: the expert-grouped form of a MatMulId with NVFP4 expert stacks [E, N, K] (EXPERT_BYTES per expert): each block computes one
// routing tile of moe_route (moe.slang; S slots, MAXTILES) against F4_BN weight rows of the tile's expert, reading the FP4 rows of its
// items' tokens (X_TOKEN: row item / S) or of the items themselves, and stores y[item, n].
//
// MULTI (2 or 3): the q / k / v group sharing x as one launch over the concatenated rows of its matrices (N all rows, MULTI_O1 /
// MULTI_O2 the members' first rows, multiples of F4_BN), each block's weight tile in one member, stored to that member's output.
//
// Specialization macros: K, N, Y_STRIDE, F4_BM, F4_BN, F4_WM, F4_WN, F4_STAGES, F4_GS, DUAL, F4N_EPI2 (DUAL), F4_MINB, F4_MOE (with
// EXPERT_BYTES, S, MAXTILES, X_TOKEN), MULTI, MULTI_O1, MULTI_O2. Grid: x = token tiles, y = weight tiles; F4_MOE: x = weight tiles,
// y = routing tiles (blocks past the tile count return).
#define SL_FAMILY_ARCH 1
#define SL_NVRTC_SASS 1

typedef unsigned char f4n_u8;
typedef unsigned short f4n_u16;
typedef unsigned int f4n_u32;
typedef unsigned long long f4n_u64;

// Bytes of one 256-value activation record and of an activation row.
#define F4_XREC 160
#define F4_XRB ((f4n_u64)(K / 256) * F4_XREC + 32)

#ifndef DUAL
#define DUAL 0
#endif
#ifndef F4_GS
#define F4_GS 0
#endif
#ifndef F4_MINB
#define F4_MINB 1
#endif
#ifndef F4_MOE
#define F4_MOE 0
#endif
#ifndef X_TOKEN
#define X_TOKEN 0
#endif
#ifndef MULTI
#define MULTI 0
#endif
#if MULTI && (DUAL || F4_MOE)
#error "gemm_fp4_native: MULTI with DUAL or F4_MOE"
#endif
#define F4_THREADS (F4_WM * F4_WN * 32)
#define F4_TM (F4_BM / F4_WM)
#define F4_TN (F4_BN / F4_WN)
#define F4_FM (F4_TM / 16)
#define F4_FN (F4_TN / 8)
// Weight block bytes, bytes of a 256-value step of a weight row, of a weight row, and the shared-memory stride of a B tile row.
#if F4_GS
#define F4_WBB 40
#else
#define F4_WBB 36
#endif
#define F4_WSTEP (4 * F4_WBB)
#define F4_WRB ((f4n_u64)(K / 64) * F4_WBB)
#define F4_BROW 160
#define F4_STAGE_BYTES ((F4_BM + F4_BN) * 160)
#if DUAL
#define F4_BOUT (F4_BN / 2)
#else
#define F4_BOUT F4_BN
#endif
#if F4_FM < 1 || F4_FN < 1 || (DUAL && F4_TN % 16 != 0)
#error "gemm_fp4_native: tile too small for the warps"
#endif

struct F4nParams
{
    const f4n_u8* x;
    const f4n_u8* w;
#if DUAL
    const f4n_u8* w2;
#endif
#if F4_MOE
    const f4n_u32* route;
#endif
    float* y;
#if MULTI
    const f4n_u8* w_1;
#if MULTI > 2
    const f4n_u8* w_2;
#endif
    float* y_1;
#if MULTI > 2
    float* y_2;
#endif
#endif
    f4n_u32 T;
};

static __device__ __forceinline__ f4n_u32 f4n_smem(const void* p)
{
    return (f4n_u32)__cvta_generic_to_shared(p);
}

static __device__ __forceinline__ void f4n_cp16(f4n_u32 dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}

static __device__ __forceinline__ void f4n_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int Pending>
static __device__ __forceinline__ void f4n_wait()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Pending));
}

static __device__ __forceinline__ void f4n_ldsm4(f4n_u32 addr, f4n_u32* r)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

static __device__ __forceinline__ f4n_u32 f4n_lds(f4n_u32 addr)
{
    f4n_u32 v;
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(addr));
    return v;
}

// c += A * B with the 16-value scales sa (this lane's row of A: lanes 4g + 0 / 4g + 1 give rows g / g + 8) and sb (lane 4n: column n).
static __device__ __forceinline__ void f4n_mma(float* c, const f4n_u32* a, f4n_u32 b0, f4n_u32 b1, f4n_u32 sa, f4n_u32 sb)
{
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

// Byte offset of 16-byte code chunk c of activation row r in a stage (chunk c ^ (r % 8): conflict-free ldmatrix; a sector's two
// chunks 2j, 2j + 1 stay a pair).
static __device__ __forceinline__ f4n_u32 f4n_swz(f4n_u32 r, f4n_u32 c)
{
    return r * 128u + ((c ^ (r & 7u)) << 4);
}

// Weight matrix row of B tile row r of the block's matrices w / w2 (rows past the matrix are clamped: their outputs are never stored).
static __device__ __forceinline__ const f4n_u8* f4n_wrow(const f4n_u8* w, const f4n_u8* w2, f4n_u32 n0, f4n_u32 r, f4n_u32 nb)
{
#if DUAL
    w = ((r >> 3) & 1) ? w2 : w;
    f4n_u32 row = min(n0 + (r >> 4) * 8 + (r & 7), nb - 1);
#else
    f4n_u32 row = min(n0 + r, nb - 1);
#endif
    return w + (f4n_u64)row * F4_WRB;
}

// The second-level scale of weight row n of w (1 for NVFP4).
static __device__ __forceinline__ float f4n_wscale(const f4n_u8* w, f4n_u32 n)
{
#if F4_GS
    return *(const float*)(w + (f4n_u64)n * F4_WRB);
#else
    return 1.0f;
#endif
}

extern "C" __global__ void __launch_bounds__(F4_THREADS, F4_MINB) gemm_fp4_native(F4nParams p)
{
    extern __shared__ __align__(128) f4n_u8 smem[];
    // Activation row read by each tile row, and output row it stores (~0u: none).
    __shared__ f4n_u32 xrows[F4_BM], yrows[F4_BM];
    const f4n_u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const f4n_u32 wm = warp / F4_WN, wn = warp % F4_WN;
    const f4n_u32 nk = K / 256;
#if F4_MOE
    // Grid x = weight tiles, y = routing tiles (moe_route: [0] tile count, [1 + 3i] {expert, offset, rows}, [1 + 3 MAXTILES + o] item).
    // The routing words come from the launch before, so this block waits for it before anything else.
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    if (blockIdx.y >= p.route[0])
    {
        return;
    }

    const f4n_u32 n0 = blockIdx.x * F4_BOUT, nb = N, ys = Y_STRIDE;
    float* const yb = p.y;
    const f4n_u32 expert = p.route[1 + 3 * blockIdx.y], offset = p.route[2 + 3 * blockIdx.y], tileRows = p.route[3 + 3 * blockIdx.y];
    const f4n_u8* wb = p.w + (f4n_u64)expert * EXPERT_BYTES;
#if DUAL
    const f4n_u8* wb2 = p.w2 + (f4n_u64)expert * EXPERT_BYTES;
#else
    const f4n_u8* wb2 = wb;
#endif
    for (f4n_u32 r = tid; r < F4_BM; r += F4_THREADS)
    {
        f4n_u32 item = p.route[1 + 3 * MAXTILES + offset + min(r, tileRows - 1)];
        xrows[r] = X_TOKEN ? item / S : item;
        yrows[r] = r < tileRows ? item : ~0u;
    }
#else
    // Blocks in groups of 16 token tiles that walk all weight tiles: the group's activations stay in L2 while the weights stream.
    const f4n_u32 id = blockIdx.y * gridDim.x + blockIdx.x, groupBlocks = 16 * gridDim.y;
    const f4n_u32 first = id / groupBlocks * 16, groupRows = min(gridDim.x - first, 16u);
    const f4n_u32 t0 = (first + id % groupBlocks % groupRows) * F4_BM;
    f4n_u32 n0 = id % groupBlocks / groupRows * F4_BOUT, nb = N, ys = Y_STRIDE;
    const f4n_u8* wb = p.w;
    float* yb = p.y;
#if MULTI
    // The member matrix the block's weight tile lies in: rows [0, MULTI_O1) of w into y, [MULTI_O1, MULTI_O2) of w_1 into y_1, the
    // rest of w_2 into y_2 (dense rows of each member's width).
    if (n0 >= MULTI_O1)
    {
#if MULTI > 2
        const bool third = n0 >= MULTI_O2;
        wb = third ? p.w_2 : p.w_1;
        yb = third ? p.y_2 : p.y_1;
        nb = third ? N - MULTI_O2 : MULTI_O2 - MULTI_O1;
        n0 -= third ? MULTI_O2 : MULTI_O1;
#else
        wb = p.w_1;
        yb = p.y_1;
        nb = N - MULTI_O1;
        n0 -= MULTI_O1;
#endif
    }
    else
    {
        nb = MULTI_O1;
    }
    ys = nb;
#endif
#if DUAL
    const f4n_u8* wb2 = p.w2;
#else
    const f4n_u8* wb2 = wb;
#endif
    for (f4n_u32 r = tid; r < F4_BM; r += F4_THREADS)
    {
        xrows[r] = min(t0 + r, p.T - 1);
        yrows[r] = t0 + r < p.T ? t0 + r : ~0u;
    }
#endif
    __syncthreads();

    // The weight bytes of step kb into stage s: 9 (NVFP4) or 10 (NVFP4_GS) 16-byte chunks per row. An NVFP4 step starting on the
    // second half of a sector lands 16 bytes into its row, so sector halves share a granule.
    auto loadW = [&](f4n_u32 s, f4n_u32 kb) {
        const f4n_u32 base = f4n_smem(smem + s * F4_STAGE_BYTES) + F4_BM * 160;
#if F4_GS
        constexpr f4n_u32 chunks = 10;
#else
        constexpr f4n_u32 chunks = 9;
#endif
#pragma unroll
        for (f4n_u32 i = 0; i < (F4_BN * chunks + F4_THREADS - 1) / F4_THREADS; i++)
        {
            f4n_u32 idx = tid + i * F4_THREADS;
            if (idx < F4_BN * chunks)
            {
                f4n_u32 r = idx / chunks, c = idx % chunks;
                const f4n_u8* src = f4n_wrow(wb, wb2, n0, r, nb) + (f4n_u64)kb * F4_WSTEP;
#if F4_GS
                f4n_u32 shift = 0;
#else
                f4n_u32 shift = (f4n_u32)((f4n_u64)src >> 4) & 1u;
#endif
                f4n_cp16(base + r * F4_BROW + (shift + c) * 16, src + c * 16);
            }
        }
    };
    // The activation record of step kb into stage s: 8 code chunks (swizzled) and the 32 bytes of scales and padding.
    auto loadX = [&](f4n_u32 s, f4n_u32 kb) {
        const f4n_u32 base = f4n_smem(smem + s * F4_STAGE_BYTES);
#pragma unroll
        for (f4n_u32 i = 0; i < (F4_BM * 10 + F4_THREADS - 1) / F4_THREADS; i++)
        {
            f4n_u32 idx = tid + i * F4_THREADS;
            if (idx < F4_BM * 10)
            {
                f4n_u32 r = idx / 10, c = idx % 10;
                f4n_u32 t = xrows[r];
                const f4n_u8* src = p.x + (f4n_u64)t * F4_XRB + (f4n_u64)kb * F4_XREC + c * 16;
                f4n_u32 dst = c < 8 ? base + f4n_swz(r, c) : base + F4_BM * 128 + r * 32 + (c - 8) * 16;
                f4n_cp16(dst, src);
            }
        }
    };

    float acc[F4_FM][F4_FN][4];
#pragma unroll
    for (int i = 0; i < F4_FM; i++)
    {
#pragma unroll
        for (int j = 0; j < F4_FN; j++)
        {
#pragma unroll
            for (int e = 0; e < 4; e++)
            {
                acc[i][j][e] = 0.0f;
            }
        }
    }

    // The first stages' weight bytes are copied before the wait for the launch that wrote x (griddepcontrol.wait; a no-op unless this
    // one is its programmatic dependent); they join the first stage's commit group.
#pragma unroll
    for (int s = 0; s < F4_STAGES - 1; s++)
    {
        if (s < (int)nk)
        {
            loadW(s, s);
        }
    }
#if !F4_MOE
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
#pragma unroll
    for (int s = 0; s < F4_STAGES - 1; s++)
    {
        if (s < (int)nk)
        {
            loadX(s, s);
        }
        f4n_commit();
    }

    const f4n_u32 g = lane >> 2, c4 = lane & 3;
    // Shared-memory offsets of this lane's B rows within a stage's B tile (the NVFP4 shift is the same in every step of a row when the
    // row's bytes per step are a multiple of 32, and alternates otherwise; it is recomputed per step from the source address).
    for (f4n_u32 kb = 0; kb < nk; kb++)
    {
        f4n_wait<F4_STAGES - 2>();
        __syncthreads();
        f4n_u32 next = kb + F4_STAGES - 1;
        if (next < nk)
        {
            loadW(next % F4_STAGES, next);
            loadX(next % F4_STAGES, next);
        }
        f4n_commit();

        const f4n_u32 sa = f4n_smem(smem + (kb % F4_STAGES) * F4_STAGE_BYTES);
        const f4n_u32 sx = sa + F4_BM * 128, sb = sa + F4_BM * 160;
        f4n_u32 brow[F4_FN];
#pragma unroll
        for (int j = 0; j < F4_FN; j++)
        {
            f4n_u32 r = wn * F4_TN + j * 8 + g;
#if F4_GS
            brow[j] = sb + r * F4_BROW;
#else
            f4n_u32 shift = (f4n_u32)((f4n_u64)(f4n_wrow(wb, wb2, n0, r, nb) + (f4n_u64)kb * F4_WSTEP) >> 4) & 1u;
            brow[j] = sb + r * F4_BROW + shift * 16;
#endif
        }
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
        {
            f4n_u32 b[F4_FN][3];
#pragma unroll
            for (int j = 0; j < F4_FN; j++)
            {
#if F4_GS
                f4n_u32 block = brow[j] + kk * 40 + 4;
#else
                f4n_u32 block = brow[j] + kk * 36;
#endif
                b[j][0] = f4n_lds(block + 4 + c4 * 4);
                b[j][1] = f4n_lds(block + 20 + c4 * 4);
                b[j][2] = f4n_lds(block);
            }
#pragma unroll
            for (int i = 0; i < F4_FM; i++)
            {
                f4n_u32 a[4];
                f4n_u32 r = wm * F4_TM + i * 16 + (lane & 15);
                f4n_ldsm4(sa + f4n_swz(r, kk * 2 + (lane >> 4)), a);
                f4n_u32 scaleA = f4n_lds(sx + (wm * F4_TM + i * 16 + g + (c4 & 1) * 8) * 32 + kk * 4);
#pragma unroll
                for (int j = 0; j < F4_FN; j++)
                {
                    f4n_mma(acc[i][j], a, b[j][0], b[j][1], scaleA, b[j][2]);
                }
            }
        }
    }
    f4n_wait<0>();

    // The weight rows' second-level scales of this thread's outputs.
    float ws[F4_FN][2];
#pragma unroll
    for (int j = 0; j < F4_FN; j++)
    {
#if DUAL
        f4n_u32 n = n0 + (wn * F4_TN) / 2 + (j / 2) * 8 + c4 * 2;
        const f4n_u8* w = (j & 1) ? wb2 : wb;
#else
        f4n_u32 n = n0 + wn * F4_TN + j * 8 + c4 * 2;
        const f4n_u8* w = wb;
#endif
        ws[j][0] = f4n_wscale(w, min(n, nb - 1));
        ws[j][1] = f4n_wscale(w, min(n + 1, nb - 1));
    }

#pragma unroll
    for (int i = 0; i < F4_FM; i++)
    {
#pragma unroll
        for (int h = 0; h < 2; h++)
        {
            f4n_u32 r = wm * F4_TM + i * 16 + g + h * 8;
            f4n_u32 t = yrows[r];
            if (t == ~0u)
            {
                continue;
            }

            const float gx = *(const float*)(p.x + (f4n_u64)xrows[r] * F4_XRB + (f4n_u64)(K / 256) * F4_XREC);
            float* yr = yb + (f4n_u64)t * ys;
#if DUAL
#pragma unroll
            for (int j = 0; j < F4_FN; j += 2)
            {
                f4n_u32 n = n0 + (wn * F4_TN) / 2 + (j / 2) * 8 + c4 * 2;
                float v0 = F4N_EPI2(acc[i][j][h * 2] * (gx * ws[j][0]), acc[i][j + 1][h * 2] * (gx * ws[j + 1][0]));
                float v1 = F4N_EPI2(acc[i][j][h * 2 + 1] * (gx * ws[j][1]), acc[i][j + 1][h * 2 + 1] * (gx * ws[j + 1][1]));
#else
#pragma unroll
            for (int j = 0; j < F4_FN; j++)
            {
                f4n_u32 n = n0 + wn * F4_TN + j * 8 + c4 * 2;
                float v0 = acc[i][j][h * 2] * (gx * ws[j][0]), v1 = acc[i][j][h * 2 + 1] * (gx * ws[j][1]);
#endif
                if (n + 1 < nb)
                {
                    *(float2*)(yr + n) = make_float2(v0, v1);
                }
                else if (n < nb)
                {
                    yr[n] = v0;
                }
            }
        }
    }
}
