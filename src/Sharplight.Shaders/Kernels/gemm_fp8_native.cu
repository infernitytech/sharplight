// FP8 GEMM written in CUDA C++ for the CUDA provider (DeviceFeatures.NativeGemmFp8, compute capability 8.9 and later):
// y[t, n] = sum_k x[t, k] * W[n, k] with E4M3 activations and weights in 128-value scale groups along K. Compiled by NVRTC from
// this file, not by Slang: it uses what the Slang kernels cannot express on CUDA, cp.async copies into a three-stage ring of
// 32 KB K blocks in dynamic shared memory (96 KB, above the 48 KB static limit), with a 128-byte XOR swizzle so that ldmatrix
// reads and the copies are free of bank conflicts.
//
// Operands, both with the scales of a K block for consecutive rows next to each other, so that a tile's scales are a few
// 16-byte copies per K block instead of one 4-byte load from a separate line per row:
//  * x: the rows of quantize_fp8 ([K codes][K / 128 f32 scales] per row, row capacity XCAP), followed by the dense copy of the
//    scales that quantize_fp8 writes with DENSE_SCALES: [K / 128][XCAP] f32 at byte XCAP * (K + K / 32).
//  * W (and W2): the 128-row groups of WeightRepack.Fp8B128Grouped (f8e4m3_b128g): per group [128 rows of K codes][K / 128
//    blocks of 128 f32 scales], the same bytes per row as f8e4m3_b128r. Requires N % 128 == 0.
//
// Tile: F8N_BM (128 or 64) tokens x F8N_BN weight rows per block of F8N_WM x F8N_WN warps; a warp owns (F8N_BM / F8N_WM) x (F8N_BN / F8N_WN)
// outputs as m16n8k32 E4M3 fragments with f32 accumulation. Per 128-value K block the warp loads its B fragments once and
// then, one 16-row fragment at a time, sums the block's MMAs in a fresh accumulator and adds it scaled by
// sx[t, block] * sw[n, block] (one fused multiply-add per output per block, as gemm_fp8.slang).
//
// DUAL: two weight matrices of the same shape (w, w2; gate and up) share x; the B tile's 8-row groups alternate between them so
// each thread holds both results of an output and stores F8N_EPI2(a, b) (the generated epilogue); a block covers F8N_BN / 2
// outputs.
//
// MULTI (2 or 3, not with DUAL): the multi-output form over the concatenated rows of two or three matrices of one K (q / k / v), each
// with its own output (dense rows of its width): rows [0, MULTI_O1) are w's into y, [MULTI_O1, MULTI_O2) w_1's into y_1, the rest w_2's
// into y_2; a block's weight tile lies in one matrix (MULTI_O1, MULTI_O2 multiples of 128), so each output sums its K blocks as the
// matrix's own launch would.
//
// Specialization macros: K, N (all rows with MULTI), Y_STRIDE, XCAP, F8N_BM (default 128), F8N_BN, DUAL, F8N_EPI2 (DUAL), MULTI, MULTI_O1,
// MULTI_O2. Requires K % 512 == 0 (16-byte aligned rows of K + K / 32 bytes), N % 128 == 0, XCAP % 128 == 0 and an even Y_STRIDE. Grid:
// x = token tiles, y = weight tiles.
#ifndef DUAL
#define DUAL 0
#endif
#ifndef F8N_BM
#define F8N_BM 128
#endif
#ifndef MULTI
#define MULTI 0
#endif
#if MULTI && DUAL
#error "gemm_fp8_native: MULTI with DUAL"
#endif
// Weight rows per block: 128, or 64 for decode batches (twice the blocks for the same weights, so the last wave is shorter).
#ifndef F8N_BN
#define F8N_BN 128
#endif
// 64-row tiles (batches of 65 to a few hundred rows: twice the blocks) keep four stages in the same 99 KB.
#if F8N_BM == 64
#define F8N_STAGES 4
#else
#define F8N_STAGES 3
#endif
#define F8N_WM 2
#define F8N_WN 4
#define F8N_THREADS (F8N_WM * F8N_WN * 32)
#define F8N_TM (F8N_BM / F8N_WM)
#define F8N_TN (F8N_BN / F8N_WN)
#define F8N_FM (F8N_TM / 16)
#define F8N_FN (F8N_TN / 8)
#define F8N_STAGE_BYTES ((F8N_BM + F8N_BN) * 128 + (F8N_BM + F8N_BN) * 4)
#define F8N_RS ((unsigned long long)K + K / 32)
#define F8N_GROUP_BYTES (128ull * F8N_RS)
#if DUAL
#define F8N_BOUT (F8N_BN / 2)
#else
#define F8N_BOUT F8N_BN
#endif

typedef unsigned char f8n_u8;
typedef unsigned int f8n_u32;
typedef unsigned long long f8n_u64;

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
};

static __device__ __forceinline__ f8n_u32 f8n_smem(const void* p)
{
    return (f8n_u32)__cvta_generic_to_shared(p);
}

static __device__ __forceinline__ void f8n_cp16(f8n_u32 dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}

static __device__ __forceinline__ void f8n_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int Pending>
static __device__ __forceinline__ void f8n_wait()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Pending));
}

static __device__ __forceinline__ void f8n_ldsm4(f8n_u32 addr, f8n_u32& r0, f8n_u32& r1, f8n_u32& r2, f8n_u32& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

static __device__ __forceinline__ void f8n_mma(float* c, const f8n_u32* a, f8n_u32 b0, f8n_u32 b1)
{
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Byte offset of 16-byte chunk c of row r in a stage tile of 128-byte rows (chunk c ^ (r % 8): conflict-free ldmatrix and copies).
static __device__ __forceinline__ f8n_u32 f8n_swz(f8n_u32 r, f8n_u32 c)
{
    return r * 128u + ((c ^ (r & 7u)) << 4);
}

// Codes of weight row n in the 128-row groups.
static __device__ __forceinline__ const f8n_u8* f8n_wcodes(const f8n_u8* w, f8n_u32 n)
{
    return w + (n >> 7) * F8N_GROUP_BYTES + (f8n_u64)(n & 127) * K;
}

// Scale of weight row n for K block kb (the 128 rows of a group are adjacent per block).
static __device__ __forceinline__ const f8n_u8* f8n_wscales(const f8n_u8* w, f8n_u32 n, f8n_u32 kb)
{
    return w + (n >> 7) * F8N_GROUP_BYTES + 128ull * K + kb * 512u + (n & 127) * 4u;
}

// Weight matrix and row of B tile row r of the block's matrix w (rows past its n rows are clamped: their outputs are never stored).
static __device__ __forceinline__ const f8n_u8* f8n_wrow(const F8nParams& p, const f8n_u8* w, f8n_u32 n, f8n_u32 n0, f8n_u32 r)
{
#if DUAL
    w = ((r >> 3) & 1) ? p.w2 : w;
    f8n_u32 row = min(n0 + (r >> 4) * 8 + (r & 7), n - 1);
#else
    f8n_u32 row = min(n0 + r, n - 1);
#endif
    return f8n_wcodes(w, row);
}

extern "C" __global__ void __launch_bounds__(F8N_THREADS, 1) gemm_fp8_native(F8nParams p)
{
    extern __shared__ __align__(128) f8n_u8 smem[];
    const f8n_u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const f8n_u32 wm = warp / F8N_WN, wn = warp % F8N_WN;
    // Blocks in groups of 16 token tiles that walk all weight tiles: the group's activations stay in L2 while the weights stream.
    const f8n_u32 id = blockIdx.y * gridDim.x + blockIdx.x, groupBlocks = 16 * gridDim.y;
    const f8n_u32 first = id / groupBlocks * 16, groupRows = min(gridDim.x - first, 16u);
    const f8n_u32 t0 = (first + id % groupBlocks % groupRows) * F8N_BM;
    f8n_u32 n0 = id % groupBlocks / groupRows * F8N_BOUT;
    // The block's matrix, output, outputs and output row stride (MULTI: the member its weight tile lies in).
    const f8n_u8* wb = p.w;
    float* yb = p.y;
    f8n_u32 nb = N, ys = Y_STRIDE;
#if MULTI
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
    const f8n_u32 nk = K / 128;
    const f8n_u8* sx = p.x + (f8n_u64)XCAP * F8N_RS;

    // Copies of K block kb into stage s: codes of both tiles (16 bytes per thread and item), then the scales: 128 of x and the
    // tile's 128 weight scales (DUAL: 64 of w, then 64 of w2), 16 bytes per copy.
    auto load = [&](f8n_u32 s, f8n_u32 kb) {
        f8n_u32 base = f8n_smem(smem + s * F8N_STAGE_BYTES);
        const f8n_u32 k0 = kb * 128;
#pragma unroll
        for (f8n_u32 i = 0; i < F8N_BM * 8 / F8N_THREADS; i++)
        {
            f8n_u32 idx = tid + i * F8N_THREADS, r = idx >> 3, c = idx & 7;
            f8n_u32 t = min(t0 + r, p.T - 1);
            f8n_cp16(base + f8n_swz(r, c), p.x + (f8n_u64)t * F8N_RS + k0 + c * 16);
        }
#pragma unroll
        for (f8n_u32 i = 0; i < F8N_BN * 8 / F8N_THREADS; i++)
        {
            f8n_u32 idx = tid + i * F8N_THREADS, r = idx >> 3, c = idx & 7;
            f8n_cp16(base + F8N_BM * 128 + f8n_swz(r, c), f8n_wrow(p, wb, nb, n0, r) + k0 + c * 16);
        }
        f8n_u32 scales = base + (F8N_BM + F8N_BN) * 128;
        if (tid < F8N_BM / 4)
        {
            f8n_cp16(scales + tid * 16, sx + ((f8n_u64)kb * XCAP + t0 + tid * 4) * 4);
        }
        else if (tid < (F8N_BM + F8N_BN) / 4)
        {
            f8n_u32 q = tid - F8N_BM / 4;
#if DUAL
            const f8n_u8* w = q < F8N_BN / 8 ? wb : p.w2;
            f8n_u32 n = n0 + (q % (F8N_BN / 8)) * 4;
#else
            const f8n_u8* w = wb;
            f8n_u32 n = n0 + q * 4;
#endif
            f8n_cp16(scales + F8N_BM * 4 + q * 16, f8n_wscales(w, n, kb));
        }
    };

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

#pragma unroll
    for (int s = 0; s < F8N_STAGES - 1; s++)
    {
        if (s < (int)nk)
        {
            load(s, s);
        }
        f8n_commit();
    }

    const f8n_u32 g = lane >> 2, c4 = lane & 3;
    for (f8n_u32 kb = 0; kb < nk; kb++)
    {
        f8n_wait<F8N_STAGES - 2>();
        __syncthreads();
        f8n_u32 next = kb + F8N_STAGES - 1;
        if (next < nk)
        {
            load(next % F8N_STAGES, next);
        }
        f8n_commit();

        const f8n_u8* stage = smem + (kb % F8N_STAGES) * F8N_STAGE_BYTES;
        const f8n_u32 sa = f8n_smem(stage), sb = sa + F8N_BM * 128;
        const float* xs = (const float*)(stage + (F8N_BM + F8N_BN) * 128);
        const float* ws = xs + F8N_BM;
        f8n_u32 b[4][F8N_FN][2];
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
        {
#pragma unroll
            for (int j = 0; j < F8N_FN; j += 2)
            {
                f8n_u32 r = wn * F8N_TN + j * 8 + (lane & 7) + ((lane >> 4) << 3);
                f8n_ldsm4(sb + f8n_swz(r, kk * 2 + ((lane >> 3) & 1)), b[kk][j][0], b[kk][j][1], b[kk][j + 1][0], b[kk][j + 1][1]);
            }
        }
        float wb[F8N_FN][2];
#pragma unroll
        for (int j = 0; j < F8N_FN; j++)
        {
            f8n_u32 r = wn * F8N_TN + j * 8 + c4 * 2;
#if DUAL
            // Tile row r: matrix (r / 8) % 2, output (r / 16) * 8 + r % 8 of the block's F8N_BN / 2.
            f8n_u32 at = ((r >> 3) & 1) * (F8N_BN / 2) + (r >> 4) * 8 + (r & 7);
#else
            f8n_u32 at = r;
#endif
            wb[j][0] = ws[at];
            wb[j][1] = ws[at + 1];
        }
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
            float x0 = xs[wm * F8N_TM + i * 16 + g], x1 = xs[wm * F8N_TM + i * 16 + g + 8];
#pragma unroll
            for (int j = 0; j < F8N_FN; j++)
            {
                acc[i][j][0] = fmaf(part[j][0], x0 * wb[j][0], acc[i][j][0]);
                acc[i][j][1] = fmaf(part[j][1], x0 * wb[j][1], acc[i][j][1]);
                acc[i][j][2] = fmaf(part[j][2], x1 * wb[j][0], acc[i][j][2]);
                acc[i][j][3] = fmaf(part[j][3], x1 * wb[j][1], acc[i][j][3]);
            }
        }
    }
    f8n_wait<0>();

#pragma unroll
    for (int i = 0; i < F8N_FM; i++)
    {
#pragma unroll
        for (int h = 0; h < 2; h++)
        {
            f8n_u32 t = t0 + wm * F8N_TM + i * 16 + g + h * 8;
            if (t >= p.T)
            {
                continue;
            }

            float* yr = yb + (f8n_u64)t * ys;
#if DUAL
#pragma unroll
            for (int j = 0; j < F8N_FN; j += 2)
            {
                f8n_u32 n = n0 + (wn * F8N_TN) / 2 + (j / 2) * 8 + c4 * 2;
                float v0 = F8N_EPI2(acc[i][j][h * 2], acc[i][j + 1][h * 2]);
                float v1 = F8N_EPI2(acc[i][j][h * 2 + 1], acc[i][j + 1][h * 2 + 1]);
#else
#pragma unroll
            for (int j = 0; j < F8N_FN; j++)
            {
                f8n_u32 n = n0 + wn * F8N_TN + j * 8 + c4 * 2;
                float v0 = acc[i][j][h * 2], v1 = acc[i][j][h * 2 + 1];
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
