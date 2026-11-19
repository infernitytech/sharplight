// FP8 GEMM written in CUDA C++ for the CUDA provider (DeviceFeatures.NativeGemmFp8, compute capability 8.9 and later):
// y[t, n] = sum_k x[t, k] * W[n, k] with E4M3 activations and weights in 128-value scale groups along K, the operands of
// gemm_fp8.slang (quantize_fp8 rows: [K codes][K / 128 f32 scales]; weights in the split layout f8e4m3_b128r, the same row
// format). Compiled by NVRTC from this file, not by Slang: it uses what the Slang kernels cannot express on CUDA, cp.async
// copies into a three-stage ring of 32 KB K blocks in dynamic shared memory (96 KB, above the 48 KB static limit), with a 128-byte
// XOR swizzle so that ldmatrix reads and the copies are free of bank conflicts.
//
// Tile: F8N_BM tokens x F8N_BN weight rows per block of F8N_WM x F8N_WN warps; a warp owns (F8N_BM / F8N_WM) x (F8N_BN / F8N_WN)
// outputs as m16n8k32 E4M3 fragments with f32 accumulation. Per 128-value K block the warp loads its B fragments once and
// then, one 16-row fragment at a time, sums the block's MMAs in a fresh accumulator and adds it scaled by
// sx[t, block] * sw[n, block] (one fused multiply-add per output per block, as gemm_fp8.slang).
//
// DUAL: two weight matrices of the same shape (w, w2; gate and up) share x; the B tile's 8-row groups alternate between them so
// each thread holds both results of an output and stores F8N_EPI2(a, b) (the generated epilogue); a block covers F8N_BN / 2
// outputs.
//
// Specialization macros: K, N, Y_STRIDE, DUAL, F8N_EPI2 (DUAL). Requires K % 512 == 0 (16-byte aligned rows of K + K / 32
// bytes) and an even Y_STRIDE. Grid: x = token tiles, y = weight tiles.
#ifndef DUAL
#define DUAL 0
#endif
#define F8N_BM 128
#define F8N_BN 128
#define F8N_STAGES 3
#define F8N_WM 2
#define F8N_WN 4
#define F8N_THREADS (F8N_WM * F8N_WN * 32)
#define F8N_TM (F8N_BM / F8N_WM)
#define F8N_TN (F8N_BN / F8N_WN)
#define F8N_FM (F8N_TM / 16)
#define F8N_FN (F8N_TN / 8)
#define F8N_STAGE_BYTES ((F8N_BM + F8N_BN) * 128 + (F8N_BM + F8N_BN) * 4)
#define F8N_RS ((unsigned long long)K + K / 32)
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

static __device__ __forceinline__ void f8n_cp4(f8n_u32 dst, const void* src)
{
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst), "l"(src));
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

// Weight matrix and row of B tile row r (rows past N are clamped: their outputs are never stored).
static __device__ __forceinline__ const f8n_u8* f8n_wrow(const F8nParams& p, f8n_u32 n0, f8n_u32 r)
{
#if DUAL
    const f8n_u8* w = ((r >> 3) & 1) ? p.w2 : p.w;
    f8n_u32 n = min(n0 + (r >> 4) * 8 + (r & 7), (f8n_u32)N - 1);
#else
    const f8n_u8* w = p.w;
    f8n_u32 n = min(n0 + r, (f8n_u32)N - 1);
#endif
    return w + (f8n_u64)n * F8N_RS;
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
    const f8n_u32 n0 = id % groupBlocks / groupRows * F8N_BOUT;
    const f8n_u32 nk = K / 128;

    // Copies of K block kb into stage s: codes of both tiles (16 bytes per thread and item), then one scale per row.
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
            f8n_cp16(base + F8N_BM * 128 + f8n_swz(r, c), f8n_wrow(p, n0, r) + k0 + c * 16);
        }
        f8n_u32 scales = base + (F8N_BM + F8N_BN) * 128;
        for (f8n_u32 r = tid; r < F8N_BM + F8N_BN; r += F8N_THREADS)
        {
            const f8n_u8* row = r < F8N_BM ? p.x + (f8n_u64)min(t0 + r, p.T - 1) * F8N_RS : f8n_wrow(p, n0, r - F8N_BM);
            f8n_cp4(scales + r * 4, row + K + kb * 4);
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
            wb[j][0] = ws[wn * F8N_TN + j * 8 + c4 * 2];
            wb[j][1] = ws[wn * F8N_TN + j * 8 + c4 * 2 + 1];
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

            float* yr = p.y + (f8n_u64)t * Y_STRIDE;
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
                if (n + 1 < N)
                {
                    *(float2*)(yr + n) = make_float2(v0, v1);
                }
                else if (n < N)
                {
                    yr[n] = v0;
                }
            }
        }
    }
}
