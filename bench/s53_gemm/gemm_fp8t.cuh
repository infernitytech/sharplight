// Native FP8 GEMM prototype T: TMA (2D tensor maps, 128-byte swizzle) + mbarrier pipeline, per-row-fragment promotion.
// x: rows of [K codes][K/128 f32 scales] (stride K + K/32), w (and w2): same layout per weight row; y f32 [T, ldy].
#ifndef F8N_BM
#define F8N_BM 128
#endif
#ifndef F8N_BN
#define F8N_BN 256
#endif
#ifndef F8N_STAGES
#define F8N_STAGES 2
#endif
#ifndef F8N_WM
#define F8N_WM 2
#endif
#ifndef F8N_WN
#define F8N_WN 8
#endif
#ifndef F8N_DUAL
#define F8N_DUAL 0
#endif
#define F8N_THREADS (F8N_WM * F8N_WN * 32)
#define F8N_TM (F8N_BM / F8N_WM)
#define F8N_TN (F8N_BN / F8N_WN)
#define F8N_FM (F8N_TM / 16)
#define F8N_FN (F8N_TN / 8)
#define F8N_STAGE_BYTES ((F8N_BM + F8N_BN) * 128)
#define F8N_SHARED (F8N_STAGES * F8N_STAGE_BYTES + 1024 + 64)

typedef unsigned char f8n_u8;
typedef unsigned int f8n_u32;
typedef unsigned long long f8n_u64;

struct __align__(64) F8nTensorMap
{
    f8n_u64 opaque[16];
};

struct F8nParams
{
    F8nTensorMap mx;
    F8nTensorMap mw;
#if F8N_DUAL
    F8nTensorMap mw2;
#endif
    const f8n_u8* x;
    const f8n_u8* w;
#if F8N_DUAL
    const f8n_u8* w2;
#endif
    float* y;
    f8n_u32 T;
    f8n_u32 N;
    f8n_u32 K;
    f8n_u32 ldy;
};

static __device__ __forceinline__ f8n_u32 f8n_smem(const void* p)
{
    return (f8n_u32)__cvta_generic_to_shared(p);
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

static __device__ __forceinline__ void f8n_mbar_init(f8n_u32 bar, f8n_u32 count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count));
}

static __device__ __forceinline__ void f8n_mbar_arrive(f8n_u32 bar)
{
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}

static __device__ __forceinline__ void f8n_mbar_expect(f8n_u32 bar, f8n_u32 bytes)
{
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void f8n_mbar_wait(f8n_u32 bar, f8n_u32 parity)
{
    asm volatile("{\n .reg .pred p;\n F8N_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra F8N_WAIT_%=;\n}\n"
                 :: "r"(bar), "r"(parity) : "memory");
}

static __device__ __forceinline__ void f8n_tma(f8n_u32 dst, const F8nTensorMap* map, int c0, int c1, f8n_u32 bar)
{
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 :: "r"(dst), "l"((f8n_u64)map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

// 128-byte swizzle of the tensor maps: 16-byte chunk c of row r sits at chunk c ^ (r % 8) (tiles 1024-byte aligned).
static __device__ __forceinline__ f8n_u32 f8n_swz(f8n_u32 r, f8n_u32 c)
{
    return r * 128u + ((c ^ (r & 7u)) << 4);
}

#ifndef F8N_EPI2
#define F8N_EPI2(a, b) ((a) / (1.0f + __expf(-(a))) * (b))
#endif

// Row of B tile row r in its weight matrix (dual: the first half of the tile holds w rows, the second w2 rows).
#if F8N_DUAL
#define F8N_BOUT (F8N_BN / 2)
#else
#define F8N_BOUT F8N_BN
#endif

extern "C" __global__ void __launch_bounds__(F8N_THREADS, 1) gemm_fp8_native(const __grid_constant__ F8nParams p)
{
    extern __shared__ __align__(1024) f8n_u8 smem_raw[];
    f8n_u8* smem = (f8n_u8*)(((f8n_u64)smem_raw + 1023) & ~(f8n_u64)1023);
    const f8n_u32 tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const f8n_u32 wm = warp / F8N_WN, wn = warp % F8N_WN;
    const f8n_u32 t0 = blockIdx.x * F8N_BM;
    const f8n_u32 n0 = blockIdx.y * F8N_BOUT;
    const f8n_u32 K = p.K;
    const f8n_u64 rs = (f8n_u64)K + K / 32;
    const f8n_u32 nk = K / 128;
    const f8n_u32 bars = f8n_smem(smem + F8N_STAGES * F8N_STAGE_BYTES);
    // full[s] at bars + 8 s, empty[s] at bars + 32 + 8 s.
    if (tid == 0)
    {
        for (int s = 0; s < F8N_STAGES; s++)
        {
            f8n_mbar_init(bars + s * 8, 1);
            f8n_mbar_init(bars + 32 + s * 8, F8N_THREADS / 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    auto issue = [&](f8n_u32 kb) {
        f8n_u32 s = kb % F8N_STAGES;
        f8n_u32 dst = f8n_smem(smem + s * F8N_STAGE_BYTES);
        f8n_u32 bar = bars + s * 8;
        f8n_mbar_expect(bar, F8N_STAGE_BYTES);
        f8n_tma(dst, &p.mx, kb * 128, t0, bar);
#if F8N_DUAL
        f8n_tma(dst + F8N_BM * 128, &p.mw, kb * 128, n0, bar);
        f8n_tma(dst + (F8N_BM + F8N_BN / 2) * 128, &p.mw2, kb * 128, n0, bar);
#else
        f8n_tma(dst + F8N_BM * 128, &p.mw, kb * 128, n0, bar);
#endif
    };

    if (tid == 0)
    {
        for (f8n_u32 kb = 0; kb < F8N_STAGES - 1 && kb < nk; kb++)
        {
            issue(kb);
        }
    }

    float acc[F8N_FM][F8N_FN][4];
#pragma unroll
    for (int i = 0; i < F8N_FM; i++)
#pragma unroll
        for (int j = 0; j < F8N_FN; j++)
#pragma unroll
            for (int e = 0; e < 4; e++)
                acc[i][j][e] = 0.0f;

    const f8n_u32 g = lane >> 2, c4 = lane & 3;
    for (f8n_u32 kb = 0; kb < nk; kb++)
    {
        if (tid == 0)
        {
            f8n_u32 next = kb + F8N_STAGES - 1;
            if (next < nk)
            {
                if (next >= F8N_STAGES)
                {
                    // The slot was last used by K block next - STAGES: wait until every warp released it.
                    f8n_mbar_wait(bars + 32 + (next % F8N_STAGES) * 8, ((next / F8N_STAGES) - 1) & 1);
                }
                issue(next);
            }
        }
        const f8n_u32 s = kb % F8N_STAGES;
        f8n_mbar_wait(bars + s * 8, (kb / F8N_STAGES) & 1);
        const f8n_u32 sa = f8n_smem(smem + s * F8N_STAGE_BYTES), sb = sa + F8N_BM * 128;
        // Weight scales of this thread's columns for this block.
        float wb[F8N_FN][2];
#pragma unroll
        for (int j = 0; j < F8N_FN; j++)
        {
#pragma unroll
            for (int h = 0; h < 2; h++)
            {
#if F8N_DUAL
                f8n_u32 col = wn * (F8N_TN / 2) + (j % (F8N_FN / 2)) * 8 + c4 * 2 + h;
                const f8n_u8* wp = j < F8N_FN / 2 ? p.w : p.w2;
#else
                f8n_u32 col = wn * F8N_TN + j * 8 + c4 * 2 + h;
                const f8n_u8* wp = p.w;
#endif
                f8n_u32 n = min(n0 + col, p.N - 1);
                wb[j][h] = __ldg((const float*)(wp + (f8n_u64)n * rs + K) + kb);
            }
        }
        f8n_u32 b[4][F8N_FN][2];
#pragma unroll
        for (int kk = 0; kk < 4; kk++)
        {
#pragma unroll
            for (int j = 0; j < F8N_FN; j += 2)
            {
#if F8N_DUAL
                f8n_u32 r = (j < F8N_FN / 2 ? 0 : F8N_BN / 2) + wn * (F8N_TN / 2) + (j % (F8N_FN / 2)) * 8 + (lane & 7) + ((lane >> 4) << 3);
#else
                f8n_u32 r = wn * F8N_TN + j * 8 + (lane & 7) + ((lane >> 4) << 3);
#endif
                f8n_ldsm4(sb + f8n_swz(r, kk * 2 + ((lane >> 3) & 1)), b[kk][j][0], b[kk][j][1], b[kk][j + 1][0], b[kk][j + 1][1]);
            }
        }
#pragma unroll
        for (int i = 0; i < F8N_FM; i++)
        {
            f8n_u32 row = wm * F8N_TM + i * 16 + g;
            float x0 = __ldg((const float*)(p.x + (f8n_u64)min(t0 + row, p.T - 1) * rs + K) + kb);
            float x1 = __ldg((const float*)(p.x + (f8n_u64)min(t0 + row + 8, p.T - 1) * rs + K) + kb);
            float tmp[F8N_FN][4];
#pragma unroll
            for (int j = 0; j < F8N_FN; j++)
#pragma unroll
                for (int e = 0; e < 4; e++)
                    tmp[j][e] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < 4; kk++)
            {
                f8n_u32 a[4];
                f8n_u32 r = wm * F8N_TM + i * 16 + (lane & 15);
                f8n_ldsm4(sa + f8n_swz(r, kk * 2 + (lane >> 4)), a[0], a[1], a[2], a[3]);
#pragma unroll
                for (int j = 0; j < F8N_FN; j++)
                    f8n_mma(tmp[j], a, b[kk][j][0], b[kk][j][1]);
            }
#pragma unroll
            for (int j = 0; j < F8N_FN; j++)
            {
                acc[i][j][0] = fmaf(tmp[j][0], x0 * wb[j][0], acc[i][j][0]);
                acc[i][j][1] = fmaf(tmp[j][1], x0 * wb[j][1], acc[i][j][1]);
                acc[i][j][2] = fmaf(tmp[j][2], x1 * wb[j][0], acc[i][j][2]);
                acc[i][j][3] = fmaf(tmp[j][3], x1 * wb[j][1], acc[i][j][3]);
            }
        }
        __syncwarp();
        if (lane == 0)
        {
            f8n_mbar_arrive(bars + 32 + s * 8);
        }
    }

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
            float* yr = p.y + (f8n_u64)t * p.ldy;
#if F8N_DUAL
#pragma unroll
            for (int j = 0; j < F8N_FN / 2; j++)
            {
                f8n_u32 n = n0 + wn * (F8N_TN / 2) + j * 8 + c4 * 2;
                float v0 = F8N_EPI2(acc[i][j][h * 2], acc[i][j + F8N_FN / 2][h * 2]);
                float v1 = F8N_EPI2(acc[i][j][h * 2 + 1], acc[i][j + F8N_FN / 2][h * 2 + 1]);
#else
#pragma unroll
            for (int j = 0; j < F8N_FN; j++)
            {
                f8n_u32 n = n0 + wn * F8N_TN + j * 8 + c4 * 2;
                float v0 = acc[i][j][h * 2], v1 = acc[i][j][h * 2 + 1];
#endif
                if (n + 1 < p.N)
                {
                    *(float2*)(yr + n) = make_float2(v0, v1);
                }
                else if (n < p.N)
                {
                    yr[n] = v0;
                }
            }
        }
    }
}
