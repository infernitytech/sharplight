// Read bandwidth ceiling: each thread streams 16-byte loads (UNROLL in flight) over a buffer of MB megabytes after an L2 flush by a
// read (FLUSHREAD) or by a memset (dirty lines written back during the timed kernel).
#include <cuda_runtime.h>
#include <cstdio>
#ifndef UNROLL
#define UNROLL 8
#endif
__global__ void rd(const uint4* p, size_t n, unsigned* out)
{
    unsigned acc = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride * UNROLL)
    {
        uint4 v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) { size_t j = i + u * stride; if (j < n) asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(p + j)); else v[u] = make_uint4(0,0,0,0); }
#pragma unroll
        for (int u = 0; u < UNROLL; u++) acc ^= v[u].x ^ v[u].w;
    }
    if (acc == 0x12345678) out[0] = acc;
}
int main()
{
    size_t mbs[] = { 17, 50, 100 };
    void* fl; cudaMalloc(&fl, 256u << 20);
    unsigned* out; cudaMalloc(&out, 4);
    for (size_t mb : mbs)
    {
        size_t bytes = mb << 20; uint4* p; cudaMalloc(&p, bytes); cudaMemset(p, 1, bytes);
        for (int blocksPerSm : { 2, 4, 8 })
        for (int flushRead : { 0, 1 })
        {
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float best = 1e9;
            for (int it = 0; it < 20; it++)
            {
                if (flushRead) rd<<<82 * 4, 256>>>((const uint4*)fl, (256u << 20) / 16, out); else cudaMemsetAsync(fl, it, 256u << 20);
                cudaEventRecord(a); rd<<<82 * blocksPerSm, 256>>>(p, bytes / 16, out); cudaEventRecord(b); cudaEventSynchronize(b);
                float ms; cudaEventElapsedTime(&ms, a, b); if (it > 2 && ms < best) best = ms;
            }
            printf("%3zu MB, %d blocks/SM, flush %s: %.1f us, %.0f GB/s\n", mb, blocksPerSm, flushRead ? "read " : "memset", best * 1e3, bytes / best / 1e6);
        }
        cudaFree(p);
    }
}
