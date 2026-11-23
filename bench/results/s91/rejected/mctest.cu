// Does cp.async.bulk (non-tensor) multicast work on this GPU? Cluster of 2, each rank multicasts 4 KB to both CTAs.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void __cluster_dims__(2,1,1) k(const unsigned* src, unsigned* out, int mode)
{
    __shared__ __align__(128) unsigned buf[2048];
    __shared__ __align__(8) unsigned long long bar;
    unsigned rank; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    unsigned b = (unsigned)__cvta_generic_to_shared(&bar), d = (unsigned)__cvta_generic_to_shared(buf);
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) buf[i] = 0;
    if (threadIdx.x == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(b)); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
    if (threadIdx.x == 0)
    {
        asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}" :: "r"(b), "r"(8192) : "memory");
        const unsigned* s = src + rank * 1024;
        if (mode == 1)
            asm volatile("{\n .reg .u64 g;\n cvta.to.global.u64 g, %1;\n cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [g], 4096, [%2], %3;\n}"
                :: "r"(d + rank * 4096), "l"(s), "r"(b), "h"((unsigned short)3) : "memory");
        else
        {   // no multicast: each loads both halves
            for (int h = 0; h < 2; h++)
                asm volatile("{\n .reg .u64 g;\n cvta.to.global.u64 g, %1;\n cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [g], 4096, [%2];\n}"
                    :: "r"(d + h * 4096), "l"(src + h * 1024), "r"(b) : "memory");
        }
        unsigned ok = 0;
        for (long it = 0; it < 20000000 && !ok; it++)
            asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(b) : "memory");
        out[blockIdx.x * 4] = ok;
        unsigned bad = 0; for (int i = 0; i < 2048; i++) bad += buf[i] != src[i];
        out[blockIdx.x * 4 + 1] = bad;
    }
    asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}
int main()
{
    unsigned *s, *o; cudaMalloc(&s, 8192); cudaMalloc(&o, 64);
    unsigned h[2048]; for (int i = 0; i < 2048; i++) h[i] = i * 2654435761u + 1; cudaMemcpy(s, h, 8192, cudaMemcpyHostToDevice);
    for (int mode = 0; mode < 2; mode++)
    {
        cudaMemset(o, 0xff, 64); k<<<4, 128>>>(s, o, mode); cudaError_t e = cudaDeviceSynchronize();
        unsigned r[16]; cudaMemcpy(r, o, 64, cudaMemcpyDeviceToHost);
        printf("mode %d (%s): %s; blocks ok/bad: %u/%u %u/%u %u/%u %u/%u\n", mode, mode ? "multicast" : "plain", cudaGetErrorString(e), r[0], r[1], r[4], r[5], r[8], r[9], r[12], r[13]);
    }
    int n = 0; cudaLaunchConfig_t c = {}; c.gridDim = 82; c.blockDim = 384; c.dynamicSmemBytes = 99 * 1024;
    cudaLaunchAttribute a; a.id = cudaLaunchAttributeClusterDimension; a.val.clusterDim.x = 2; a.val.clusterDim.y = 1; a.val.clusterDim.z = 1; c.attrs = &a; c.numAttrs = 1;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024 - 8300);
    c.dynamicSmemBytes = 99 * 1024 - 8300;
    cudaError_t e = cudaOccupancyMaxActiveClusters(&n, (void*)k, &c);
    printf("max active clusters of 2 at ~99 KB: %d (%s)\n", n, cudaGetErrorString(e));
}
