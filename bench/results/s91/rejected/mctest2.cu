// Tensor-copy multicast on this GPU? Cluster of 2, rank r copies rows [64 r, 64 r + 64) of a 128 x 128 byte tile to both CTAs.
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
struct __align__(64) Map { unsigned long long o[16]; };
__global__ void __cluster_dims__(2,1,1) k(const __grid_constant__ Map m, const unsigned char* src, unsigned* out, int mode)
{
    __shared__ __align__(1024) unsigned char buf[16384];
    __shared__ __align__(8) unsigned long long bar;
    unsigned rank; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    unsigned b = (unsigned)__cvta_generic_to_shared(&bar), d = (unsigned)__cvta_generic_to_shared(buf);
    if (threadIdx.x == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(b)); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
    asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
    if (threadIdx.x == 0)
    {
        asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}" :: "r"(b), "r"(16384) : "memory");
        if (mode == 1)
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(d + rank * 8192), "l"((unsigned long long)&m), "r"(0), "r"((int)rank * 64), "r"(b), "h"((unsigned short)3) : "memory");
        else
            for (int h = 0; h < 2; h++)
                asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                    :: "r"(d + h * 8192), "l"((unsigned long long)&m), "r"(0), "r"(h * 64), "r"(b) : "memory");
        unsigned ok = 0;
        for (long it = 0; it < 2000000 && !ok; it++)
            asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(b) : "memory");
        out[blockIdx.x * 4] = ok;
        unsigned bad = 0; for (int i = 0; i < 16384; i++) bad += buf[i] != src[i];
        out[blockIdx.x * 4 + 1] = bad;
    }
    asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}
int main()
{
    unsigned char* s; unsigned* o; cudaMalloc(&s, 16384); cudaMalloc(&o, 64);
    unsigned char h[16384]; for (int i = 0; i < 16384; i++) h[i] = (unsigned char)(i * 37 + 5); cudaMemcpy(s, h, 16384, cudaMemcpyHostToDevice);
    Map m; cuuint64_t dims[2] = { 128, 128 }, st[1] = { 128 }; cuuint32_t box[2] = { 128, 64 }, es[2] = { 1, 1 };
    CUresult r = cuTensorMapEncodeTiled((CUtensorMap*)&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, s, dims, st, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    printf("encode %d\n", (int)r);
    for (int mode = 0; mode < 2; mode++)
    {
        cudaMemset(o, 0xff, 64); k<<<4, 128>>>(m, s, o, mode); cudaError_t e = cudaDeviceSynchronize();
        unsigned q[16]; cudaMemcpy(q, o, 64, cudaMemcpyDeviceToHost);
        printf("mode %d (%s): %s; blocks ok/bad: %u/%u %u/%u %u/%u %u/%u\n", mode, mode ? "tensor multicast" : "tensor plain", cudaGetErrorString(e), q[0], q[1], q[4], q[5], q[8], q[9], q[12], q[13]);
    }
}
