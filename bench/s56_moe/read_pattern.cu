// Read-pattern test of the MXFP4R expert layout (docs/benchmarks.md section 64): 1440 blocks of 128 weight rows (16-row groups of
// 16 x 1440 code bytes + 1440 scale bytes, gpt-oss-20b K = 2880), each copying C code bytes per row and stage with cp.async into a
// ring of S shared stages (PF=1: with the L2::256B prefetch hint), one barrier per stage, as gemm_moe_native.cu does.
// nvcc -O3 -arch=sm_120 -DC=32 -DS=4 read_pattern.cu -o rp && ./rp
#include <cstdio>
#ifndef C
#define C 32
#endif
#ifndef S
#define S 4
#endif
#ifndef PF
#define PF 1
#endif
#define ROWB 1530
#define ROWS 128
__global__ void __launch_bounds__(256) rdp(const unsigned char* w, unsigned* out)
{
    extern __shared__ __align__(128) unsigned char sm[];
    const unsigned char* base = w + (size_t)blockIdx.x * ROWS * ROWB;
    const int per = ROWS * C / 16 / 256 > 0 ? ROWS * C / 16 / 256 : 1;
    const int nk = 1440 / C;
    auto load = [&](int s, int kb) {
        for (int i = 0; i < per; i++) {
            int idx = threadIdx.x + i * 256; if (idx >= ROWS * C / 16) break;
            int r = idx / (C / 16), c = idx % (C / 16);
            unsigned dst = (unsigned)__cvta_generic_to_shared(sm + s * ROWS * C + idx * 16);
            const void* src = base + (size_t)(r / 16) * 16 * ROWB + (size_t)(r % 16) * 1440 + kb * C + c * 16;
#if PF
            asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
#else
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
#endif
        }
    };
    for (int s = 0; s < S - 1; s++) { if (s < nk) load(s, s); asm volatile("cp.async.commit_group;\n" ::); }
    unsigned acc = 0;
    for (int kb = 0; kb < nk; kb++) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(S - 2));
        __syncthreads();
        acc ^= *(unsigned*)(sm + (kb % S) * ROWS * C + threadIdx.x * 4 % (ROWS * C));
        int nx = kb + S - 1; if (nx < nk) load(nx % S, nx);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    if (acc == 0x12345678) out[0] = acc;
}
int main() {
    size_t blocks = 1440, bytes = blocks * ROWS * ROWB; unsigned char* w; cudaMalloc(&w, bytes); cudaMemset(w, 3, bytes);
    void* f; cudaMalloc(&f, 256 << 20); unsigned* o; cudaMalloc(&o, 4); int smem = S * ROWS * C;
    cudaFuncSetAttribute(rdp, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float best = 1e9;
    for (int i = 0; i < 20; i++) { cudaMemsetAsync(f, i, 256 << 20); cudaEventRecord(a); rdp<<<blocks, 256, smem>>>(w, o); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); if (i > 2 && ms < best) best = ms; }
    printf("C=%3d S=%d PF=%d smem %6d: %.3f ms %.0f GB/s (code bytes)\n", C, S, PF, smem, best, blocks * ROWS * 1440.0 / best / 1e6);
}
