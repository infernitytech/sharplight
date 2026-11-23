// Determinism harness for gemm_fp8_tma.cu's multi-output form: launches the kernel REPS times on fixed operands and compares
// every output bit with the first launch. CUBIN=path loads that machine code (the engine's NVRTC cubin) instead of nvcc's.
// PDL=1 launches with programmatic stream serialization, NOISE=1 puts a memory-bound kernel between the launches.
#pragma push_macro("K")
#pragma push_macro("N")
#undef K
#undef N
#include <cuda.h>
#pragma pop_macro("K")
#pragma pop_macro("N")
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include KFILE
#ifndef TROWS
#define TROWS 512
#endif
#ifndef REPS
#define REPS 200
#endif
static unsigned hash(unsigned x) { x *= 2654435761u; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15; return x; }
__global__ void noise(float* p, size_t n, float v) { for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = p[i] * 0.5f + v; }
static void grouped(std::vector<unsigned char>& b, size_t rows, unsigned seed)
{
    size_t rs = K + K / 32; b.assign(rows * rs, 0);
    for (size_t r = 0; r < rows; r++)
    {
        size_t g = r / 128 * 128 * rs, i = r % 128;
        for (int k = 0; k < K; k++) { unsigned x = hash((unsigned)(r * K + k) ^ seed); b[g + i * K + k] = (x & 0x80) | (0x28 + (x >> 8) % 0x20); }
        for (int kb = 0; kb < K / 128; kb++) { float s = getenv("WS1") ? 1.0f : 0.5f + hash((unsigned)((r / 128) * 1000 + kb) ^ (seed * 7)) % 1000 / 1000.0f; memcpy(&b[g + 128 * K + kb * 512 + i * 4], &s, 4); }
    }
}
int main()
{
    cudaFree(0);
    CUfunction fn = nullptr;
    const char* cub = getenv("CUBIN");
    if (cub)
    {
        FILE* f = fopen(cub, "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<char> b(n + 1, 0); fread(b.data(), 1, n, f); fclose(f);
        CUmodule m; CUresult r = cuModuleLoadData(&m, b.data()); if (r) { printf("load failed %d\n", (int)r); return 1; }
        cuModuleGetFunction(&fn, m, "gemm_fp8_tma");
    }
    else
    {
        cudaGetFuncBySymbol((cudaFunction_t*)&fn, (const void*)gemm_fp8_tma);
    }
    cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, F8N_SHARED);
    CUfunction refFn; cudaGetFuncBySymbol((cudaFunction_t*)&refFn, (const void*)gemm_fp8_tma);
    cuFuncSetAttribute(refFn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, F8N_SHARED);
    const int T = getenv("T") ? atoi(getenv("T")) : TROWS;
    size_t rs = K + K / 32;
    std::vector<unsigned char> hx(XCAP * rs + (size_t)(K / 128) * XCAP * 4, 0);
    for (int t = 0; t < XCAP; t++)
    {
        for (int k = 0; k < K; k++) { unsigned x = hash((unsigned)(t * K + (getenv("XREP") ? k % 128 : k)) * 3u + 11u); hx[t * rs + k] = (x & 0x80) | (0x28 + (x >> 8) % 0x20); }
        for (int kb = 0; kb < K / 128; kb++) { float s = getenv("XS1") ? 1.0f : 0.5f + hash(t * 977 + kb) % 1000 / 1000.0f; memcpy(&hx[t * rs + K + kb * 4], &s, 4); memcpy(&hx[XCAP * rs + ((size_t)kb * XCAP + t) * 4], &s, 4); }
    }
    std::vector<unsigned char> h0, h1; grouped(h0, MULTI_O1, 5); grouped(h1, N - MULTI_O1, 9);
    unsigned char *dx, *dw0, *dw1; float *y0, *y1, *nz;
    cudaMalloc(&dx, hx.size()); cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dw0, h0.size()); cudaMemcpy(dw0, h0.data(), h0.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dw1, h1.size()); cudaMemcpy(dw1, h1.data(), h1.size(), cudaMemcpyHostToDevice);
    size_t n0 = (size_t)XCAP * MULTI_O1, n1 = (size_t)XCAP * (N - MULTI_O1), nn = 64u << 20;
    cudaMalloc(&y0, n0 * 4); cudaMalloc(&y1, n1 * 4); cudaMalloc(&nz, nn * 4); cudaMemset(nz, 0, nn * 4);
    F8nParams p; memset(&p, 0, sizeof p);
    p.x = dx; p.w = dw0; p.w_1 = dw1; p.y = y0; p.y_1 = y1; p.T = T;
    auto enc = [](void* m, const void* base, int rank, cuuint64_t* dims, cuuint64_t* strides, cuuint32_t* box) {
        cuuint32_t es[3] = { 1, 1, 1 };
        CUresult r = cuTensorMapEncodeTiled((CUtensorMap*)m, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, (void*)base, dims, strides, box, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r) printf("encode failed %d\n", (int)r);
    };
    cuuint64_t xd[2] = { K, XCAP }, xs[1] = { rs }; cuuint32_t xb[2] = { 128, 128 };
    enc(&p.mx, dx, 2, xd, xs, xb);
    cuuint64_t wd0[3] = { K, 128, MULTI_O1 / 128 }, wd1[3] = { K, 128, (N - MULTI_O1) / 128 }, wst[2] = { K, 128 * rs }; cuuint32_t wb[3] = { 128, 128, 1 };
    enc(&p.mw, dw0, 3, wd0, wst, wb); enc(&p.mw_1, dw1, 3, wd1, wst, wb);
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int tiles = (T + 127) / 128 * (N / 128); unsigned grid = tiles < sms ? tiles : sms;
    bool pdl = getenv("PDL") && atoi(getenv("PDL")), noiseOn = getenv("NOISE") && atoi(getenv("NOISE"));
    cudaStream_t st; cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    auto launch = [&](CUfunction f) {
        void* args[] = { &p };
        CUlaunchConfig cfg = {}; cfg.gridDimX = grid; cfg.gridDimY = 1; cfg.gridDimZ = 1; cfg.blockDimX = F8N_THREADS; cfg.blockDimY = 1; cfg.blockDimZ = 1;
        cfg.sharedMemBytes = F8N_SHARED; cfg.hStream = (CUstream)st;
        CUlaunchAttribute at[1]; at[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION; at[0].value.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
        CUresult r = cuLaunchKernelEx(&cfg, f, args, nullptr); if (r) { printf("launch failed %d\n", (int)r); exit(1); }
    };
    std::vector<float> r0(n0), r1(n1), a0(n0), a1(n1);
    int bad = 0;
    const int reps = getenv("REPS") ? atoi(getenv("REPS")) : REPS;
    for (int rep = 0; rep < reps; rep++)
    {
        cudaMemsetAsync(y0, 0xff, n0 * 4, st); cudaMemsetAsync(y1, 0xff, n1 * 4, st);
        if (noiseOn) noise<<<sms * 4, 256, 0, st>>>(nz, nn, (float)rep);
        launch(rep == 0 ? refFn : fn);
        if (noiseOn) noise<<<sms * 4, 256, 0, st>>>(nz, nn, (float)rep);
        cudaError_t e = cudaStreamSynchronize(st); if (e) { printf("error %s\n", cudaGetErrorString(e)); return 1; }
        std::vector<float>& o0 = rep ? a0 : r0; std::vector<float>& o1 = rep ? a1 : r1;
        cudaMemcpy(o0.data(), y0, (size_t)T * MULTI_O1 * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(o1.data(), y1, (size_t)T * (N - MULTI_O1) * 4, cudaMemcpyDeviceToHost);
        if (rep)
        {
            long d0 = 0, d1 = 0; long first = -1;
            for (size_t i = 0; i < (size_t)T * MULTI_O1; i++) if (memcmp(&a0[i], &r0[i], 4)) { d0++; if (first < 0) first = (long)i; }
            for (size_t i = 0; i < (size_t)T * (N - MULTI_O1); i++) if (memcmp(&a1[i], &r1[i], 4)) d1++;
            if (d0 || d1)
            {
                bad++;
                if (bad <= 6)
                {
                    printf("rep %d: %ld / %ld outputs differ\n", rep, d0, d1);
                    int shown = 0;
                    for (int m = 0; m < 2; m++)
                    {
                        size_t w = m ? N - MULTI_O1 : MULTI_O1; std::vector<float>& a = m ? a1 : a0; std::vector<float>& r = m ? r1 : r0;
                        for (size_t i = 0; i < (size_t)T * w && shown < 40; i++)
                            if (memcmp(&a[i], &r[i], 4)) { printf("   y%d row %zu col %zu: %.6g vs ref %.6g\n", m, i / w, i % w, a[i], r[i]); shown++; }
                    }
                }
            }
        }
    }
    if (getenv("TIME"))
    {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int w = 0; w < 200; w++) launch(fn);
        cudaEventRecord(e0, st);
        for (int w = 0; w < 1000; w++) launch(fn);
        cudaEventRecord(e1, st); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        printf("sustained %.4f ms per launch\n", ms / 1000);
    }
    printf("T=%d grid %u pdl %d noise %d: %d of %d repeats differ\n", T, grid, pdl, noiseOn, bad, reps - 1);
    return 0;
}
