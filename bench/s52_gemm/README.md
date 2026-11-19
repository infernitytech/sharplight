# Section 52 GEMM prototypes

Standalone harnesses behind docs/benchmarks.md §52.3-52.4 (random data, timing with events, a reference kernel for the error check).
Build with the CUDA toolkit (`nvcc`, sm_120 here) and run with `CUDA_VISIBLE_DEVICES` pinned:

```bash
# f16 GEMM (the form of src/Sharplight.Shaders/Kernels/gemm_f16_native.cu): 128 x 256 tiles, two stages
nvcc -O3 -arch=sm_120 -std=c++17 -DH_BM=128 -DH_BN=256 -DH_WM=2 -DH_WN=4 -DH_STAGES=2 bench_f16.cu -o bench_f16 && ./bench_f16
# FP8 variants (KHDR selects the kernel; F8N_* / H_* macros its tiles; NOMMA / NOLOAD / NOPROMO / NOSTORE / NORESCALE / NOSCALE ablations)
nvcc -O3 -arch=sm_120 -std=c++17 -DKHDR='"gemm_fp8b.cuh"' -DF8N_BN=128 -DF8N_WN=4 -DF8N_STAGES=3 bench_any.cu -lcuda -o fp8b && ./fp8b
nvcc -O3 -arch=sm_120 -std=c++17 -DKHDR='"gemm_fp8t.cuh"' -DF8N_TMA -DF8N_BN=128 -DF8N_WN=4 -DF8N_STAGES=3 bench_any.cu -lcuda -o fp8t && ./fp8t
nvcc -O3 -arch=sm_120 -std=c++17 -DKHDR='"gemm_fp8r.cuh"' -DH_BM=128 -DH_BN=256 -DH_WM=2 -DH_WN=4 -DH_STAGES=2 bench_any.cu -lcuda -o fp8r && ./fp8r
# mma.sync peak and ldmatrix + mma loop
nvcc -O3 -arch=sm_120 peak2.cu -o peak2 && ./peak2; nvcc -O3 -arch=sm_120 peak3.cu -o peak3 && ./peak3
```

`gemm_fp8b.cuh` is the form that became `gemm_fp8_native.cu`; `gemm_fp8t.cuh` the TMA / mbarrier form (tensor maps encoded by the
harness with `cuTensorMapEncodeTiled`); `gemm_fp8r.cuh` the f16 kernel's skeleton with E4M3 MMAs and an accumulator kept divided by
the current block's scale product.
