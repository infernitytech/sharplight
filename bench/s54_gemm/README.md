# Section 54 FP8 GEMM harness

`bench_fp8.cu` times the hand-written FP8 GEMM (`src/Sharplight.Shaders/Kernels/gemm_fp8_native.cu`, or any kernel file with the
same parameters in `KFILE`) on random E4M3 codes and scales in the layouts it reads: activation rows followed by the dense scale copy,
weights in 128-row groups (`-DOLD`: the per-row scale layout of the phase-3 kernel, e.g. `git show 911a999e:src/Sharplight.Shaders/Kernels/gemm_fp8_native.cu`).
It checks one row count against a reference kernel and evicts L2 before every timed launch (256 MB memset), so decode-sized
shapes stream their weights from memory as in a model. Build with the CUDA toolkit and pin the GPU:

```bash
K=gemm_fp8_native.cu   # path of the kernel file
for s in "4096 4096 0" "4096 1024 0" "4096 12288 1" "12288 4096 0"; do set -- $s
  nvcc -O3 -arch=sm_120 -std=c++17 -DK=$1 -DN=$2 -DDUAL=$3 -DKFILE="\"$K\"" bench_fp8.cu -o b && CUDA_VISIBLE_DEVICES=1 ./b
done
# decode / mixed sizes: -DT_MAX=512 -DCHECK_T=64 -DMS_A=16 -DMS_B=64 (two row counts) or -DMS_MID (80-384 rows),
# -DF8N_BM=64 (64-row tiles), -DSPLITK=n (grid z splits K; partial slices, summed by the check)
```

The ablations of section 54.3 removed the scale copies, the promotion or the code copies from a copy of the kernel; the
warp-specialized and larger-tile variants were edits of the same file (not kept).
