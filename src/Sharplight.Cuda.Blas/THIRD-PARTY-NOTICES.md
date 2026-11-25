# Third-party notices: Sharplight.Cuda.Blas

This package contains an NVIDIA CUDA Toolkit file, redistributed unmodified under the NVIDIA CUDA Toolkit End User License Agreement
(`LICENSE-NVIDIA-CUDA.txt`, the license file of the archive it comes from). Section 2.2 ("Distribution") of the EULA lists the
distributable files in Attachment A; this one comes from its component "CUDA BLAS Library":

| File | Attachment A component | Source archive (NVIDIA redistributable server, CUDA 13.0 Update 2) |
| --- | --- | --- |
| `libcublasLt.so.13` (linux-x64, linux-arm64) | CUDA BLAS Library: `libcublasLt.so` | `libcublas` 13.1.0.3 |
| `cublasLt64_13.dll` (win-x64) | CUDA BLAS Library: `cublasLt.dll` | `libcublas` 13.1.0.3 |

The file may only be used by applications built on Sharplight (EULA section 1.1.2, "Distribution Requirements"). The NVIDIA driver
(`libcuda.so.1`, `nvcuda.dll`) is not included: it comes with the driver installation.
