# Third-party notices: Sharplight.Cuda.Runtime

This package contains NVIDIA CUDA Toolkit files, redistributed unmodified under the NVIDIA CUDA Toolkit End User License Agreement
(`LICENSE-NVIDIA-CUDA.txt`, the license file of the archives they come from). Section 2.2 ("Distribution") of the EULA lists the
distributable files in Attachment A; these come from its components:

| File | Attachment A component | Source archive (NVIDIA redistributable server, CUDA 13.0 Update 2) |
| --- | --- | --- |
| `libnvrtc.so.13`, `libnvrtc-builtins.so.13.0` (linux-x64, linux-arm64) | NVIDIA Runtime Compilation Library (NVRTC): `libnvrtc.so`, `libnvrtc-builtins.so` | `cuda_nvrtc` 13.0.88 |
| `nvrtc64_130_0.dll`, `nvrtc-builtins64_130.dll` (win-x64) | NVIDIA Runtime Compilation Library (NVRTC): `nvrtc.dll`, `nvrtc-builtins.dll` | `cuda_nvrtc` 13.0.88 |
| `cuda_fp16.h`, `cuda_fp16.hpp`, `cuda_bf16.h`, `cuda_bf16.hpp`, `cuda_fp8.h`, `cuda_fp8.hpp` | CUDA Floating Point Type Headers | `cuda_cudart` 13.0.96 |

The files may only be used by applications built on Sharplight (EULA section 1.1.2, "Distribution Requirements"). The NVIDIA driver
(`libcuda.so.1`, `nvcuda.dll`) is not included: it comes with the driver installation.
