# Model load times

Wall-clock time of `sharplight <model> -p hi -n 1` (GGUF) and `sharplight-onnxrun <model> --provider vulkan --runs 1` (ONNX),
including process start, on an RTX PRO 6000 (Vulkan), 32 logical CPUs, a GPU shared with other applications (so single numbers
carry +-20 % noise). Phase timings come from `SHARPLIGHT_LOAD_TIMING=1`.

- **Cold**: kernel cache empty (a temporary `XDG_DATA_HOME`, so the user's `~/.local/share/sharplight/kernels` is untouched; the
  driver's own shader cache stays warm because the SPIR-V is byte-identical to earlier runs).
- **Warm**: kernel cache populated; best of three runs.

## Before and after (wall seconds)

| Model | Cold before | Cold after | Warm before | Warm after |
|---|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 | 10.5 | 2.4 | 0.94 | 0.65 |
| Gemma-3-1b Q8_0 | 10.4 | 2.5 | 0.89 | 0.77 |
| Qwen3.8-27B Q8_0 | 20.7 | 7.4 | 5.0 | 4.7 to 5.7 (upload bound, see below) |
| ResNet-50 (ONNX) | 25.2 | 4.3 | 0.85 | 0.53 |
| YOLOv8n (ONNX) | 56.4 | 8.1 | 0.70 | 0.43 |
| ViT-base (ONNX) | 8.8 | 2.2 | 1.01 | 1.0 (upload 0.45 s -> 0.16 s when idle) |
| ConvNeXt-tiny (ONNX) | 17.1 | 3.6 | 0.75 | 0.77 |
| DenseNet-9 (ONNX) | 79.9 | 11.9 | - | - |
| U2-Net-P (ONNX) | 64.7 | 11.5 | - | - |

## Phases (before -> after, ms)

| Model, cold | import | lowering | Slang (summed CPU) | waiting for kernels | upload | first run |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 377 -> 162 | 8998 -> 87 | 9300 -> 1760 (8 threads) | in lowering -> 378 | 194 -> 405 | 44 -> 8 |
| Gemma-3-1b | 146 -> 150 | 9456 -> 70 | 9900 -> 1920 | in lowering -> 398 | 345 -> 342 | 28 -> 10 |
| Qwen3.8-27B | 302 -> 194 | 13319 -> 171 | 19100 -> 5600 | in lowering -> 6 | 5569 -> 5165 | 54 -> 83 |
| ResNet-50 | 127 -> 135 | 24571 -> 55 | 24700 -> 3800 | in lowering -> 3621 | 105 -> 107 | 6.6 -> 1.8 |
| YOLOv8n | 38 -> 46 | 55604 -> 48 | 55600 -> 7700 | in lowering -> 7621 | 12 -> 10 | 3.4 -> 3.2 |

"Lowering" before includes the serial Slang compiles; after, lowering only emits kernel requests and Slang runs on worker
threads. For the 27B model the compilation is completely hidden behind the 28.6 GB weight upload.

Warm phases, e.g. Qwen3-0.6B: lowering 250 -> 70 ms (library sources and their hashes are computed once per process instead of
per launch), kernel/pipeline wait about 3 ms, pipelines are created in parallel before the first run (first run 20 -> 9 ms).

## What changed

- `SlangCompiler` no longer serializes compiles behind a process-wide lock. `Compile` returns a `CompiledKernel` immediately;
  its `Code` waits for a bounded pool of worker threads (`SlangWorkerPool`), each with its own Slang global session (a global
  session is not thread-safe). Identical requests are de-duplicated while in flight. Degree of parallelism: half the
  processors, 2 to 8 (`SHARPLIGHT_COMPILE_THREADS`); more threads slowed the weight upload that overlaps the compilation (27B:
  16 threads 11.3 s, 8 threads 7.0 s). Compile errors surface from `CompiledKernel.Code` (the Vulkan device rethrows them
  from `Compile`).
- Cache keys hash the kernel source plus the content of the library files it `#include`s (transitively), instead of the whole
  library, so editing one `.slang` file no longer invalidates unrelated kernels. The embedded files are read and hashed once per
  process.
- Vulkan: pipelines are created in parallel (no global creation lock) while the weights upload, so the first execution
  creates nothing. The driver `VkPipelineCache` is persisted next to the kernel cache
  (`pipelines-<vendor>-<device uuid>-<driver>.bin`), which matters when the driver's own shader cache is cold: with
  `__GL_SHADER_DISK_CACHE=0`, ViT pipeline creation drops from 1205 ms to about 50 ms.
- Weights that are computed on first read (ONNX transposes, GGUF/HF re-layouts) are produced on four threads ahead of the copy
  loop (`IWeightSource.Materialize`): ViT upload 500 -> 160 ms.

## Verification

SPIR-V is byte-identical: for Qwen3-0.6B, Gemma-3-1b and ConvNeXt-tiny every kernel dumped with `SHARPLIGHT_DUMP_KERNELS` was
compiled before and after and the (source+macros) -> SPIR-V mapping compared; DenseNet-9 and U2-Net-P request exactly the
same 221 / 127 kernel sources. Vulkan, Vision, Families, Package and Serving test suites pass.

## Why some conv models are slow cold (DenseNet-9, U2-Net-P, and per the ONNX survey PP-OCRv5, ResNet101-DUC, Zipformer)

Conv kernels are specialised with per-layer macros (`C`, `H`, `W`, `O`, `HO`, `WO`, `XS_N`, `XS_C`, `ROW_BYTES`, `SPLITK`, `KPS`
...). DenseNet-9's channel count grows with every dense layer, so its 87 conv kernels are all distinct (67 `conv_gemm` + 20
split-K reduces), plus 129 distinct fused elementwise kernels: 221 Slang compilations of about 380 ms each (80 s serial). The
kernels cannot be de-duplicated by the cache because they genuinely differ. Parallel compilation brings DenseNet-9 to 11.9 s
and U2-Net-P to 11.5 s on this machine (about 1/6.7), and warm loads of such models stay below one second.
Further reduction needs the kernels themselves to change (spatial extents and strides as push constants instead of macros,
which would change the generated code and possibly runtime speed, and touches the Shaders lowering owned by other work), so
it is not done here.

## Not changed

Import and graph build take 30 to 300 ms and warm lowering is now about 70 ms, so there was no worthwhile per-load result to
cache. The 27B weight upload (28.6 GB, 3 to 5 s, bound by the repacking copy from the mapped file into BAR staging memory)
dominates its warm load; a plain host-memory staging buffer was tried and was not faster.
